use std::cmp::{max, min};
use std::collections::{HashSet, VecDeque};

use macroquad::prelude::*;
use macroquad::prelude::KeyCode::{Escape, Q, V};
use macroquad::rand::ChooseRandom;
use rayon::iter::ParallelIterator;
use rayon::prelude::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelSlice};

use utils::{Direction, Position};
use utils::Direction::{Down, Left, Right, Up};

use crate::brain::NeuralNetwork;
use crate::utils::{DIRECTIONS, draw_text_corner, find_closest_pos};

mod brain;
mod utils;

// game board size
const MAX_ROWS: usize = 100;
const MAX_COLUMNS: usize = 100;

// how many pieces of food to spawn at a time
const FOOD_COUNT: usize = 1;
// how many snakes to spawn per run of the simulation
const SNAKE_COUNT: usize = 1000;
// what ratio of the population to select for the next generation once a simulation run is complete
const SELECTION_RATE: f32 = 0.2;
// number of snakes to select for reproduction from each generation
const SAMPLE_SIZE: usize = (SNAKE_COUNT as f32 * SELECTION_RATE) as usize;
// how much score the snake is rewarded for eating a piece of food
const FOOD_REWARD: f32 = 10000.;
// snake dies of starvation if it doesn't get to food in this many simulation ticks.
// Prevents snakes looping around near food permanently as a winning strategy.
const MAX_TICKS_WITH_NO_FOOD: usize = 1000;
// how many generations to iterate over when training is triggered
const GENERATIONS_PER_TRAINING_RUN: usize = 200;
// spend at most this much time on a generation, to avoid infinitely long training steps
const MAX_GENERATION_DURATION_SECS: f32 = 4.;

// whether the snake will die if it collides with itself
const SELF_COLLISION_ENABLED: bool = true;
// reward the snake for getting close to food, even if it doesn't consume it
// this is done to encourage early generations to head towards the food, but can result in the
// wrong behavior being learned (i.e. snakes crawling around food without consuming it to
// avoid death and rack up score)
const REWARD_FOOD_PROXIMITY: bool = true;
// reduce the snake's final score if it dies prior to starving (eg by running into a wall or itself)
// this encourages snakes to be more careful
const PUNISH_FOR_CRASHING: bool = true;

#[derive(Clone)]
struct Snake {
    body: SnakeBody,
    network: NeuralNetwork,
    ticks_until_starvation: usize,
    score: f32,
    food: SnakeFood,
}

impl Snake {
    /// Creates a new snake using the supplied neural network. If not provided, spawns the snake
    /// with a randomly initialized NN.
    fn new(neural_network: Option<NeuralNetwork>) -> Self {
        Self {
            body: SnakeBody::new(),
            network: neural_network.unwrap_or_default(),
            score: 0.,
            ticks_until_starvation: MAX_TICKS_WITH_NO_FOOD,
            food: SnakeFood::new(),
        }
    }

    /// Predicts the snake's next direction given its knowledge about its environment and its
    /// neural network.
    ///
    /// # Arguments
    ///
    /// * `food_pos`: location of the food that the snake is trying to move towards
    ///
    /// returns: a direction for the snake to move in (doesn't necessarily change from current
    /// direction)
    fn predict_direction(&self, food_pos: Position) -> Direction {
        let head_position = self.body.get_head_position();

        // row and column diffs compared to food location, normalized for the board size
        let row_food_proximity = (head_position.row as f32 - food_pos.row as f32) / MAX_ROWS as f32;
        let col_food_proximity =
            (head_position.col as f32 - food_pos.col as f32) / MAX_COLUMNS as f32;

        // proximity to self or board borders, whichever is closer
        // used to warn the snake of impending collision with itself
        let mut left_boundary = 0;
        let mut right_boundary = MAX_COLUMNS;
        let mut up_boundary = 0;
        let mut down_boundary = MAX_ROWS;
        for chunk in self.body.chunks.iter() {
            if *chunk == head_position {
                // skip the head chunk when checking for impending collision,
                // we're only interested in the rest of the body
                continue;
            }
            if chunk.row == head_position.row {
                if chunk.col < head_position.col {
                    left_boundary = max(left_boundary, chunk.col);
                } else {
                    right_boundary = min(right_boundary, chunk.col);
                }
            } else if chunk.col == head_position.col {
                if chunk.row < head_position.row {
                    up_boundary = max(up_boundary, chunk.row);
                } else {
                    down_boundary = min(down_boundary, chunk.row);
                }
            }
        }

        let row_self_proximity =
            (head_position.row - up_boundary) as f32 / (down_boundary - up_boundary) as f32;
        let col_self_proximity =
            (head_position.col - left_boundary) as f32 / (right_boundary - left_boundary) as f32;

        // proximity to board borders (0..MAX_ROWS/COLUMNS), normalized for the board size
        // we could technically get away with not providing this information (because it's already
        // incorporated into self-proximity above), but I think having a stable representation of
        // the snake's position within the game world that doesn't haphazardly change as the snake
        // moves around is actually really important for helping it avoid colliding with itself
        // when near borders at greater body lengths
        let row_bounds_proximity = head_position.row as f32 / MAX_ROWS as f32;
        let col_bounds_proximity = head_position.col as f32 / MAX_COLUMNS as f32;

        // NOTE: inputs contain an encoding of the snake's direction. This is currently applied by
        // just indexing into the inputs array, so be careful changing the order of inputs here
        let mut input_vec = [
            // to be used for snake direction encoding
            0.,
            0.,
            0.,
            0.,
            // positional info
            row_food_proximity,
            col_food_proximity,
            row_self_proximity,
            col_self_proximity,
            row_bounds_proximity,
            col_bounds_proximity,
        ];

        // encode snake's current direction
        input_vec[self.body.direction as usize] = 1.;

        // ensure we normalized all inputs
        assert!(
            input_vec.iter().all(|i| -1. <= *i && *i <= 1.),
            "Input vec had non-normalized values: {:?}",
            input_vec
        );

        self.network.classify(&input_vec, &DIRECTIONS)
    }

    /// Paints the snake to the screen, using the provided grid cell/chunk sizes to calculate
    /// snake's body proportions
    fn draw(&self, chunk_width: f32, chunk_height: f32) {
        let snake_color = if self.body.dead { GRAY } else { GREEN };
        for chunk in self.body.chunks.iter() {
            let x = (chunk.col as f32) * chunk_width;
            let y = (chunk.row as f32) * chunk_height;
            draw_rectangle(x, y, chunk_width, chunk_height, snake_color);
        }

        self.food.draw(chunk_width, chunk_height);
    }

    /// Moves the snake in the direction it's currently facing
    /// Checks for collision with the target food
    /// Rewards/punishes/kills the snake depending on the state of things
    fn crawl(&mut self, target_food_position: Position) {
        if self.body.dead {
            // no-op
            return;
        }

        assert!(
            !self.body.chunks.is_empty(),
            "Snake should always have at least one chunk"
        );

        // starvation
        self.ticks_until_starvation -= 1;
        if self.ticks_until_starvation == 0 {
            self.body.dead = true;
            if PUNISH_FOR_CRASHING {
                // punishment for running into itself
                // TODO: separate reward from snake state, so we don't have to duplicate this logic
                self.score *= 0.2;
            }
            return;
        }

        let Position {
            col: previous_col,
            row: previous_row,
        } = self.body.get_head_position();

        // compute next head position based on snake's direction
        let (new_col, new_row) = match self.body.direction {
            Left => (previous_col.checked_sub(1), Some(previous_row)),
            Right => (previous_col.checked_add(1), Some(previous_row)),
            Up => (Some(previous_col), previous_row.checked_sub(1)),
            Down => (Some(previous_col), previous_row.checked_add(1)),
        };

        // mark the snake as dead if next col or row is out of bounds which means that the snake hit a wall
        if new_row.is_none()
            || new_col.is_none()
            || new_row.unwrap() > MAX_ROWS - 1
            || new_col.unwrap() > MAX_COLUMNS - 1
        {
            self.body.dead = true;
            if PUNISH_FOR_CRASHING {
                // punishment for running into a wall
                self.score *= 0.2;
            }
            return;
        }

        let new_position = Position::new(new_row.unwrap(), new_col.unwrap());

        // snake is also dead if the new position is in the existing chunk set which means that the snake collided with itself
        if SELF_COLLISION_ENABLED && self.body.chunk_set.contains(&new_position) {
            self.body.dead = true;
            return;
        }

        // if we reached food, we don't lose our tail, causing the snake to grow!
        // we do eat the food, however
        if target_food_position == new_position {
            // reward for food
            self.score += FOOD_REWARD;
            // no longer starving
            self.ticks_until_starvation = MAX_TICKS_WITH_NO_FOOD;
            // consume the food and spawn some more
            self.food.positions.remove(&target_food_position);
            self.food.refill();
        } else {
            // reward for being alive, proportional to how close to the food we got
            if REWARD_FOOD_PROXIMITY {
                let distance =
                    Vec2::from(new_position).distance_squared(target_food_position.into());
                self.score += 1. / distance;
            }

            // shrink the snake, removing its tail
            let chunk = self.body.chunks.pop_front().unwrap();
            self.body.chunk_set.remove(&chunk);
        }

        // grow the snake, appending its head
        self.body.chunk_set.insert(new_position);
        self.body.chunks.push_back(new_position);
    }

    /// Progresses the snake game by one frame/tick
    ///
    /// # Arguments
    ///
    /// * `should_respawn`: Whether to respawn the snake once it dies. Useful when checking out
    /// trained snake behavior.
    fn run_game_logic(&mut self, should_respawn: bool) {
        // locate the closest food item to attempt to move towards
        let target_food_position =
            find_closest_pos(self.body.get_head_position(), &self.food.positions);

        // set the new direction for the snake
        let predicted_direction = self.predict_direction(target_food_position);
        self.body.change_dir(predicted_direction);

        // update the snake's state
        self.crawl(target_food_position);

        // respawn
        if should_respawn && self.body.dead {
            self.respawn();
        }
    }

    fn respawn(&mut self) {
        self.body = SnakeBody::new();
        self.food = SnakeFood::new();
        self.score = 0.;
        self.ticks_until_starvation = MAX_TICKS_WITH_NO_FOOD;
    }
}

#[derive(Clone)]
struct SnakeFood {
    positions: HashSet<Position>,
}

impl SnakeFood {
    fn new() -> Self {
        let mut food = SnakeFood {
            positions: HashSet::new(),
        };

        food.refill();
        food
    }

    /// Spawns food in random unoccupied (by food) locations on the board until we hit the food limit
    fn refill(&mut self) {
        while self.positions.len() < FOOD_COUNT {
            let new_pos = loop {
                // TODO: optimize...
                // the range for row/col is adjusted here to avoid food spawning directly on the
                // board edges, which are deadly to snakes
                let random_row = rand::gen_range(2, MAX_ROWS - 3);
                let random_col = rand::gen_range(2, MAX_COLUMNS - 3);
                let spawn_pos = Position::new(random_row, random_col);
                if !self.positions.contains(&spawn_pos) {
                    break spawn_pos;
                }
            };

            self.positions.insert(new_pos);
        }
    }

    /// Display each piece of food on the board
    fn draw(&self, chunk_width: f32, chunk_height: f32) {
        for pos in self.positions.iter() {
            let x = (pos.col as f32) * chunk_width;
            let y = (pos.row as f32) * chunk_height;
            draw_rectangle(x, y, chunk_width, chunk_height, ORANGE);
        }
    }
}

#[derive(Clone)]
struct SnakeBody {
    chunks: VecDeque<Position>,
    chunk_set: HashSet<Position>,
    direction: Direction,
    dead: bool,
}

impl SnakeBody {
    fn new() -> Self {
        // start in the center of the board
        let start_row = MAX_ROWS / 2;
        let start_col = MAX_COLUMNS / 2;
        let chunk = Position::new(start_row, start_col);
        let chunks = vec![chunk];

        Self {
            chunks: VecDeque::from(chunks.clone()),
            chunk_set: HashSet::from_iter(chunks),
            direction: Up,
            dead: false,
        }
    }

    /// Attempt to point the snake in a new direction. Snake cannot turn back onto itself.
    fn change_dir(&mut self, new_direction: Direction) {
        let old_direction = self.direction;
        // forbid moving in the directly opposite direction
        if (old_direction == Up && new_direction == Down)
            || (old_direction == Down && new_direction == Up)
            || (old_direction == Left && new_direction == Right)
            || (old_direction == Right && new_direction == Left)
        {
            return;
        }
        self.direction = new_direction;
    }

    fn get_head_position(&self) -> Position {
        self.chunks
            .back()
            .copied()
            .expect("snake should always have a head")
    }
}

/// Initializes N new snake games
fn init_snakes(count: usize) -> Vec<Snake> {
    let mut snakes = Vec::with_capacity(count);

    for _ in 0..count {
        snakes.push(Snake::new(None));
    }

    snakes
}

/// Runs the simulation for the provided snakes over the specified number of generations.
/// Returns a new vector of snakes, that is hopefully better at the task than the original group.
/// Also returns the best snake from the last generation
fn run_simulation(
    snakes: Vec<Snake>,
    max_generations: usize,
    curr_generation: &mut usize,
) -> (Vec<Snake>, Snake) {
    // grab just the NNs from the supplied snakes
    let mut snakes_in_training = snakes
        .into_iter()
        .map(|g| Snake::new(Some(g.network)))
        .collect::<Vec<_>>();

    // time how long training runs take to provide an estimate for completion
    let mut generation_start_ts = std::time::Instant::now();
    let mut generation_durations = VecDeque::new();
    let gen_durations_to_store = 10;

    let mut generation_counter = 0;

    loop {
        // progress all snakes in parallel
        snakes_in_training
            .par_iter_mut()
            .for_each(|g| g.run_game_logic(false));

        // trigger evolution once all the snakes are dead OR the simulation has been running for long enough
        if snakes_in_training.par_iter().all(|g| g.body.dead)
            || generation_start_ts.elapsed().as_secs_f32() > MAX_GENERATION_DURATION_SECS
        {
            // TODO: pick some poor performers too, to expand the gene pool
            // find the best performing snakes
            snakes_in_training.sort_by(|a, b| a.score.total_cmp(&b.score));
            let mut top_snakes = snakes_in_training
                .iter()
                .rev() // reverse here to get descending order
                .take(SAMPLE_SIZE)
                .collect::<Vec<_>>();

            // TODO: plot this instead, probably via a crate
            let generation_progress_summary = format!(
                "Generation {}. Avg score of the {} selected snakes of prev generation: {}",
                *curr_generation + generation_counter,
                SAMPLE_SIZE,
                top_snakes.iter().map(|s| s.score).sum::<f32>() / top_snakes.len() as f32
            );

            let mut new_snakes = Vec::with_capacity(SNAKE_COUNT);

            // preserve the NNs from the top snakes of the previous round
            for top_snake in top_snakes.iter() {
                new_snakes.push(Snake::new(Some(top_snake.network.clone())))
            }

            // preserve the neural network of the best snake of the previous generation, so we can
            // watch it play later
            let last_best_snake = Snake::new(Some(top_snakes[0].clone().network));

            // randomize the order of the top snakes and trigger reproduction
            top_snakes.shuffle();

            let snake_pairs = top_snakes.par_chunks(2).collect::<Vec<_>>();

            while new_snakes.len() < SNAKE_COUNT {
                for pair in snake_pairs.iter() {
                    if new_snakes.len() >= SNAKE_COUNT {
                        break;
                    }
                    if let [snake_a, snake_b] = pair {
                        let offspring_nn = snake_a.network.mate(&snake_b.network);
                        let offspring = Snake::new(Some(offspring_nn));
                        new_snakes.push(offspring);
                    } else {
                        // no-op for unpaired snakes for now to avoid over-representing the unlucky bachelors
                    }
                }
            }

            assert_eq!(new_snakes.len(), SNAKE_COUNT);

            // calculate remaining time for the training
            // TODO: clean this up
            let loop_duration_secs = generation_start_ts.elapsed().as_secs_f32();
            generation_start_ts = std::time::Instant::now();

            if generation_durations.len() > gen_durations_to_store {
                generation_durations.pop_front();
            }
            generation_durations.push_back(loop_duration_secs);

            let avg_gen_duration =
                generation_durations.iter().sum::<f32>() / generation_durations.len() as f32;
            let generations_remaining = max_generations - generation_counter;
            let remaining_time_secs = avg_gen_duration * generations_remaining as f32;

            let msg = format!(
                "{}. Training should complete in ~{:.1} seconds",
                generation_progress_summary, remaining_time_secs
            );

            println!("{}", msg);

            // use the newly created snakes in the next generation
            snakes_in_training = new_snakes;
            generation_counter += 1;

            // stop training once we reach our desired generation number
            if generation_counter == max_generations {
                *curr_generation += generation_counter;
                break (snakes_in_training, last_best_snake);
            }
        }
    }
}

#[macroquad::main("Snake")]
async fn main() {
    let chunk_width = screen_width() / MAX_COLUMNS as f32;
    let chunk_height = screen_height() / MAX_ROWS as f32;

    let mut games = init_snakes(SNAKE_COUNT);
    let mut generation = 0;
    let mut best_snake = games[0].clone();

    // main render loop
    loop {
        if is_key_pressed(Escape) || is_key_pressed(Q) {
            // quit
            break;
        }

        if is_key_pressed(V) {
            // trigger evolution. This takes a while.
            (games, best_snake) =
                run_simulation(games, GENERATIONS_PER_TRAINING_RUN, &mut generation);
        }

        // TODO: display snake score, food eaten, and length
        // TODO: render @60fps rather than unlocked
        best_snake.run_game_logic(true);

        best_snake.draw(chunk_width, chunk_height);

        // render fps and generation counters
        let fps_msg = format!("FPS: {}", get_fps());
        let gen_counter_msg = format!("Generation: {}", generation);
        draw_text_corner(&[fps_msg.as_str(), gen_counter_msg.as_str()]);

        next_frame().await
    }
}

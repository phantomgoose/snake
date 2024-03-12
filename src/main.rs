use std::collections::{HashSet, VecDeque};

use macroquad::prelude::*;
use macroquad::prelude::KeyCode::{Escape, Q, V};
use macroquad::rand::ChooseRandom;

use utils::{Direction, Position};
use utils::Direction::{Down, Left, Right, Up};

use crate::brain::NeuralNetwork;
use crate::utils::{DIRECTIONS, draw_text_center_default, draw_text_corner, find_closest_pos};

mod brain;
mod utils;

const MAX_ROWS: usize = 100;
const MAX_COLUMNS: usize = 100;

const FOOD_COUNT: usize = 1;
const SNAKE_COUNT: usize = 1000;

const SELECTION_RATE: f32 = 0.2;
const FOOD_REWARD: f32 = 10000.;
// snake dies of starvation if it doesn't get to food in this many ticks. Prevents snakes looping around permanently.
const MAX_TICKS_WITH_NO_FOOD: usize = 400;
// whether the snake will die if it collides with itself
const SELF_COLLISION_ENABLED: bool = false;
// reward the snake for getting close to food, even if it doesn't consume it
const REWARD_FOOD_PROXIMITY: bool = true;
// reduce the snake's final score if it dies prior to starving (eg by running into a wall or itself)
const PUNISH_FOR_CRASHING: bool = true;

#[derive(Clone)]
struct SnakeGame {
    snake: Snake,
    network: NeuralNetwork,
    ticks_until_starvation: usize,
    score: f32,
    food: Food,
}

impl SnakeGame {
    /// Creates a new snake using the supplied neural network. If not provided, spawns the snake with a randomly initialized NN.
    fn new(neural_network: Option<NeuralNetwork>) -> Self {
        Self {
            snake: Snake::new(),
            network: neural_network.unwrap_or_default(),
            score: 0.,
            ticks_until_starvation: MAX_TICKS_WITH_NO_FOOD,
            food: Food::new(),
        }
    }

    /// Predicts the snake's next direction given its knowledge about its environment (and a neural network).
    ///
    /// # Arguments
    ///
    /// * `food_pos`: location of the food to move towards
    ///
    /// returns: new Direction for the snake to move in
    fn predict_direction(&self, food_pos: Position) -> Direction {
        let head_pos = self.snake.get_head_position();

        // row and column diffs compared to food location, normalized for the board size
        let row_diff = (head_pos.row as f32 - food_pos.row as f32) / MAX_ROWS as f32;
        let col_diff = (head_pos.col as f32 - food_pos.col as f32) / MAX_COLUMNS as f32;

        // proximity to board borders (0..MAX_ROWS/COLUMNS), normalized for board size
        let row_bounds_proximity = head_pos.row as f32 / MAX_ROWS as f32;
        let col_bounds_proximity = head_pos.col as f32 / MAX_COLUMNS as f32;

        // TODO: add proximity to self (ray-based from head, left/right/forward relative to current dir

        let mut input_vec = [
            row_diff,
            col_diff,
            row_bounds_proximity,
            col_bounds_proximity,
            0.,
            0.,
            0.,
            0.,
        ];

        // encode snake's current direction
        input_vec[self.snake.direction as usize + 4] = 1.;

        // ensure we normalized all inputs
        assert!(
            input_vec.iter().all(|i| -1. <= *i && *i <= 1.),
            "Input vec had non-normalized values: {:?}",
            input_vec
        );

        self.network.classify(&input_vec, &DIRECTIONS)
    }

    /// Paints the snake to screen, using the provided grid cell/chunk sizes to calculate snake body proportions
    fn draw(&self, chunk_width: f32, chunk_height: f32) {
        let snake_color = if self.snake.dead { GRAY } else { GREEN };
        for chunk in self.snake.chunks.iter() {
            let x = (chunk.col as f32) * chunk_width;
            let y = (chunk.row as f32) * chunk_height;
            draw_rectangle(x, y, chunk_width, chunk_height, snake_color);
        }

        self.food.draw(chunk_width, chunk_height);
    }

    /// Moves the snake in the direction it's currently facing
    /// Checks for collision with closest known food on the board
    /// Rewards/punishes/kills the snake depending on its state
    fn crawl(&mut self, closest_food_pos: Position) {
        assert!(
            !self.snake.chunks.is_empty(),
            "Snake should always have at least one chunk"
        );

        if self.snake.dead {
            return;
        }

        // starvation
        self.ticks_until_starvation -= 1;
        if self.ticks_until_starvation == 0 {
            self.snake.dead = true;
            return;
        }

        let Position {
            col: previous_col,
            row: previous_row,
        } = self.snake.get_head_position();

        // compute next position based on snake's direction
        let (new_col, new_row) = match self.snake.direction {
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
            self.snake.dead = true;
            if PUNISH_FOR_CRASHING {
                // punishment for running into a wall
                self.score *= 0.2;
            }
            return;
        }

        let new_position = Position::new(new_row.unwrap(), new_col.unwrap());

        // snake is also dead if the new position is in the existing chunk set which means that the snake collided with itself
        if SELF_COLLISION_ENABLED && self.snake.chunk_set.contains(&new_position) {
            self.snake.dead = true;
            return;
        }

        // if we reached food, we don't lose our tail, causing the snake to grow!
        // we do eat the food, however
        if closest_food_pos == new_position {
            // reward for food
            self.score += FOOD_REWARD;
            // no longer starving
            self.ticks_until_starvation = MAX_TICKS_WITH_NO_FOOD;
            // food is eaten and not available to other snakes
            self.food.positions.remove(&closest_food_pos);
            // spawn new food
            self.food.refill();
        } else {
            // reward for being alive, proportional to how close to the food we got
            if REWARD_FOOD_PROXIMITY {
                let distance = Vec2::from(new_position).distance_squared(closest_food_pos.into());
                self.score += 1. / distance;
            }

            // shrink the snake, removing its tail
            let chunk = self.snake.chunks.pop_front().unwrap();
            self.snake.chunk_set.remove(&chunk);
        }

        // grow the snake, appending its head
        self.snake.chunk_set.insert(new_position);
        self.snake.chunks.push_back(new_position);
    }

    /// Progresses the snake game by one frame/tick
    fn run_game_logic(&mut self, should_respawn: bool) {
        // locate the closest food item to attempt to move towards
        let target_food_position =
            find_closest_pos(self.snake.get_head_position(), &self.food.positions);

        // set new direction for the snake
        let predicted_direction = self.predict_direction(target_food_position);
        self.snake.change_dir(predicted_direction);

        // update snake's state
        self.crawl(target_food_position);

        // respawn?
        if should_respawn && self.snake.dead {
            self.respawn();
        }
    }

    fn respawn(&mut self) {
        self.snake = Snake::new();
        self.food = Food::new();
        self.score = 0.;
        self.ticks_until_starvation = MAX_TICKS_WITH_NO_FOOD;
    }
}

#[derive(Clone)]
struct Food {
    positions: HashSet<Position>,
}

impl Food {
    fn new() -> Self {
        let mut food = Food {
            positions: HashSet::new(),
        };

        food.refill();
        food
    }

    /// Spawns food in random unoccupied (by food) locations on the board until hitting the food limit
    fn refill(&mut self) {
        while self.positions.len() < FOOD_COUNT {
            let new_pos = loop {
                // TODO: optimize...
                let random_row = rand::gen_range(3, MAX_ROWS - 3);
                let random_col = rand::gen_range(3, MAX_COLUMNS - 3);
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
struct Snake {
    chunks: VecDeque<Position>,
    chunk_set: HashSet<Position>,
    direction: Direction,
    dead: bool,
}

impl Snake {
    fn new() -> Self {
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
        // forbid moving in the opposite direction
        if (old_direction == Up && new_direction == Down)
            || (old_direction == Down && new_direction == Up)
            || (old_direction == Left && new_direction == Right)
            || (old_direction == Right && new_direction == Left)
        {
            return;
        }
        self.direction = new_direction;
    }

    /// Gets the snake's head position
    fn get_head_position(&self) -> Position {
        self.chunks
            .back()
            .copied()
            .expect("snake should always have a head")
    }
}

/// Initializes N new snake games
fn init_games(count: usize) -> Vec<SnakeGame> {
    let mut games = Vec::with_capacity(count);

    for _ in 0..count {
        games.push(SnakeGame::new(None));
    }

    games
}

/// Runs the simulation for the provided games over the specified number of generations.
/// Returns a new vector of games, that is hopefully better at the task than the original group.
/// Also returns the best snake from the last generation
fn run_simulation(
    games: Vec<SnakeGame>,
    max_generations: usize,
    curr_generation: &mut usize,
) -> (Vec<SnakeGame>, SnakeGame) {
    // grab just the NNs from the supplied games
    let mut cloned_games = games
        .into_iter()
        .map(|g| SnakeGame::new(Some(g.network)))
        .collect::<Vec<_>>();

    let mut generation_counter = 0;

    loop {
        cloned_games
            .iter_mut()
            .for_each(|g| g.run_game_logic(false));

        // trigger evolution once all the snakes are dead
        if cloned_games.iter().all(|g| g.snake.dead) {
            // TODO: pick some poor performers too, to expand the gene pool
            // find the best performing snakes
            cloned_games.sort_by(|a, b| a.score.total_cmp(&b.score));
            let mut top_snakes = cloned_games
                .iter()
                .rev() // gotta remember to reverse here to get descending order
                .take((SNAKE_COUNT as f32 * SELECTION_RATE) as usize)
                .collect::<Vec<_>>();

            // TODO: plot this instead, prolly via a crate
            println!(
                "Generation {}. Avg score of the top 10 snakes of prev generation {}",
                *curr_generation + generation_counter,
                top_snakes.iter().take(10).map(|s| s.score).sum::<f32>() / 10.
            );

            let mut new_snakes = Vec::with_capacity(SNAKE_COUNT);

            // preserve the NNs from the top snakes of the previous round
            for top_snake in top_snakes.iter() {
                new_snakes.push(SnakeGame::new(Some(top_snake.network.clone())))
            }

            let last_best_snake = SnakeGame::new(Some(top_snakes[0].clone().network));

            // randomize top snakes and trigger reproduction
            top_snakes.shuffle();

            let snake_pairs = top_snakes.chunks(2).collect::<Vec<_>>();

            while new_snakes.len() < SNAKE_COUNT {
                for pair in snake_pairs.iter() {
                    if new_snakes.len() >= SNAKE_COUNT {
                        break;
                    }
                    if let [snake_a, snake_b] = pair {
                        let offspring_nn = snake_a.network.mate(&snake_b.network);
                        let offspring = SnakeGame::new(Some(offspring_nn));
                        new_snakes.push(offspring);
                    } else {
                        // no-op for now to avoid over-representing the lucky bachelors
                    }
                }
            }

            assert_eq!(new_snakes.len(), SNAKE_COUNT);

            cloned_games = new_snakes;
            generation_counter += 1;

            if generation_counter == max_generations {
                *curr_generation += generation_counter;
                break (cloned_games, last_best_snake);
            }
        }
    }
}

#[macroquad::main("Snake")]
async fn main() {
    let chunk_width = screen_width() / MAX_COLUMNS as f32;
    let chunk_height = screen_height() / MAX_ROWS as f32;

    let mut games = init_games(SNAKE_COUNT);
    let mut generation = 0;
    let mut best_snake = games[0].clone();

    // main render loop
    loop {
        if is_key_pressed(Escape) || is_key_pressed(Q) {
            break;
        }

        if is_key_pressed(V) {
            // trigger evolution
            (games, best_snake) = run_simulation(games, 50, &mut generation);
        }

        best_snake.run_game_logic(true);

        best_snake.draw(chunk_width, chunk_height);

        // render fps counter
        draw_text_corner(&[get_fps().to_string().as_str()]);

        // render the generation counter
        let msg = format!("Generation {}", generation);
        draw_text_center_default(msg.as_str());

        next_frame().await
    }
}

use std::collections::{HashSet, VecDeque};

use macroquad::input::KeyCode::Escape;
use macroquad::prelude::*;
use macroquad::prelude::KeyCode::{A, D, S, W};
use macroquad::rand::ChooseRandom;

use crate::brain::NeuralNetwork;
use crate::Direction::{Down, Left, Right, Up};
use crate::utils::draw_text_center_default;

mod brain;
mod utils;

const MAX_ROWS: usize = 100;
const MAX_COLUMNS: usize = 100;

const GAME_STATE_UPDATE_RATE_SECS: f64 = 1. / 600.;
const SNAKE_COUNT: usize = 1000;
const SELECTION_DIVISOR: usize = 3;
const FOOD_REWARD: f32 = 10000.;
const MAX_TICKS_WITH_NO_FOOD: usize = 400;
const SELF_COLLISION_ENABLED: bool = false;

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
enum Direction {
    Left,
    Right,
    Up,
    Down,
}

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
struct Position {
    row: usize,
    col: usize,
}

impl Position {
    fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
}

#[derive(Clone)]
struct SnakeGame {
    snake: Snake,
    network: NeuralNetwork,
    prev_tick_time: f64,
    ticks_with_no_food: usize,
    // if this hits zero, snake dies
    score: f32,
}

impl SnakeGame {
    fn new(snake_id: usize, neural_network: Option<NeuralNetwork>) -> Self {
        Self {
            snake: Snake::new(MAX_ROWS / 2, MAX_COLUMNS / 2, snake_id),
            network: neural_network.unwrap_or_default(),
            prev_tick_time: 0.,
            score: 0.,
            ticks_with_no_food: 0,
        }
    }

    fn predict_direction(&self, food: &Food) -> Direction {
        let food_pos = food.pos;
        let head_pos = self.snake.get_head_position();

        // determine direction that will move us towards food
        // row diff normalized for the row count
        let row_diff = (head_pos.row as f32 - food_pos.row as f32) / MAX_ROWS as f32;
        // col diff normalized for the col count
        let col_diff = (head_pos.col as f32 - food_pos.col as f32) / MAX_COLUMNS as f32;

        let row_bounds_closeness = head_pos.row as f32 / MAX_ROWS as f32;
        let col_bounds_closeness = head_pos.col as f32 / MAX_COLUMNS as f32;

        let dirs = vec![Right, Left, Up, Down];

        let snake_dir_encoding = dirs
            .iter()
            .map(|dir| {
                if *dir == self.snake.direction {
                    1.0
                } else {
                    0.
                }
            })
            .collect::<Vec<_>>();

        let mut input_vec: Vec<f32> = vec![
            row_diff,
            col_diff,
            row_bounds_closeness,
            col_bounds_closeness,
        ];

        input_vec.extend_from_slice(snake_dir_encoding.as_slice());

        assert!(input_vec.iter().all(|i| -1. <= *i && *i <= 1.));

        self.network.classify(input_vec, dirs)
    }

    fn draw(&self, chunk_width: f32, chunk_height: f32) {
        if self.snake.dead {
            return;
        }
        let snake_color = if self.snake.dead { GRAY } else { GREEN };
        for chunk in self.snake.chunks.iter() {
            let x = (chunk.col as f32) * chunk_width;
            let y = (chunk.row as f32) * chunk_height;
            draw_rectangle(x, y, chunk_width, chunk_height, snake_color);
        }

        // if self.snake.dead {
        //     let msg = format!("You're dead! Score: {}", self.get_score());
        //     draw_text_center_default(msg.as_str());
        // }
    }

    fn crawl(&mut self, food: &mut Food) {
        assert!(
            !self.snake.chunks.is_empty(),
            "Snake should always have at least one chunk"
        );

        if self.snake.dead {
            return;
        }

        // TODO: train all the agents in parallel and let them live longer
        // starvation
        self.ticks_with_no_food += 1;
        if self.ticks_with_no_food > MAX_TICKS_WITH_NO_FOOD {
            self.snake.dead = true;
            return;
        }

        let Position {
            row: last_row,
            col: last_col,
        } = self.snake.chunks.iter().last().unwrap();

        let (new_chunk_col, new_chunk_row) = match self.snake.direction {
            Left => (last_col.checked_sub(1), Some(*last_row)),
            Right => (last_col.checked_add(1), Some(*last_row)),
            Up => (Some(*last_col), last_row.checked_sub(1)),
            Down => (Some(*last_col), last_row.checked_add(1)),
        };

        // mark snake as dead if col or row is none or above max which means snake hit a wall
        if new_chunk_row.is_none()
            || new_chunk_col.is_none()
            || new_chunk_row.unwrap() > MAX_ROWS - 1
            || new_chunk_col.unwrap() > MAX_COLUMNS - 1
        {
            self.snake.dead = true;
            // punishment for running into a wall
            self.score *= 0.2;
            return;
        }

        let new_position = Position::new(new_chunk_row.unwrap(), new_chunk_col.unwrap());

        // snake is also dead if new position is in existing chunk set which means snake hit itself
        if SELF_COLLISION_ENABLED && self.snake.chunk_set.contains(&new_position) {
            self.snake.dead = true;
            return;
        }

        // if we reached food, we don't lose our tail, causing the snake to grow! We do eat the food, however
        if food.pos == new_position {
            // reward for food
            self.score += FOOD_REWARD;
            // no longer starving
            self.ticks_with_no_food = 0;
            // spawn new food
            food.respawn();
        } else {
            // reward for being alive, proportional to how close to the food it is
            // find
            let head_pos = Vec2::new(new_position.col as f32, new_position.row as f32);
            let food_pos = Vec2::new(food.pos.col as f32, food.pos.row as f32);
            let distance = head_pos.distance_squared(food_pos);
            self.score += 1. / distance;

            let chunk = self.snake.chunks.pop_front().unwrap();
            self.snake.chunk_set.remove(&chunk);
        }

        self.snake.chunk_set.insert(new_position);
        self.snake.chunks.push_back(new_position);
    }
}

struct Food {
    pos: Position,
}

impl Food {
    fn new() -> Self {
        let mut food = Food {
            pos: Position::new(0, 0),
        };
        food.respawn();
        food
    }
    fn respawn(&mut self) {
        let random_row = rand::gen_range(3, MAX_ROWS - 3);
        let random_col = rand::gen_range(3, MAX_COLUMNS - 3);
        self.pos = Position::new(random_row, random_col);
    }

    fn draw(&self, chunk_width: f32, chunk_height: f32) {
        let x = (self.pos.col as f32) * chunk_width;
        let y = (self.pos.row as f32) * chunk_height;
        draw_rectangle(x, y, chunk_width, chunk_height, ORANGE);
    }
}

#[derive(Clone)]
struct Snake {
    id: usize,
    chunks: VecDeque<Position>,
    chunk_set: HashSet<Position>,
    direction: Direction,
    dead: bool,
}

impl Snake {
    fn new(start_row: usize, start_col: usize, id: usize) -> Self {
        let chunk = Position::new(start_row, start_col);
        let mut chunk_set = HashSet::new();
        chunk_set.insert(chunk);
        Self {
            id,
            chunks: VecDeque::from(vec![chunk]),
            chunk_set,
            direction: Up,
            dead: false,
        }
    }

    fn change_dir(&mut self, old_direction: Direction, new_direction: Direction) {
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

    fn get_head_position(&self) -> &Position {
        self.chunks
            .front()
            .expect("snake should always have at least one chunk")
    }
}

fn init_games(count: usize) -> Vec<SnakeGame> {
    let mut games = Vec::new();

    for i in 0..count {
        games.push(SnakeGame::new(i, None));
    }

    games
}

#[macroquad::main("Snake")]
async fn main() {
    let manual_control = false;
    let mut games = init_games(SNAKE_COUNT);

    // direction that the snake was heading in (which is known to not have caused it to die)

    let chunk_width = screen_width() / MAX_COLUMNS as f32;
    let chunk_height = screen_height() / MAX_ROWS as f32;

    let mut food = Food::new();

    let mut generation = 0;

    let mut do_draw = true;

    loop {
        // TODO: turn the following into a macro
        if is_key_pressed(Escape) {
            break;
        }

        if is_key_pressed(KeyCode::V) {
            do_draw = !do_draw;
        }

        // check if it's time to tick the game logic
        // TODO: this should be based on frame time
        let time = get_time();

        for (i, game) in &mut games.iter_mut().enumerate() {
            let last_good_direction = game.snake.direction;

            // update direction
            if manual_control {
                match get_last_key_pressed() {
                    Some(W) => game.snake.change_dir(last_good_direction, Up),
                    Some(A) => game.snake.change_dir(last_good_direction, Left),
                    Some(S) => game.snake.change_dir(last_good_direction, Down),
                    Some(D) => game.snake.change_dir(last_good_direction, Right),
                    _ => {}
                }
            } else {
                game.snake
                    .change_dir(last_good_direction, game.predict_direction(&food));
            }

            // update state
            let time_since_last_tick = time - game.prev_tick_time;
            if time_since_last_tick > GAME_STATE_UPDATE_RATE_SECS {
                game.crawl(&mut food);

                game.prev_tick_time = time;
            }

            if do_draw {
                game.draw(chunk_width, chunk_height);
            }
        }

        food.draw(chunk_width, chunk_height);
        let msg = format!("Generation {}", generation);
        draw_text_center_default(msg.as_str());

        // see if round is over, trigger evolution if so
        if games.iter().all(|g| g.snake.dead) {
            // TODO: pick some poor performers too
            games.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
            let mut top_snakes = games
                .iter()
                .rev()
                .take(SNAKE_COUNT / SELECTION_DIVISOR)
                .collect::<Vec<_>>();

            let mut new_snakes = Vec::with_capacity(SNAKE_COUNT);

            for top_snake in top_snakes.iter() {
                new_snakes.push(SnakeGame::new(0, Some(top_snake.network.clone())))
            }

            top_snakes.shuffle();

            let snake_pairs = top_snakes.chunks(2).collect::<Vec<_>>();

            while new_snakes.len() < SNAKE_COUNT {
                for pair in snake_pairs.iter() {
                    if new_snakes.len() >= SNAKE_COUNT {
                        break;
                    }
                    if let [snake_a, snake_b] = pair {
                        let [offspring_nn_a, offspring_nn_b] =
                            snake_a.network.mate(&snake_b.network);
                        let offspring_a = SnakeGame::new(0, Some(offspring_nn_a));
                        let offspring_b = SnakeGame::new(0, Some(offspring_nn_b));
                        new_snakes.push(offspring_a);
                        new_snakes.push(offspring_b);
                    } else {
                        // self-replicate
                        let child_nns = pair[0].network.mate(&pair[0].network);
                        for child_nn in child_nns {
                            new_snakes.push(SnakeGame::new(0, Some(child_nn)));
                        }
                    }
                }
            }

            while new_snakes.len() > SNAKE_COUNT {
                new_snakes.pop();
            }

            assert_eq!(new_snakes.len(), SNAKE_COUNT);

            games = new_snakes;
            generation += 1;
        }

        next_frame().await
    }
}

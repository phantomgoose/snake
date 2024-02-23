use std::collections::{HashSet, VecDeque};

use macroquad::prelude::*;
use macroquad::prelude::KeyCode::{A, D, Escape, S, W};
use macroquad::rand::ChooseRandom;

use crate::brain::NeuralNetwork;
use crate::Direction::{Down, Left, Right, Up};
use crate::utils::draw_text_center_default;

mod brain;
mod utils;

const MAX_ROWS: usize = 20;
const MAX_COLUMNS: usize = 25;

const GAME_STATE_UPDATE_RATE_SECS: f64 = 0.1;

#[derive(Eq, PartialEq, Copy, Clone)]
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

struct SnakeGame {
    snake: Snake,
    food: Option<Position>,
    network: NeuralNetwork,
}

impl SnakeGame {
    fn new() -> Self {
        let mut game = Self {
            snake: Snake::new(MAX_ROWS / 2, MAX_COLUMNS / 2),
            food: None,
            network: NeuralNetwork::new(),
        };

        game.spawn_food();

        game
    }

    fn predict_direction(&self) -> Direction {
        let dirs = [Right, Left, Up, Down];

        let food_pos = self.food.unwrap();
        let head_pos = self.snake.get_head_position();

        // determine direction that will move us towards food
        // row diff normalized for the row count
        let row_diff = (head_pos.row as f32 - food_pos.row as f32) / MAX_ROWS as f32;
        // col diff normalized for the col count
        let col_diff = (head_pos.col as f32 - food_pos.col as f32) / MAX_COLUMNS as f32;

        let input_vec: Vec<f32> = vec![row_diff, col_diff];

        let prediction = self.network.probabilities(input_vec);
        // find index of the max value in prediction vector
        let max_probability = prediction.iter().copied().reduce(f32::max).unwrap();
        let max_index = prediction
            .iter()
            .position(|&x| x == max_probability)
            .unwrap();
        dirs[max_index]
    }

    fn draw(&self, chunk_width: f32, chunk_height: f32) {
        for chunk in self.snake.chunks.iter() {
            let x = (chunk.col as f32) * chunk_width;
            let y = (chunk.row as f32) * chunk_height;
            draw_rectangle(x, y, chunk_width, chunk_height, GREEN);
        }

        if let Some(food) = &self.food {
            let x = (food.col as f32) * chunk_width;
            let y = (food.row as f32) * chunk_height;
            draw_rectangle(x, y, chunk_width, chunk_height, ORANGE);
        }

        if self.snake.dead {
            let msg = format!("You're dead! Score: {}", self.get_score());
            draw_text_center_default(msg.as_str());
        }
    }

    fn get_score(&self) -> usize {
        self.snake.chunks.len()
    }

    fn crawl(&mut self) {
        assert!(
            !self.snake.chunks.is_empty(),
            "Snake should always have at least one chunk"
        );

        if self.snake.dead {
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
            return;
        }

        let new_position = Position::new(new_chunk_row.unwrap(), new_chunk_col.unwrap());

        // snake is also dead if new position is in existing chunk set which means snake hit itself
        if self.snake.chunk_set.contains(&new_position) {
            self.snake.dead = true;
            return;
        }

        // if we reached food, we don't lose our tail, causing the snake to grow! We do eat the food, however
        if self.food == Some(new_position) {
            // spawn new food
            self.spawn_food();
        } else {
            let chunk = self.snake.chunks.pop_front().unwrap();
            self.snake.chunk_set.remove(&chunk);
        }
        self.snake.chunk_set.insert(new_position);
        self.snake.chunks.push_back(new_position);
    }

    fn spawn_food(&mut self) {
        let random_row = rand::gen_range(2, MAX_ROWS);
        let random_col = rand::gen_range(2, MAX_COLUMNS);
        self.food = Some(Position::new(random_row, random_col));
    }
}

struct Snake {
    chunks: VecDeque<Position>,
    chunk_set: HashSet<Position>,
    direction: Direction,
    dead: bool,
}

impl Snake {
    fn new(start_row: usize, start_col: usize) -> Self {
        let chunk = Position::new(start_row, start_col);
        let mut chunk_set = HashSet::new();
        chunk_set.insert(chunk);
        Self {
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

#[macroquad::main("Snake")]
async fn main() {
    let mut game = SnakeGame::new();

    let mut prev_tick_time = 0.;
    // direction that the snake was heading in (which is known to not have caused it to die)
    let mut last_good_direction = game.snake.direction;

    let chunk_width = screen_width() / MAX_COLUMNS as f32;
    let chunk_height = screen_height() / MAX_ROWS as f32;

    loop {
        if is_key_pressed(Escape) {
            break;
        }

        match get_last_key_pressed() {
            Some(W) => game.snake.change_dir(last_good_direction, Up),
            Some(A) => game.snake.change_dir(last_good_direction, Left),
            Some(S) => game.snake.change_dir(last_good_direction, Down),
            Some(D) => game.snake.change_dir(last_good_direction, Right),
            _ => game
                .snake
                .change_dir(last_good_direction, game.predict_direction()),
        }

        // check if it's time to tick the game logic
        // TODO: this should be based on frame time
        let time = get_time();
        if time - prev_tick_time > GAME_STATE_UPDATE_RATE_SECS {
            game.crawl();

            last_good_direction = game.snake.direction;
            prev_tick_time = time;
        }

        clear_background(BLACK);
        game.draw(chunk_width, chunk_height);

        next_frame().await
    }
}

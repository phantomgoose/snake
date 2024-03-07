use std::collections::{HashSet, VecDeque};

use macroquad::input::KeyCode::Escape;
use macroquad::prelude::*;
use macroquad::prelude::KeyCode::{A, D, R, S, W};
use macroquad::rand::ChooseRandom;

use crate::brain::NeuralNetwork;
use crate::Direction::{Down, Left, Right, Up};
use crate::utils::draw_text_center_default;

mod brain;
mod utils;

const MAX_ROWS: usize = 50;
const MAX_COLUMNS: usize = 50;

const GAME_STATE_UPDATE_RATE_SECS: f64 = 1. / 60.;
const SNAKE_COUNT: usize = 100;
const SELECTION_DIVISOR: usize = 2;
const MAX_TICKS_WITH_NO_FOOD: usize = 1000;
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
    food: Option<Position>,
    network: NeuralNetwork,
    prev_tick_time: f64,
    ticks_with_no_food: usize,
    // if this hits zero, snake dies
    score: f32,
}

impl SnakeGame {
    fn new(snake_id: usize, neural_network: Option<NeuralNetwork>) -> Self {
        let mut game = Self {
            snake: Snake::new(MAX_ROWS / 2, MAX_COLUMNS / 2, snake_id),
            food: None,
            network: neural_network.unwrap_or_default(),
            prev_tick_time: 0.,
            score: 0.,
            ticks_with_no_food: 0,
        };

        game.spawn_food();

        game
    }

    fn predict_direction(&self) -> Direction {
        let food_pos = self.food.unwrap();
        let head_pos = self.snake.get_head_position();

        // determine direction that will move us towards food
        // row diff normalized for the row count
        let row_diff = (head_pos.row as f32 - food_pos.row as f32) / MAX_ROWS as f32;
        // col diff normalized for the col count
        let col_diff = (head_pos.col as f32 - food_pos.col as f32) / MAX_COLUMNS as f32;

        let input_vec: Vec<f32> = vec![row_diff, col_diff];

        let dirs = vec![Right, Left, Up, Down];
        self.network.classify(input_vec, dirs)
    }

    fn draw(&self, chunk_width: f32, chunk_height: f32) {
        let snake_color = if self.snake.dead { GRAY } else { GREEN };
        for chunk in self.snake.chunks.iter() {
            let x = (chunk.col as f32) * chunk_width;
            let y = (chunk.row as f32) * chunk_height;
            draw_rectangle(x, y, chunk_width, chunk_height, snake_color);
        }

        if let Some(food) = &self.food {
            let x = (food.col as f32) * chunk_width;
            let y = (food.row as f32) * chunk_height;
            draw_rectangle(x, y, chunk_width, chunk_height, ORANGE);
        }

        // if self.snake.dead {
        //     let msg = format!("You're dead! Score: {}", self.get_score());
        //     draw_text_center_default(msg.as_str());
        // }
    }

    fn crawl(&mut self) {
        assert!(
            !self.snake.chunks.is_empty(),
            "Snake should always have at least one chunk"
        );

        if self.snake.dead {
            return;
        }

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
            return;
        }

        let new_position = Position::new(new_chunk_row.unwrap(), new_chunk_col.unwrap());

        // snake is also dead if new position is in existing chunk set which means snake hit itself
        if SELF_COLLISION_ENABLED && self.snake.chunk_set.contains(&new_position) {
            self.snake.dead = true;
            return;
        }

        // if we reached food, we don't lose our tail, causing the snake to grow! We do eat the food, however
        if self.food == Some(new_position) {
            // reward for food
            self.score += 10.;
            // no longer starving
            self.ticks_with_no_food = 0;
            // spawn new food
            self.spawn_food();
        } else {
            let chunk = self.snake.chunks.pop_front().unwrap();
            self.snake.chunk_set.remove(&chunk);
        }

        // reward for being alive
        self.score += 0.01;
        self.snake.chunk_set.insert(new_position);
        self.snake.chunks.push_back(new_position);
    }

    fn spawn_food(&mut self) {
        let random_row = rand::gen_range(2, MAX_ROWS);
        let random_col = rand::gen_range(2, MAX_COLUMNS);
        self.food = Some(Position::new(random_row, random_col));
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

    loop {
        // TODO: turn the following into a macro
        if is_key_pressed(Escape) {
            break;
        }

        // check if it's time to tick the game logic
        // TODO: this should be based on frame time
        let time = get_time();

        for game in &mut games {
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
                    .change_dir(last_good_direction, game.predict_direction());
            }

            // update state
            let time_since_last_tick = time - game.prev_tick_time;
            if time_since_last_tick > GAME_STATE_UPDATE_RATE_SECS {
                game.crawl();

                game.prev_tick_time = time;
            }

            // draw
            game.draw(chunk_width, chunk_height);
        }

        if games.iter().all(|g| {
            g.snake.dead
                // TODO: figure out a better way to kill snakes that live too long without eating
                || g.score > 5.
        }) {
            // TODO: pick some poor performers too
            games.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
            let mut top_snakes = games
                .iter()
                .take(SNAKE_COUNT / SELECTION_DIVISOR)
                .collect::<Vec<_>>();

            top_snakes.shuffle();

            let mut new_snakes = Vec::new();

            // TODO: clear existing snakes' state, only keep NNs

            for pair in top_snakes.chunks(2) {
                if let [snake_a, snake_b] = pair {
                    let [offspring_nn_a, offspring_nn_b] = snake_a.network.mate(&snake_b.network);
                    let offspring_a = SnakeGame::new(0, Some(offspring_nn_a));
                    let offspring_b = SnakeGame::new(0, Some(offspring_nn_b));
                    new_snakes.push(offspring_a);
                    new_snakes.push(offspring_b);
                } else {
                    panic!("Found a pair of snakes with len != 2");
                }
            }

            // push existing snakes to new snakes, but clear their state
            for old_snake in top_snakes {
                let clear_snake = SnakeGame::new(0, Some(old_snake.network.clone()));
                new_snakes.push(clear_snake);
            }

            assert_eq!(new_snakes.len(), SNAKE_COUNT);

            loop {
                draw_text_center_default("All sneks dead");
                match get_last_key_pressed() {
                    Some(R) => {
                        games = new_snakes;
                        break;
                    }
                    _ => next_frame().await,
                }
            }
        }

        next_frame().await
    }
}

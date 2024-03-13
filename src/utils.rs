use std::collections::HashSet;

use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use macroquad::color::WHITE;
use macroquad::math::Vec2;
use macroquad::prelude::{Color, draw_text, get_text_center, screen_height, screen_width};

use crate::Snake;
use crate::utils::Direction::{Down, Left, Right, Up};

const DEFAULT_TEXT_COLOR: Color = WHITE;
const CENTER_TEXT_SIZE: u16 = 32;
const CORNER_TEXT_SIZE: u16 = 16;
const TEXT_PADDING: f32 = 5.;

pub(crate) const DIRECTIONS: [Direction; 4] = [Left, Right, Up, Down];

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub enum Direction {
    Left = 0,
    Right = 1,
    Up = 2,
    Down = 3,
}

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
pub(crate) struct Position {
    pub(crate) row: usize,
    pub(crate) col: usize,
}

impl Position {
    pub(crate) fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
}

impl From<Position> for Vec2 {
    fn from(pos: Position) -> Self {
        Self::new(pos.col as f32, pos.row as f32)
    }
}

pub(crate) fn draw_text_center(msg_str: &str, font_size: u16, text_color: Color) {
    let text_center = get_text_center(msg_str, None, font_size, 1., 0.0);
    draw_text(
        msg_str,
        screen_width() / 2. - text_center.x,
        screen_height() / 2. + text_center.y,
        font_size as f32,
        text_color,
    );
}

pub(crate) fn draw_text_center_default(msg_str: &str) {
    draw_text_center(msg_str, CENTER_TEXT_SIZE, DEFAULT_TEXT_COLOR)
}

pub(crate) fn draw_text_corner(messages: &[&str]) {
    let x = TEXT_PADDING;
    let mut y = TEXT_PADDING + CORNER_TEXT_SIZE as f32;
    for msg in messages {
        draw_text(msg, x, y, CORNER_TEXT_SIZE as f32, DEFAULT_TEXT_COLOR);
        y += CORNER_TEXT_SIZE as f32;
    }
}

pub(crate) fn predict_direction_naive(snake: &Snake, food: &Option<Position>) -> Direction {
    let current_dir = snake.direction;
    if food.is_none() {
        // no food -> no change in direction
        return current_dir;
    }

    let food_pos = food.unwrap();
    let head_pos = snake.get_head_position();

    // determine direction that will move us towards food
    let row_diff = head_pos.row as i32 - food_pos.row as i32;
    let col_diff = head_pos.col as i32 - food_pos.col as i32;

    if row_diff.abs() > col_diff.abs()
        && current_dir != Direction::Up
        && current_dir != Direction::Down
    {
        // vertical adjustment is best
        if row_diff < 0 {
            Direction::Down
        } else {
            Direction::Up
        }
    } else if current_dir != Direction::Left && current_dir != Direction::Right {
        // horizontal adjustment is best
        if col_diff < 0 {
            Direction::Right
        } else {
            Direction::Left
        }
    } else {
        current_dir
    }
}

pub(crate) fn find_closest_pos(curr_pos: Position, targets: &HashSet<Position>) -> Position {
    if targets.len() < 2 {
        // no need for any fancy optimization here. Useful for the default scenario of only one food being spawned at a time
        return *targets
            .iter()
            .next()
            .expect("Expected to find at least one target, but the set was empty");
    }

    let mut tree = KdTree::new(2);

    let targets_slice = targets.iter().copied().collect::<Vec<_>>();

    for (i, pos) in targets_slice.iter().enumerate() {
        tree.add([pos.col as f32, pos.row as f32], i).unwrap();
    }

    let closest_positions = tree
        .nearest(
            &[curr_pos.col as f32, curr_pos.row as f32],
            1,
            &squared_euclidean,
        )
        .unwrap();

    let closest_idx = *closest_positions[0].1;
    targets_slice[closest_idx]
}

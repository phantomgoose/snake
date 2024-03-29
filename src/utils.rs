use std::collections::HashSet;

use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use macroquad::color::WHITE;
use macroquad::math::Vec2;
use macroquad::prelude::{Color, draw_text, get_text_center, screen_height, screen_width};

use crate::SnakeBody;
use crate::utils::Direction::{Down, Left, Right, Up};

const DEFAULT_TEXT_COLOR: Color = WHITE;
const CENTER_TEXT_SIZE: u16 = 32;
const CORNER_TEXT_SIZE: u16 = 16;
const TEXT_PADDING: f32 = 5.;

pub(crate) const DIRECTIONS: [Direction; 4] = [Left, Right, Up, Down];

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub(crate) enum Direction {
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

/// Draws the msg to center of the screen
pub(crate) fn draw_text_center(msg_str: &str) {
    let text_center = get_text_center(msg_str, None, CENTER_TEXT_SIZE, 1., 0.0);
    draw_text(
        msg_str,
        screen_width() / 2. - text_center.x,
        screen_height() / 2. + text_center.y,
        CENTER_TEXT_SIZE as f32,
        DEFAULT_TEXT_COLOR,
    );
}

/// Draws the msg to top left corner of the screen
pub(crate) fn draw_text_corner(messages: &[&str]) {
    let x = TEXT_PADDING;
    let mut y = TEXT_PADDING + CORNER_TEXT_SIZE as f32;
    for msg in messages {
        draw_text(msg, x, y, CORNER_TEXT_SIZE as f32, DEFAULT_TEXT_COLOR);
        y += CORNER_TEXT_SIZE as f32;
    }
}

/// Naive, hardcoded AI for the snake for comparison
pub(crate) fn predict_direction_naive(snake: &SnakeBody, food: &Option<Position>) -> Direction {
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

    if row_diff.abs() > col_diff.abs() && current_dir != Up && current_dir != Down {
        // vertical adjustment is best
        if row_diff < 0 {
            Down
        } else {
            Up
        }
    } else if current_dir != Left && current_dir != Right {
        // horizontal adjustment is best
        if col_diff < 0 {
            Right
        } else {
            Left
        }
    } else {
        current_dir
    }
}

/// Helper function for finding the closest position to the current one from a set of targets
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

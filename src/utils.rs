use macroquad::color::WHITE;
use macroquad::prelude::{Color, draw_text, get_text_center, screen_height, screen_width};

use crate::{Direction, Position, Snake};

const CENTER_TEXT_COLOR: Color = WHITE;
const CENTER_TEXT_SIZE: u16 = 32;

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
    draw_text_center(msg_str, CENTER_TEXT_SIZE, CENTER_TEXT_COLOR)
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

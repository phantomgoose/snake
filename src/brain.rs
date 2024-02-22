use crate::{Direction, Position, Snake};

pub(crate) fn predict_direction_naive(snake: &Snake, food: &Option<Position>) -> Direction {
    let current_dir = snake.direction;
    if food.is_none() {
        // no food -> no change in direction
        return current_dir;
    }

    let food_pos = food.unwrap();
    let head_pos = *snake.get_head_position();

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

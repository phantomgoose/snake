use macroquad::color::GRAY;
use macroquad::prelude::{draw_text, get_text_center, screen_height, screen_width, Color};

const CENTER_TEXT_COLOR: Color = GRAY;
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

//! Contact sheets: a grid of previews with an index number drawn on each cell (a built-in 3×5
//! bitmap digit font, so no font files), for a model to look at and refer to photos by number.

use crate::pixels::Rgba;

/// 3×5 digits, one row per byte, bit 2 = left column.
const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111],
    [0b010, 0b110, 0b010, 0b010, 0b111],
    [0b111, 0b001, 0b111, 0b100, 0b111],
    [0b111, 0b001, 0b111, 0b001, 0b111],
    [0b101, 0b101, 0b111, 0b001, 0b001],
    [0b111, 0b100, 0b111, 0b001, 0b111],
    [0b111, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b001, 0b001, 0b001, 0b001],
    [0b111, 0b101, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b111, 0b001, 0b111],
];

pub const BACKGROUND: [u8; 4] = [28, 28, 30, 255];

pub struct Canvas {
    pub img: Rgba,
}

impl Canvas {
    pub fn new(w: u32, h: u32) -> Self {
        let px = BACKGROUND.iter().copied().cycle().take(w as usize * h as usize * 4).collect();
        Self { img: Rgba { w, h, px } }
    }

    pub fn fill(&mut self, x: u32, y: u32, w: u32, h: u32, c: [u8; 4]) {
        let (cw, ch) = (self.img.w, self.img.h);
        for yy in y.min(ch)..(y.saturating_add(h)).min(ch) {
            for xx in x.min(cw)..(x.saturating_add(w)).min(cw) {
                let o = (yy as usize * cw as usize + xx as usize) * 4;
                self.img.px[o..o + 4].copy_from_slice(&c);
            }
        }
    }

    pub fn blit(&mut self, src: &Rgba, x: u32, y: u32) {
        let (cw, ch) = (self.img.w as usize, self.img.h as usize);
        for sy in 0..src.h as usize {
            let ty = y as usize + sy;
            if ty >= ch {
                break;
            }
            let n = (src.w as usize).min(cw.saturating_sub(x as usize));
            let so = sy * src.w as usize * 4;
            let to = (ty * cw + x as usize) * 4;
            self.img.px[to..to + n * 4].copy_from_slice(&src.px[so..so + n * 4]);
        }
    }

    /// Draws `n` in white on a black box at `(x, y)`; returns the box size.
    pub fn label(&mut self, n: usize, x: u32, y: u32, scale: u32) -> (u32, u32) {
        let s = scale.max(1);
        let text = n.to_string();
        let pad = s;
        let w = text.len() as u32 * 4 * s - s + 2 * pad;
        let h = 5 * s + 2 * pad;
        self.fill(x, y, w, h, [0, 0, 0, 255]);
        for (i, ch) in text.bytes().enumerate() {
            let d = DIGITS[(ch - b'0') as usize % 10];
            let ox = x + pad + i as u32 * 4 * s;
            for (row, bits) in d.iter().enumerate() {
                for col in 0..3u32 {
                    if bits & (0b100 >> col) != 0 {
                        self.fill(ox + col * s, y + pad + row as u32 * s, s, s, [255, 255, 255, 255]);
                    }
                }
            }
        }
        (w, h)
    }

    /// A placeholder for a photo that could not be shown: a grey cell with a cross.
    pub fn cross(&mut self, x: u32, y: u32, w: u32, h: u32) {
        self.fill(x, y, w, h, [70, 70, 74, 255]);
        let steps = w.max(h);
        for i in 0..steps {
            let (px, py) = (x + i * w / steps.max(1), y + i * h / steps.max(1));
            self.fill(px, py, 2, 2, [160, 60, 60, 255]);
            self.fill(x + w.saturating_sub(1) - (px - x), py, 2, 2, [160, 60, 60, 255]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_draws_white_pixels_inside_its_box() {
        let mut c = Canvas::new(40, 20);
        let (w, h) = c.label(10, 1, 1, 2);
        assert_eq!((w, h), (2 * 4 * 2 - 2 + 4, 14));
        let white = c.img.px.as_chunks::<4>().0.iter().filter(|p| p[0] == 255).count();
        // "1" has 8 cells lit, "0" has 12, each 2×2.
        assert_eq!(white, (8 + 12) * 4);
    }
}

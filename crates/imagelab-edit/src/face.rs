//! Face-aware liquify: sliders for eyes, nose, mouth, smile, face width, chin and forehead,
//! turned into PhotoCraft `filter.liquify` strokes placed and sized from each face that
//! `image.detectFaces` finds (its box and five landmarks). One call is one Liquify history step.
//!
//! Every slider is -100..100 and 0 leaves that feature alone. Sizes and distances scale with
//! the face's width, so the same value looks the same on a small face and a close-up.

use imagelab_api::{ApiError, ApiResult, Headless};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::looks::{self, Face};

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct FaceParams {
    /// Which face (0 = the most confident); all faces when absent.
    #[serde(default)]
    pub face: Option<usize>,
    #[serde(default)]
    pub eyes: f64,
    #[serde(default)]
    pub nose: f64,
    #[serde(default)]
    pub mouth: f64,
    #[serde(default)]
    pub smile: f64,
    #[serde(default)]
    pub face_width: f64,
    #[serde(default)]
    pub chin: f64,
    #[serde(default)]
    pub forehead: f64,
}

/// A stationary tool (bloat / pucker) held for a time that grows with |v|.
fn hold(tool: &str, x: f64, y: f64, size: f64, v: f64) -> Value {
    let ms = (v.abs() / 100.0 * 1400.0).round();
    json!({"tool": tool, "size": size.round().max(4.0), "density": 50, "pressure": 100, "rate": 80,
           "points": [[x.round(), y.round(), 1.0, 0], [x.round(), y.round(), 1.0, ms]]})
}

/// A forward-warp drag from (x, y) by (dx, dy), in small steps.
fn push(x: f64, y: f64, dx: f64, dy: f64, size: f64) -> Value {
    let n = 24;
    let points: Vec<Value> = (0..=n).map(|i| {
        let f = f64::from(i) / f64::from(n);
        json!([(x + dx * f).round(), (y + dy * f).round(), 1.0])
    }).collect();
    json!({"tool": "forwardWarp", "size": size.round().max(4.0), "density": 50, "pressure": 100, "rate": 80, "points": points})
}

fn clamp(v: f64) -> f64 {
    v.clamp(-100.0, 100.0)
}

/// Liquify strokes for one face.
pub fn strokes(f: &Face, p: &FaceParams) -> Vec<Value> {
    let [x, y, w, h] = f.rect;
    let cx = x + w / 2.0;
    let mut out = Vec::new();
    // Landmarks, or estimates from the box when detection gave none.
    let eye_r = f.eyes.first().copied().unwrap_or([x + w * 0.32, y + h * 0.4]);
    let eye_l = f.eyes.get(1).copied().unwrap_or([x + w * 0.68, y + h * 0.4]);
    let nose = f.nose.unwrap_or([cx, y + h * 0.6]);
    let m_r = f.mouth.first().copied().unwrap_or([x + w * 0.36, y + h * 0.78]);
    let m_l = f.mouth.get(1).copied().unwrap_or([x + w * 0.64, y + h * 0.78]);
    let mouth = [(m_r[0] + m_l[0]) / 2.0, (m_r[1] + m_l[1]) / 2.0];

    let eyes = clamp(p.eyes);
    if eyes != 0.0 {
        let tool = if eyes > 0.0 { "bloat" } else { "pucker" };
        for e in [eye_r, eye_l] {
            out.push(hold(tool, e[0], e[1], w * 0.26, eyes));
        }
    }
    let nose_v = clamp(p.nose);
    if nose_v != 0.0 {
        out.push(hold(if nose_v > 0.0 { "bloat" } else { "pucker" }, nose[0], nose[1] + h * 0.03, w * 0.26, nose_v));
    }
    let mouth_v = clamp(p.mouth);
    if mouth_v != 0.0 {
        out.push(hold(if mouth_v > 0.0 { "bloat" } else { "pucker" }, mouth[0], mouth[1], w * 0.42, mouth_v));
    }
    let smile = clamp(p.smile) / 100.0;
    if smile != 0.0 {
        let d = w * 0.07 * smile;
        out.push(push(m_r[0], m_r[1], -d * 0.4, -d, w * 0.24));
        out.push(push(m_l[0], m_l[1], d * 0.4, -d, w * 0.24));
    }
    let width = clamp(p.face_width) / 100.0;
    if width != 0.0 {
        // Cheeks and jaw: drag each side's edge in (narrower) or out (wider).
        // Five overlapping rows from below the eyes to the jaw keep the edge smooth.
        let d = w * 0.06 * width;
        let (top, bottom) = ((eye_r[1] + eye_l[1]) / 2.0 + h * 0.08, y + h * 0.88);
        for i in 0..5 {
            let yy = top + (bottom - top) * f64::from(i) / 4.0;
            // The jaw narrows toward the chin: move the drag inward there.
            let inset = w * (0.06 + 0.08 * f64::from(i) / 4.0);
            out.push(push(x + inset, yy, -d, 0.0, w * 0.42));
            out.push(push(x + w - inset, yy, d, 0.0, w * 0.42));
        }
    }
    let chin = clamp(p.chin) / 100.0;
    if chin != 0.0 {
        out.push(push(cx, y + h * 0.97, 0.0, h * 0.06 * chin, w * 0.38));
    }
    let forehead = clamp(p.forehead) / 100.0;
    if forehead != 0.0 {
        out.push(push(cx, y + h * 0.04, 0.0, -h * 0.06 * forehead, w * 0.5));
    }
    out
}

/// `edit.faceLiquify`.
pub fn apply(h: &mut Headless, p: &FaceParams) -> ApiResult<Value> {
    let faces = looks::faces(h)?;
    if faces.is_empty() {
        return Ok(json!({"applied": false, "faces": 0, "skipped": ["no faces found"]}));
    }
    let chosen: Vec<&Face> = match p.face {
        Some(i) => vec![faces.get(i).ok_or_else(|| ApiError::bad_params(format!("face {i} of {}", faces.len())))?],
        None => faces.iter().collect(),
    };
    let all: Vec<Value> = chosen.iter().flat_map(|f| strokes(f, p)).collect();
    if all.is_empty() {
        return Ok(json!({"applied": false, "faces": faces.len()}));
    }
    let n = all.len();
    crate::ops::run_allowed(h, "filter.liquify", json!({"strokes": all}))?;
    Ok(json!({"applied": true, "faces": faces.len(), "strokes": n}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face() -> Face {
        Face { rect: [100.0, 100.0, 200.0, 240.0], eyes: vec![[160.0, 190.0], [240.0, 190.0]], nose: Some([200.0, 240.0]), mouth: vec![[170.0, 290.0], [230.0, 290.0]] }
    }

    #[test]
    fn zero_sliders_make_no_strokes() {
        assert!(strokes(&face(), &FaceParams::default()).is_empty());
    }

    #[test]
    fn slimmer_face_drags_the_edges_inward() {
        let s = strokes(&face(), &FaceParams { face_width: -100.0, ..Default::default() });
        assert_eq!(s.len(), 10);
        let first = s[0]["points"].as_array().unwrap();
        let (x0, x1) = (first[0][0].as_f64().unwrap(), first.last().unwrap()[0].as_f64().unwrap());
        assert!(x1 > x0, "left edge moves right (inward): {x0} -> {x1}");
    }

    #[test]
    fn bigger_eyes_bloat_at_each_eye() {
        let s = strokes(&face(), &FaceParams { eyes: 50.0, ..Default::default() });
        assert_eq!(s.len(), 2);
        assert_eq!(s[0]["tool"], "bloat");
        assert_eq!(s[0]["points"][0][0], json!(160.0));
    }
}

//! Shape analysis on masks: connectedComponents, findContours, houghLines, dominantAngle.

use imagelab_api::{ApiError, ApiResult, Ctx};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::raster::{gaussian, gray_of, load, load_gray, put_mask, round2, shrink_gray};

/// Union-find labelling. Returns per-pixel labels (0 = background, 1.. = component) and the count.
pub fn label(bin: &[bool], w: usize, h: usize, eight: bool) -> (Vec<u32>, u32) {
    let mut labels = vec![0u32; w * h];
    let mut parent: Vec<u32> = vec![0];
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    fn union(p: &mut [u32], a: u32, b: u32) -> u32 {
        let (ra, rb) = (find(p, a), find(p, b));
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        p[hi as usize] = lo;
        lo
    }
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !bin[i] {
                continue;
            }
            let mut l = 0u32;
            let consider = |j: usize, l: &mut u32, parent: &mut Vec<u32>| {
                let lj = labels[j];
                if lj != 0 {
                    *l = if *l == 0 { lj } else { union(parent, *l, lj) };
                }
            };
            if x > 0 {
                consider(i - 1, &mut l, &mut parent);
            }
            if y > 0 {
                consider(i - w, &mut l, &mut parent);
                if eight {
                    if x > 0 {
                        consider(i - w - 1, &mut l, &mut parent);
                    }
                    if x + 1 < w {
                        consider(i - w + 1, &mut l, &mut parent);
                    }
                }
            }
            if l == 0 {
                l = parent.len() as u32;
                parent.push(l);
            }
            labels[i] = l;
        }
    }
    // Compact roots to 1..=n.
    let mut map = vec![0u32; parent.len()];
    let mut n = 0u32;
    for i in 1..parent.len() {
        let r = find(&mut parent, i as u32);
        if map[r as usize] == 0 {
            n += 1;
            map[r as usize] = n;
        }
        map[i] = map[r as usize];
    }
    labels.par_iter_mut().for_each(|l| *l = map[*l as usize]);
    (labels, n)
}

#[derive(Clone, Default)]
pub struct Comp {
    pub area: u64,
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    pub sx: f64,
    pub sy: f64,
    /// First pixel in raster order (top-left of the outer border).
    pub start: usize,
}

pub fn components(labels: &[u32], n: u32, w: usize) -> Vec<Comp> {
    let mut c = vec![Comp { x0: usize::MAX, y0: usize::MAX, start: usize::MAX, ..Default::default() }; n as usize + 1];
    for (i, &l) in labels.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let (x, y) = (i % w, i / w);
        let k = &mut c[l as usize];
        k.area += 1;
        k.x0 = k.x0.min(x);
        k.y0 = k.y0.min(y);
        k.x1 = k.x1.max(x);
        k.y1 = k.y1.max(y);
        k.sx += x as f64;
        k.sy += y as f64;
        if k.start == usize::MAX {
            k.start = i;
        }
    }
    c
}

fn comp_json(k: &Comp) -> Value {
    json!({
        "area": k.area,
        "bbox": [k.x0, k.y0, k.x1 - k.x0 + 1, k.y1 - k.y0 + 1],
        "centroid": [round2(k.sx / k.area as f64), round2(k.sy / k.area as f64)],
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ConnectedComponents {
    #[serde(alias = "img")]
    mask: String,
    #[serde(default = "eight")]
    connectivity: u8,
    #[serde(default)]
    min_area: u64,
    #[serde(default = "max_items")]
    max_components: usize,
    #[serde(default = "half")]
    threshold: u8,
}

fn eight() -> u8 {
    8
}
fn max_items() -> usize {
    100
}
fn half() -> u8 {
    127
}

pub fn connected_components(ctx: &Ctx, p: ConnectedComponents) -> ApiResult<Value> {
    if p.connectivity != 4 && p.connectivity != 8 {
        return Err(ApiError::bad_params("connectivity must be 4 or 8"));
    }
    if !(1..=1000).contains(&p.max_components) {
        return Err(ApiError::bad_params("maxComponents must be in 1..=1000"));
    }
    let (w, h, g) = load_gray(ctx, &p.mask)?;
    let bin: Vec<bool> = g.iter().map(|&v| v > p.threshold).collect();
    let (labels, n) = label(&bin, w, h, p.connectivity == 8);
    ctx.check_cancel()?;
    let comps = components(&labels, n, w);
    let mut idx: Vec<usize> = (1..comps.len()).filter(|&i| comps[i].area >= p.min_area.max(1)).collect();
    idx.sort_by(|&a, &b| comps[b].area.cmp(&comps[a].area).then(a.cmp(&b)));
    let total = idx.len();
    idx.truncate(p.max_components);
    let mut keep = vec![false; comps.len()];
    for &i in &idx {
        keep[i] = true;
    }
    let kept: Vec<u8> = labels.par_iter().map(|&l| if keep[l as usize] { 255 } else { 0 }).collect();
    let list: Vec<Value> = idx.iter().enumerate().map(|(rank, &i)| {
        let mut v = comp_json(&comps[i]);
        v["index"] = json!(rank);
        v
    }).collect();
    let mut r = put_mask(ctx, w, h, kept)?;
    r["count"] = json!(total);
    r["components"] = json!(list);
    Ok(r)
}

const DIRS: [(isize, isize); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];

/// Outer border of the component labelled `lab` starting at its first raster pixel
/// (Moore-neighbour tracing with Jacob's stopping criterion), clockwise in image coordinates.
pub fn trace(labels: &[u32], w: usize, h: usize, start: usize, lab: u32) -> Vec<(usize, usize)> {
    let is = |x: isize, y: isize| x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && labels[y as usize * w + x as usize] == lab;
    let s = ((start % w) as isize, (start / w) as isize);
    let mut pts = vec![(s.0 as usize, s.1 as usize)];
    let next = |p: (isize, isize), from: usize| -> Option<(usize, (isize, isize))> {
        (0..8).map(|k| (from + k) % 8).find_map(|d| {
            let q = (p.0 + DIRS[d].0, p.1 + DIRS[d].1);
            is(q.0, q.1).then_some((d, q))
        })
    };
    let Some((d0, first)) = next(s, 4) else { return pts };
    let (mut p, mut d) = (first, d0);
    let limit = 4 * labels.len() + 16;
    for _ in 0..limit {
        if p == s {
            // Stop when leaving the start the same way as the first time.
            match next(p, (d + 6 - (d & 1)) % 8) {
                Some((nd, _)) if nd == d0 => break,
                _ => {}
            }
        }
        pts.push((p.0 as usize, p.1 as usize));
        let Some((nd, q)) = next(p, (d + 6 - (d & 1)) % 8) else { break };
        p = q;
        d = nd;
    }
    pts
}

fn perpendicular(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = dx.hypot(dy);
    if len < 1e-12 { (p.0 - a.0).hypot(p.1 - a.1) } else { ((p.0 - a.0) * dy - (p.1 - a.1) * dx).abs() / len }
}

/// Douglas-Peucker on an open polyline.
pub fn simplify(pts: &[(f64, f64)], eps: f64) -> Vec<(f64, f64)> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let mut keep = vec![false; pts.len()];
    keep[0] = true;
    keep[pts.len() - 1] = true;
    let mut stack = vec![(0usize, pts.len() - 1)];
    while let Some((a, b)) = stack.pop() {
        let (mut best, mut bi) = (0.0, 0);
        for i in a + 1..b {
            let d = perpendicular(pts[i], pts[a], pts[b]);
            if d > best {
                best = d;
                bi = i;
            }
        }
        if best > eps && bi > a {
            keep[bi] = true;
            stack.push((a, bi));
            stack.push((bi, b));
        }
    }
    pts.iter().zip(&keep).filter(|&(_, &k)| k).map(|(p, _)| *p).collect()
}

/// Simplify a closed contour to at most `max` points (raising epsilon as needed).
pub fn simplify_closed(pts: &[(usize, usize)], max: usize) -> Vec<(f64, f64)> {
    let f: Vec<(f64, f64)> = pts.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    if f.len() <= 2 {
        return f;
    }
    // Split at the point farthest from the first so both halves are open polylines.
    let far = (0..f.len()).max_by(|&i, &j| {
        let di = (f[i].0 - f[0].0).hypot(f[i].1 - f[0].1);
        let dj = (f[j].0 - f[0].0).hypot(f[j].1 - f[0].1);
        di.total_cmp(&dj)
    }).unwrap_or(0);
    let mut eps = 0.7;
    loop {
        let a = simplify(&f[..=far], eps);
        let mut tail: Vec<(f64, f64)> = f[far..].to_vec();
        tail.push(f[0]);
        let b = simplify(&tail, eps);
        let mut out = a;
        out.extend_from_slice(&b[1..b.len().saturating_sub(1)]);
        if out.len() <= max || eps > 1e6 {
            return out;
        }
        eps *= 1.5;
    }
}

fn perimeter(pts: &[(usize, usize)]) -> f64 {
    if pts.len() < 2 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        s += (a.0 as f64 - b.0 as f64).hypot(a.1 as f64 - b.1 as f64);
    }
    s
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct FindContours {
    #[serde(alias = "img")]
    mask: String,
    #[serde(default)]
    min_area: u64,
    #[serde(default = "max_items")]
    max_contours: usize,
    #[serde(default = "sixty_four")]
    max_points: usize,
    #[serde(default = "half")]
    threshold: u8,
}

fn sixty_four() -> usize {
    64
}

pub fn find_contours(ctx: &Ctx, p: FindContours) -> ApiResult<Value> {
    if !(1..=1000).contains(&p.max_contours) || !(3..=256).contains(&p.max_points) {
        return Err(ApiError::bad_params("maxContours must be in 1..=1000 and maxPoints in 3..=256"));
    }
    let (w, h, g) = load_gray(ctx, &p.mask)?;
    let bin: Vec<bool> = g.iter().map(|&v| v > p.threshold).collect();
    let (labels, n) = label(&bin, w, h, true);
    ctx.check_cancel()?;
    let comps = components(&labels, n, w);
    let mut idx: Vec<usize> = (1..comps.len()).filter(|&i| comps[i].area >= p.min_area.max(1)).collect();
    idx.sort_by(|&a, &b| comps[b].area.cmp(&comps[a].area).then(a.cmp(&b)));
    let total = idx.len();
    idx.truncate(p.max_contours);
    let contours: Vec<Value> = idx
        .par_iter()
        .map(|&i| {
            let k = &comps[i];
            let border = trace(&labels, w, h, k.start, i as u32);
            let pts = simplify_closed(&border, p.max_points);
            let mut v = comp_json(k);
            v["perimeter"] = json!(round2(perimeter(&border)));
            v["points"] = json!(pts.iter().map(|&(x, y)| [x as i64, y as i64]).collect::<Vec<_>>());
            v
        })
        .collect();
    Ok(json!({"count": total, "contours": contours, "width": w, "height": h}))
}

/// Edge points of a mask (value > 127).
fn points_of(g: &[u8], w: usize) -> Vec<(f32, f32)> {
    g.iter().enumerate().filter(|&(_, &v)| v > 127).map(|(i, _)| ((i % w) as f32, (i / w) as f32)).collect()
}

pub struct Line {
    pub rho: f64,
    pub theta: f64,
    pub votes: u32,
}

/// Standard Hough transform over `thetas` (radians). Returns local maxima with ≥ `threshold`
/// votes (or ≥ 30 % of the peak when `threshold` is None), strongest first.
pub fn hough(pts: &[(f32, f32)], w: usize, h: usize, rho_step: f64, thetas: &[f64], threshold: Option<u32>, max_lines: usize) -> Vec<Line> {
    let diag = ((w * w + h * h) as f64).sqrt();
    let nr = ((2.0 * diag) / rho_step).ceil() as usize + 1;
    let nt = thetas.len();
    // Accumulator stored theta-major (one column per theta) so thetas vote in parallel.
    let acc: Vec<Vec<u32>> = thetas
        .par_iter()
        .map(|&t| {
            let (c, s) = (t.cos() as f32, t.sin() as f32);
            let mut col = vec![0u32; nr];
            for &(x, y) in pts {
                let r = ((x * c + y * s) as f64 + diag) / rho_step;
                let ri = r.round() as usize;
                if ri < nr {
                    col[ri] += 1;
                }
            }
            col
        })
        .collect();
    let peak = acc.iter().flat_map(|c| c.iter()).copied().max().unwrap_or(0);
    let thr = threshold.unwrap_or(((peak as f64 * 0.3).ceil() as u32).max(10)).max(1);
    let mut cands = Vec::new();
    for t in 0..nt {
        for r in 0..nr {
            let v = acc[t][r];
            if v < thr {
                continue;
            }
            let mut is_max = true;
            'n: for dt in -1i64..=1 {
                for dr in -1i64..=1 {
                    if dt == 0 && dr == 0 {
                        continue;
                    }
                    let (tt, rr) = (t as i64 + dt, r as i64 + dr);
                    if tt < 0 || rr < 0 || tt >= nt as i64 || rr >= nr as i64 {
                        continue;
                    }
                    let u = acc[tt as usize][rr as usize];
                    if u > v || (u == v && (dt, dr) < (0, 0)) {
                        is_max = false;
                        break 'n;
                    }
                }
            }
            if is_max {
                cands.push(Line { rho: r as f64 * rho_step - diag, theta: thetas[t], votes: v });
            }
        }
    }
    cands.sort_by_key(|c| std::cmp::Reverse(c.votes));
    let mut out: Vec<Line> = Vec::new();
    for c in cands {
        if out.iter().any(|o| (o.rho - c.rho).abs() < 4.0 * rho_step && (o.theta - c.theta).abs() < 2f64.to_radians()) {
            continue;
        }
        out.push(c);
        if out.len() >= max_lines {
            break;
        }
    }
    out
}

/// Extent of the points lying within `tol` px of a line, as segment endpoints.
type Segment = ((f64, f64), (f64, f64), usize);

fn support(pts: &[(f32, f32)], l: &Line, tol: f64) -> Option<Segment> {
    let (c, s) = (l.theta.cos(), l.theta.sin());
    let (mut lo, mut hi, mut n) = (f64::INFINITY, f64::NEG_INFINITY, 0usize);
    for &(x, y) in pts {
        let (x, y) = (x as f64, y as f64);
        if (x * c + y * s - l.rho).abs() <= tol {
            let t = -x * s + y * c;
            lo = lo.min(t);
            hi = hi.max(t);
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    let (x0, y0) = (l.rho * c, l.rho * s);
    let a = (x0 - lo * s + 0.0, y0 + lo * c + 0.0);
    let b = (x0 - hi * s + 0.0, y0 + hi * c + 0.0);
    // Left end first (top end first for vertical lines).
    let (a, b) = if (a.0.round(), a.1.round()) <= (b.0.round(), b.1.round()) { (a, b) } else { (b, a) };
    Some((a, b, n))
}

/// Orientation of a line with normal angle `theta`, in degrees, counterclockwise on screen from
/// horizontal, in (−90, 90].
fn line_angle(theta: f64) -> f64 {
    // Direction vector (-sin, cos) in image coords (y down); on screen y is flipped.
    let (dx, dy) = (-theta.sin(), -theta.cos());
    let mut a = dy.atan2(dx).to_degrees();
    while a <= -90.0 {
        a += 180.0;
    }
    while a > 90.0 {
        a -= 180.0;
    }
    a
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct HoughLines {
    #[serde(alias = "img")]
    mask: String,
    #[serde(default = "onef")]
    rho: f64,
    /// Degrees.
    #[serde(default = "onef")]
    theta: f64,
    #[serde(default)]
    threshold: Option<u32>,
    #[serde(default = "twenty")]
    max_lines: usize,
    #[serde(default)]
    min_angle: Option<f64>,
    #[serde(default)]
    max_angle: Option<f64>,
}

fn onef() -> f64 {
    1.0
}
fn twenty() -> usize {
    20
}

/// Most edge points the Hough transform votes with; denser masks are subsampled.
const MAX_HOUGH_POINTS: usize = 2_000_000;

pub fn hough_lines(ctx: &Ctx, p: HoughLines) -> ApiResult<Value> {
    if !(0.5..=100.0).contains(&p.rho) || !(0.05..=10.0).contains(&p.theta) || !(1..=200).contains(&p.max_lines) {
        return Err(ApiError::bad_params("rho in 0.5..=100 px, theta in 0.05..=10 degrees, maxLines in 1..=200"));
    }
    let (w, h, g) = load_gray(ctx, &p.mask)?;
    let mut pts = points_of(&g, w);
    if pts.len() > MAX_HOUGH_POINTS {
        let step = pts.len().div_ceil(MAX_HOUGH_POINTS);
        pts = pts.into_iter().step_by(step).collect();
    }
    let nt = (180.0 / p.theta).round() as usize;
    let thetas: Vec<f64> = (0..nt).map(|i| (i as f64 * p.theta).to_radians()).collect();
    ctx.check_cancel()?;
    let mut lines = hough(&pts, w, h, p.rho, &thetas, p.threshold, p.max_lines * 4);
    lines.retain(|l| {
        let a = line_angle(l.theta);
        p.min_angle.is_none_or(|m| a >= m) && p.max_angle.is_none_or(|m| a <= m)
    });
    lines.truncate(p.max_lines);
    let out: Vec<Value> = lines
        .iter()
        .map(|l| {
            let mut v = json!({"rho": round2(l.rho), "theta": (l.theta * 1e5).round() / 1e5, "angle": round2(line_angle(l.theta)), "votes": l.votes});
            if let Some(((x1, y1), (x2, y2), n)) = support(&pts, l, 1.5 * p.rho) {
                v["x1"] = json!(x1.round() + 0.0);
                v["y1"] = json!(y1.round() + 0.0);
                v["x2"] = json!(x2.round() + 0.0);
                v["y2"] = json!(y2.round() + 0.0);
                v["support"] = json!(n);
            }
            v
        })
        .collect();
    Ok(json!({"lines": out, "edgePoints": pts.len()}))
}

/// Total-least-squares angle (degrees, screen CCW) of points near a line.
fn refine_angle(pts: &[(f32, f32)], l: &Line, tol: f64) -> Option<(f64, usize)> {
    let (c, s) = (l.theta.cos(), l.theta.sin());
    let near: Vec<(f64, f64)> = pts.iter().map(|&(x, y)| (x as f64, y as f64)).filter(|&(x, y)| (x * c + y * s - l.rho).abs() <= tol).collect();
    if near.len() < 10 {
        return None;
    }
    let n = near.len() as f64;
    let (mx, my) = near.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (mx, my) = (mx / n, my / n);
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for &(x, y) in &near {
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
        sxy += (x - mx) * (y - my);
    }
    // Principal direction in image coords; flip y for screen.
    let phi = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let mut a = (-phi).to_degrees();
    while a <= -90.0 {
        a += 180.0;
    }
    while a > 90.0 {
        a -= 180.0;
    }
    Some((a, near.len()))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DominantAngle {
    #[serde(alias = "mask")]
    img: String,
    #[serde(default = "thirty")]
    max_tilt: f64,
    #[serde(default)]
    mode: Option<String>,
}

fn thirty() -> f64 {
    30.0
}

pub fn dominant_angle(ctx: &Ctx, p: DominantAngle) -> ApiResult<Value> {
    if !(0.5..=45.0).contains(&p.max_tilt) {
        return Err(ApiError::bad_params("maxTilt must be in 0.5..=45 degrees"));
    }
    let (horizontal, vertical) = match p.mode.as_deref().unwrap_or("horizontal") {
        "horizontal" | "horizon" => (true, false),
        "vertical" => (false, true),
        "both" => (true, true),
        _ => return Err(ApiError::bad_params("mode must be horizontal, vertical or both")),
    };
    let src = load(ctx, &p.img)?;
    let (w, h) = (src.w(), src.h());
    let g = gray_of(&src);
    let (g, gw, gh, _) = shrink_gray(&g, w, h, 1024);
    let edges = if src.is_mask() {
        crate::morph::morph_buf(&g, gw, gh, crate::morph::MorphOp::Gradient, 3, crate::morph::Shape::Rect, 1)
    } else {
        let b = gaussian(&g, gw, gh, 1, None, 1.5);
        crate::filter::canny_buf(&b, gw, gh, 20.0, 60.0, true)
    };
    ctx.check_cancel()?;
    let pts = points_of(&edges, gw);
    if pts.len() < 20 {
        return Ok(json!({"angle": 0.0, "correction": 0.0, "confidence": 0.0, "note": "too few edges to measure a tilt"}));
    }
    let step = 0.1f64;
    let n = (p.max_tilt / step).round() as i64;
    let mut thetas = Vec::new();
    for i in -n..=n {
        let d = i as f64 * step;
        if horizontal {
            thetas.push((90.0 + d).to_radians());
        }
        if vertical {
            thetas.push(d.rem_euclid(180.0).to_radians());
        }
    }
    let lines = hough(&pts, gw, gh, 1.0, &thetas, None, 8);
    let Some(best) = lines.first() else {
        return Ok(json!({"angle": 0.0, "correction": 0.0, "confidence": 0.0, "note": "no straight line found"}));
    };
    let coarse = line_angle(best.theta);
    let is_vertical = coarse.abs() > 45.0;
    let (mut angle, support) = refine_angle(&pts, best, 2.0).unwrap_or((coarse, best.votes as usize));
    if (angle - coarse).abs() > 1.0 && (angle - coarse).abs() < 179.0 {
        angle = coarse;
    }
    // Tilt relative to the nearest axis.
    let tilt = if is_vertical { if angle > 0.0 { angle - 90.0 } else { angle + 90.0 } } else { angle };
    let long = gw.max(gh) as f64;
    let confidence = (support as f64 / long).min(1.0);
    let others: Vec<Value> = lines.iter().take(5).map(|l| json!({"angle": round2(line_angle(l.theta)), "votes": l.votes})).collect();
    Ok(json!({
        "angle": round2(tilt),
        "correction": round2(-tilt),
        "lineAngle": round2(angle),
        "kind": if is_vertical { "vertical" } else { "horizontal" },
        "confidence": round2(confidence),
        "candidates": others,
        "measuredAt": [gw, gh],
    }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn labels_blobs() {
        // Three blobs, one diagonal pair joined only under 8-connectivity.
        let w = 10;
        let mut b = vec![false; 100];
        for &(x, y) in &[(1, 1), (2, 1), (1, 2), (2, 2), (6, 6), (7, 7), (9, 0)] {
            b[y * w + x] = true;
        }
        assert_eq!(label(&b, w, 10, true).1, 3);
        assert_eq!(label(&b, w, 10, false).1, 4);
        // U shape merges two provisional labels.
        let u = ["#.#", "#.#", "###"];
        let ub: Vec<bool> = u.iter().flat_map(|r| r.chars().map(|c| c == '#')).collect();
        assert_eq!(label(&ub, 3, 3, false).1, 1);
    }

    #[test]
    fn trace_square_border() {
        let w = 8;
        let mut b = vec![false; 64];
        for y in 2..6 {
            for x in 2..6 {
                b[y * w + x] = true;
            }
        }
        let (l, _) = label(&b, w, 8, true);
        let t = trace(&l, w, 8, 2 * w + 2, 1);
        assert_eq!(t.len(), 12, "{t:?}");
        assert!((perimeter(&t) - 12.0).abs() < 1e-9);
        let s = simplify_closed(&t, 64);
        assert_eq!(s.len(), 4, "{s:?}");
        // Single pixel and a thin line.
        let mut one = vec![false; 9];
        one[4] = true;
        let (l1, _) = label(&one, 3, 3, true);
        assert_eq!(trace(&l1, 3, 3, 4, 1).len(), 1);
    }

    #[test]
    fn angle_conventions() {
        assert!((line_angle(90f64.to_radians())).abs() < 1e-9);
        // A line rising to the right on screen (y decreasing as x grows) has a positive angle.
        let pts: Vec<(f32, f32)> = (0..100).map(|x| (x as f32, 100.0 - x as f32 * 0.1)).collect();
        let l = hough(&pts, 120, 120, 1.0, &(800..1000).map(|i| (i as f64 / 10.0).to_radians()).collect::<Vec<_>>(), None, 1);
        let a = line_angle(l[0].theta);
        assert!((a - 5.71).abs() < 0.3, "{a}");
        let (r, _) = refine_angle(&pts, &l[0], 2.0).unwrap();
        assert!((r - 5.71).abs() < 0.05, "{r}");
    }
}

//! # imagelab-cv
//!
//! The `cv` area (docs/INTERFACES.md §7): OpenCV-style image operations on handles, in pure
//! Rust. Images (`img:N`, 8-bit RGBA) and masks (`mask:N`, 8-bit coverage) live in the
//! [`imagelab_api::Handles`] arena; every function takes handle strings plus parameters and
//! returns new handles plus small JSON, never pixel arrays.
//!
//! Names and parameters follow OpenCV (`cv.gaussianBlur {img, ksize, sigma}`,
//! `cv.threshold {img, thresh, type}`) so models already know them; departures are spelled out
//! in each function's description, which is what `help("cv")` shows.
//!
//! [`image_to_png`], [`mask_to_png`] and [`decode_image`] are plain helpers for other crates.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod color;
pub mod extra;
pub mod filter;
pub mod geom;
pub mod io;
pub mod mask;
pub mod morph;
pub mod raster;
pub mod shapes;
pub mod thresh;

use imagelab_api::{Effects, Module, Registry};
use serde_json::{Value, json};

pub use io::{decode_image, image_to_png, mask_to_png};

/// Registers the `cv` area.
pub struct CvModule;

const AREA: &str = "cv";

fn obj(props: Value, required: &[&str]) -> Value {
    json!({"type": "object", "properties": props, "required": required, "additionalProperties": false})
}

fn h(desc: &str) -> Value {
    json!({"type": "string", "pattern": "^(img|mask):[0-9]+$", "description": desc})
}

fn img_any() -> Value {
    h("image handle img:N (a mask:N also works; the result is then a mask)")
}

fn num(desc: &str) -> Value {
    json!({"type": "number", "description": desc})
}

fn int(desc: &str) -> Value {
    json!({"type": "integer", "description": desc})
}

fn interp() -> Value {
    json!({"type": "string", "enum": ["nearest", "linear", "area", "cubic"], "description": "interpolation"})
}

fn border() -> Value {
    json!({
        "borderMode": {"type": "string", "enum": ["constant", "replicate", "reflect"], "description": "pixels outside the source: constant (borderValue, default), replicate edge, reflect"},
        "borderValue": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 255}, "minItems": 4, "maxItems": 4, "description": "[r,g,b,a] for constant borders; default [0,0,0,0] (transparent)"}
    })
}

fn merge(mut a: Value, b: Value) -> Value {
    if let (Some(a), Value::Object(b)) = (a.as_object_mut(), b) {
        a.extend(b);
    }
    a
}

const IMG_OUT: &str = "Returns {img, width, height}.";
const MASK_OUT: &str = "Returns {mask, width, height, coverage} (coverage = mean of the mask, 0..1).";

impl Module for CvModule {
    fn register(&self, r: &mut Registry) {
        r.area(
            AREA,
            "OpenCV-style image ops on handles (img:N RGBA, mask:N coverage): read/write, resize/rotate/warp, blur, canny, threshold, morphology, contours, hough lines, dominantAngle (horizon), template matching, inpaint, mask algebra, toDocumentLayer",
        );
        register_io(r);
        register_geom(r);
        register_color(r);
        register_filters(r);
        register_thresh_morph(r);
        register_shapes(r);
        register_masks(r);
        register_extra(r);
    }
}

fn register_io(r: &mut Registry) {
    r.function(AREA, "read", "Read an image file from the workspace into an img handle")
        .description(&format!("Decodes PNG, JPEG, WebP, TIFF, GIF, BMP, … (PhotoCraft's codecs) to 8-bit RGBA, turned upright by its EXIF orientation. `path` is relative to the workspace (e.g. \"library/IMG_1.jpg\"); absolute paths and `..` are refused. {IMG_OUT} plus `format`."))
        .params(obj(json!({"path": {"type": "string", "description": "workspace-relative file path"}}), &["path"]))
        .typed(io::read)
        .register();
    let write_params = obj(
        json!({
            "img": h("img:N or mask:N to save (masks are saved as grayscale)"),
            "path": {"type": "string", "description": "workspace-relative; the extension picks the format (.png .jpg .webp .tif …). Goes under exports/ unless it starts with exports/ or .imagelab/tmp/"},
            "quality": {"type": "integer", "minimum": 1, "maximum": 100, "description": "JPEG/WebP quality (default 90 JPEG, lossless WebP)"},
            "overwrite": {"type": "boolean", "description": "cv.write never overwrites; this only changes the error hint. Use cv.overwrite."}
        }),
        &["img", "path"],
    );
    r.function(AREA, "write", "Save an img/mask to a new file under exports/")
        .description("Encodes and writes a new file. Refuses (code `exists`) if the file is already there: overwriting is cv.overwrite, which is Destructive. Returns {path, bytes, format}.")
        .params(write_params.clone())
        .effects(Effects::Revertible)
        .typed(io::write)
        .register();
    r.function(AREA, "overwrite", "Save an img/mask, replacing an existing file (Destructive)")
        .description("Same as cv.write but replaces an existing file under exports/ or .imagelab/tmp/. The old file cannot be recovered. Returns {path, bytes, format, overwrote}.")
        .params(write_params)
        .effects(Effects::Destructive)
        .typed(io::overwrite)
        .register();
    r.function(AREA, "fromDocument", "Render the open PhotoCraft document into an img handle")
        .description(&format!("Flattened render of the active (or `document` index) PhotoCraft document. maxSide bounds the long side (default 2048; 0 = full size). {IMG_OUT}"))
        .params(obj(json!({"maxSide": int("long side in px; 0 = full resolution; default 2048"), "document": int("document index; default the active one")}), &[]))
        .typed(io::from_document)
        .register();
    r.function(AREA, "toDocumentLayer", "Place an img into the open PhotoCraft document as a new layer")
        .description("Adds the image as a new pixel layer (or embedded smart object with smartObject:true) above the active layer, centred and unscaled; an image the size of the document lands exactly over it. Undoable from the document's history. Returns {placed: {layer, bounds, …}, name}.")
        .params(obj(
            json!({"img": h("img:N"), "name": {"type": "string", "description": "layer name (default the handle)"}, "smartObject": {"type": "boolean"}, "document": int("document index; default the active one")}),
            &["img"],
        ))
        .effects(Effects::Revertible)
        .typed(io::to_document_layer)
        .register();
    r.function(AREA, "info", "Size and kind of a handle, and arena usage")
        .description("With img (or mask): {kind, width, height, channels, bytes, opaque | coverage, arena}. Without: {arena: {handles, usedBytes}}.")
        .params(obj(json!({"img": h("img:N or mask:N (optional)")}), &[]))
        .typed(io::info)
        .register();
    r.function(AREA, "free", "Free handles you no longer need")
        .description("Releases images/masks from the arena (it is bounded; a full arena makes calls fail with too_large). Handles made in a script run are freed when it ends anyway. Returns {freed, missing, usedBytes}.")
        .params(obj(json!({"handles": {"type": "array", "items": {"type": "string"}, "maxItems": 10000}}), &["handles"]))
        .typed(io::free)
        .register();
}

fn register_geom(r: &mut Registry) {
    r.function(AREA, "resize", "Resize to width/height, by scale, or to fit maxSide")
        .description(&format!("cv2.resize. Give width and height; or one of them (aspect kept); or scale (fx=fy); or maxSide (shrink only, aspect kept). interpolation: nearest | linear | area | cubic; default area when shrinking, linear when enlarging. Transparent pixels are filtered premultiplied. {IMG_OUT}"))
        .params(obj(
            json!({"img": img_any(), "width": int("px"), "height": int("px"), "scale": num("factor, e.g. 0.5"), "maxSide": int("fit the long side, never enlarges"), "interpolation": interp()}),
            &["img"],
        ))
        .typed(geom::resize)
        .register();
    r.function(AREA, "crop", "Crop a rectangle x, y, w, h")
        .description(&format!("Clipped to the image; fails if the rectangle misses it. {IMG_OUT} plus the actual `rect`."))
        .params(obj(json!({"img": img_any(), "x": int("left"), "y": int("top"), "w": int("width"), "h": int("height")}), &["img", "x", "y", "w", "h"]))
        .typed(geom::crop)
        .register();
    r.function(AREA, "rotate", "Rotate by an angle in degrees (counterclockwise positive)")
        .description(&format!("Like cv2.getRotationMatrix2D + warpAffine about the centre: positive angle turns the picture counterclockwise. Multiples of 90 are exact and swap width/height. Otherwise the canvas keeps its size (corners filled with borderValue, transparent by default), or expand:true grows it to fit everything, or crop:true returns the largest centred rectangle of the original aspect with no empty corners (use this to straighten a horizon: angle = cv.dominantAngle(...).correction). {IMG_OUT}"))
        .params(obj(
            merge(json!({"img": img_any(), "angle": num("degrees, counterclockwise positive"), "expand": {"type": "boolean"}, "crop": {"type": "boolean"}, "interpolation": interp()}), border()),
            &["img", "angle"],
        ))
        .typed(geom::rotate)
        .register();
    r.function(AREA, "flip", "Mirror horizontally, vertically or both")
        .description(&format!("mode: horizontal (left-right, default) | vertical | both; or OpenCV flipCode 1 | 0 | -1. {IMG_OUT}"))
        .params(obj(json!({"img": img_any(), "mode": {"type": "string", "enum": ["horizontal", "vertical", "both"]}, "flipCode": {"type": "integer", "enum": [1, 0, -1]}}), &["img"]))
        .typed(geom::flip)
        .register();
    r.function(AREA, "warpAffine", "Apply a 2x3 affine matrix")
        .description(&format!("cv2.warpAffine: matrix [a, b, tx, c, d, ty] maps source to destination (dst = M·src); inverseMap:true means it maps destination to source. width/height default to the source size. interpolation nearest | linear (default) | cubic. {IMG_OUT}"))
        .params(obj(
            merge(
                json!({"img": img_any(), "matrix": {"type": "array", "items": {"type": "number"}, "minItems": 6, "maxItems": 6}, "width": int("output width"), "height": int("output height"), "inverseMap": {"type": "boolean"}, "interpolation": interp()}),
                border(),
            ),
            &["img", "matrix"],
        ))
        .typed(geom::warp_affine)
        .register();
    let quad = json!({"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}, "minItems": 4, "maxItems": 4});
    r.function(AREA, "warpPerspective", "Apply a homography (3x3) or map a quad to a rectangle")
        .description(&format!("cv2.warpPerspective. Either matrix (9 numbers, row-major, src→dst) or srcQuad [[x,y]×4] (top-left, top-right, bottom-right, bottom-left) with optional dstQuad (cv2.getPerspectiveTransform). With srcQuad alone the quad is rectified to an upright rectangle sized from its edges (document/whiteboard/facade correction). width/height default to the source size or the rectified size. {IMG_OUT} plus the `matrix` used."))
        .params(obj(
            merge(
                json!({"img": img_any(), "matrix": {"type": "array", "items": {"type": "number"}, "minItems": 9, "maxItems": 9}, "srcQuad": quad.clone(), "dstQuad": quad, "width": int("output width"), "height": int("output height"), "inverseMap": {"type": "boolean"}, "interpolation": interp()}),
                border(),
            ),
            &["img"],
        ))
        .typed(geom::warp_perspective)
        .register();
}

fn register_color(r: &mut Registry) {
    r.function(AREA, "cvtColor", "Convert colour space (gray, hsv, lab, ycrcb, back to rgb)")
        .description(&format!("cv2.cvtColor on the RGB channels; alpha is kept and the result is still an RGBA img. Encodings (OpenCV 8-bit): gray → R=G=B=luma; hsv → R=H (0..179, degrees/2), G=S, B=V (0..255); lab → R=L*·255/100, G=a*+128, B=b*+128 (D65); ycrcb → R=Y, G=Cr, B=Cb; bgr swaps R and B. hsv2rgb, lab2rgb, ycrcb2rgb convert back. OpenCV names like COLOR_RGB2HSV work too. To threshold one channel use cv.extractChannel or `channel` on cv.threshold. {IMG_OUT}"))
        .params(obj(json!({"img": h("img:N"), "code": {"type": "string", "description": "gray | hsv | lab | ycrcb | bgr | hsv2rgb | lab2rgb | ycrcb2rgb"}}), &["img", "code"]))
        .typed(color::cvt_color)
        .register();
    let channel = json!({"type": "string", "enum": ["gray", "r", "g", "b", "a", "h", "s", "v", "l"], "description": "gray (luma, default), r, g, b, a, h (hue 0..360° as 0..255), s, v, l (L* 0..100 as 0..255)"});
    r.function(AREA, "extractChannel", "One channel of an image as a mask")
        .description(&format!("cv2.extractChannel / split, plus hue, saturation, value and L*. {MASK_OUT}"))
        .params(obj(json!({"img": h("img:N"), "channel": channel.clone()}), &["img"]))
        .typed(color::extract_channel)
        .register();
    r.function(AREA, "adjust", "Quick brightness/contrast/saturation/gamma (for previews)")
        .description(&format!("Simple global adjustment on a copy, for previews and analysis; real edits go through edit.* on the document. brightness, contrast, saturation in -100..100 (0 = unchanged), gamma 0.1..10 (1 = unchanged, >1 brightens midtones). {IMG_OUT}"))
        .params(obj(json!({"img": h("img:N"), "brightness": num("-100..100"), "contrast": num("-100..100"), "saturation": num("-100..100 (-100 = gray)"), "gamma": num("0.1..10")}), &["img"]))
        .typed(color::adjust)
        .register();
    r.function(AREA, "histogram", "Histogram plus exposure stats per channel")
        .description("cv2.calcHist for one or more channels (default gray), optionally only where mask ≥ 128. Returns {bins, channels: {<name>: {hist: [counts], stats: {count, mean, std, min, max, p1, p5, median, p95, p99, clippedLow, clippedHigh}}}}; clippedLow/High = fraction of pixels at 0..2 / 253..255 (crushed shadows / blown highlights). Use bins: 16-32 for a compact look.")
        .params(obj(
            json!({"img": img_any(), "channel": channel.clone(), "channels": {"type": "array", "items": channel, "maxItems": 9}, "bins": {"type": "integer", "minimum": 1, "maximum": 256}, "mask": h("mask:N restricting the count (optional)")}),
            &["img"],
        ))
        .typed(color::histogram)
        .register();
    r.function(AREA, "meanStd", "Mean and standard deviation per channel")
        .description("cv2.meanStdDev, optionally only where mask ≥ 128. Returns {count, mean: [r,g,b,a], std: [r,g,b,a], channels} for images, {count, mean: [v], std: [v]} for masks.")
        .params(obj(json!({"img": img_any(), "mask": h("mask:N (optional)")}), &["img"]))
        .typed(color::mean_std)
        .register();
}

fn register_filters(r: &mut Registry) {
    r.function(AREA, "gaussianBlur", "Gaussian blur")
        .description(&format!("cv2.GaussianBlur: ksize (odd, ≤255) and/or sigma (px). Only sigma: ksize = 6σ+1. Only ksize: σ from OpenCV's formula. Large kernels (>31) use a three-pass box approximation (fast at any radius). Reflect-101 borders. Works on masks too (soften a mask). {IMG_OUT}"))
        .params(obj(json!({"img": img_any(), "ksize": {"type": "integer", "minimum": 1, "maximum": 255, "description": "odd kernel size"}, "sigma": num("standard deviation in px (0..500)")}), &["img"]))
        .typed(filter::gaussian_blur)
        .register();
    r.function(AREA, "medianBlur", "Median filter (removes salt-and-pepper noise, keeps edges)")
        .description(&format!("cv2.medianBlur per channel, ksize odd 3..31. {IMG_OUT}"))
        .params(obj(json!({"img": img_any(), "ksize": {"type": "integer", "minimum": 3, "maximum": 31}}), &["img"]))
        .typed(filter::median_blur)
        .register();
    r.function(AREA, "bilateralFilter", "Edge-preserving smoothing")
        .description(&format!("cv2.bilateralFilter: d = window diameter (1..25, default 9), sigmaColor (default 75; colour distance is the sum of |ΔR|+|ΔG|+|ΔB|), sigmaSpace (default 75). Alpha unchanged. Cost grows with d²: keep d ≤ 9 on full-size photos. {IMG_OUT}"))
        .params(obj(json!({"img": img_any(), "d": {"type": "integer", "minimum": 1, "maximum": 25}, "sigmaColor": num("colour sigma"), "sigmaSpace": num("space sigma, px")}), &["img"]))
        .typed(filter::bilateral_filter)
        .register();
    r.function(AREA, "sobel", "Sobel gradient image (3x3)")
        .description("cv2.Sobel on the gray image, returned viewable: |gradient|/4·scale clamped to 0..255 as a gray img (a mask for mask input). dx:1 → vertical edges, dy:1 → horizontal edges, neither (or both) → gradient magnitude. Returns {img|mask, width, height, maxGradient}.")
        .params(obj(json!({"img": img_any(), "dx": {"type": "integer", "enum": [0, 1]}, "dy": {"type": "integer", "enum": [0, 1]}, "scale": num("multiplier (0..100], default 1")}), &["img"]))
        .typed(filter::sobel)
        .register();
    r.function(AREA, "canny", "Canny edge detector → edge mask")
        .description(&format!("cv2.Canny(img, low, high): 3x3 Sobel, non-maximum suppression, hysteresis; gradient magnitude |gx|+|gy| (or L2 with l2gradient:true), so thresholds are on the same scale as OpenCV's (typical photo: 50/150). Like OpenCV it does not blur first: pass sigma (e.g. 1.4) or run cv.gaussianBlur on noisy photos. Edges are 255 in the mask. {MASK_OUT}"))
        .params(obj(json!({"img": img_any(), "low": num("lower hysteresis threshold"), "high": num("upper threshold"), "sigma": num("optional Gaussian pre-blur sigma, px"), "l2gradient": {"type": "boolean"}}), &["img", "low", "high"]))
        .typed(filter::canny)
        .register();
    r.function(AREA, "laplacianVariance", "Sharpness score: variance of the Laplacian")
        .description("Variance of cv2.Laplacian (4-neighbour) on the gray image; higher = sharper. Scores depend on resolution and content, so compare photos at the same maxSide (e.g. 1024) and ideally the same subject; roughly < 50 is blurry at 1024 px. roi: a mask to score only the subject. Returns {variance, pixels, measuredAt}.")
        .params(obj(json!({"img": img_any(), "maxSide": int("downscale first so the long side is at most this"), "roi": h("mask:N (optional)")}), &["img"]))
        .typed(filter::laplacian_variance)
        .register();
    r.function(AREA, "denoise", "Reduce noise (bilateral, or non-local means)")
        .description(&format!("strength 0..100 (default 30). method bilateral (default, fast) or nlmeans (cv2.fastNlMeansDenoisingColored-like: 3x3 luma patches, 7x7 search; better on grain, slower, up to 16 MP). Alpha unchanged. {IMG_OUT}"))
        .params(obj(json!({"img": img_any(), "strength": num("0..100"), "method": {"type": "string", "enum": ["bilateral", "nlmeans"]}}), &["img"]))
        .typed(filter::denoise)
        .register();
}

fn register_thresh_morph(r: &mut Registry) {
    let channel = json!({"type": "string", "enum": ["gray", "r", "g", "b", "a", "h", "s", "v", "l"], "description": "which channel to threshold (default gray)"});
    r.function(AREA, "threshold", "Threshold → mask (fixed, Otsu or triangle)")
        .description(&format!("cv2.threshold on one channel (default gray). type: binary (v > thresh → maxValue) | binary_inv | trunc | tozero | tozero_inv | otsu | triangle (automatic level, binary). otsu:true adds Otsu to binary/binary_inv like THRESH_OTSU. {MASK_OUT} plus `thresh` (the level used)."))
        .params(obj(
            json!({"img": img_any(), "thresh": num("0..255 (ignored for otsu/triangle)"), "maxValue": num("default 255"), "type": {"type": "string", "enum": ["binary", "binary_inv", "trunc", "tozero", "tozero_inv", "otsu", "triangle"]}, "otsu": {"type": "boolean"}, "channel": channel.clone()}),
            &["img"],
        ))
        .typed(thresh::threshold)
        .register();
    r.function(AREA, "adaptiveThreshold", "Local threshold → mask (uneven lighting, documents)")
        .description(&format!("cv2.adaptiveThreshold: v > local mean (or Gaussian-weighted mean) of a blockSize window minus C → maxValue. blockSize odd 3..255 (default 11), C default 2, method mean | gaussian, type binary | binary_inv. {MASK_OUT}"))
        .params(obj(
            json!({"img": img_any(), "blockSize": {"type": "integer", "minimum": 3, "maximum": 255}, "C": num("subtracted from the local mean"), "method": {"type": "string", "enum": ["mean", "gaussian"]}, "type": {"type": "string", "enum": ["binary", "binary_inv"]}, "maxValue": num("default 255"), "channel": channel}),
            &["img"],
        ))
        .typed(thresh::adaptive_threshold)
        .register();
    r.function(AREA, "morphology", "Erode, dilate, open, close, gradient, tophat, blackhat")
        .description(&format!("cv2.morphologyEx on a mask (an img is reduced to gray). op: erode | dilate | open (remove specks) | close (fill small holes) | gradient (outline) | tophat | blackhat. ksize ≤255 for rect, ≤101 for ellipse/cross (odd); shape rect | ellipse | cross; iterations 1..50. For growing/shrinking by a radius prefer cv.growMask / cv.shrinkMask. {MASK_OUT}"))
        .params(obj(
            json!({"mask": h("mask:N"), "op": {"type": "string", "enum": ["erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"]}, "ksize": {"type": "integer", "minimum": 1, "maximum": 255}, "shape": {"type": "string", "enum": ["rect", "ellipse", "cross"]}, "iterations": {"type": "integer", "minimum": 1, "maximum": 50}}),
            &["mask", "op"],
        ))
        .typed(morph::morphology)
        .register();
    let radius = |d: &str| obj(json!({"mask": h("mask:N"), "radius": {"type": "number", "minimum": 0, "maximum": 1000, "description": d}}), &["mask", "radius"]);
    r.function(AREA, "growMask", "Expand a mask by a radius in px")
        .description(&format!("Dilation by a disc of `radius` px using an exact Euclidean distance transform (any radius, same cost), antialiased edge. {MASK_OUT}"))
        .params(radius("px"))
        .typed(morph::grow_mask)
        .register();
    r.function(AREA, "shrinkMask", "Contract a mask by a radius in px")
        .description(&format!("Erosion by a disc of `radius` px (exact Euclidean distance), antialiased edge. {MASK_OUT}"))
        .params(radius("px"))
        .typed(morph::shrink_mask)
        .register();
    r.function(AREA, "featherMask", "Soften a mask's edge by a radius in px")
        .description(&format!("Gaussian blur of the mask with sigma = radius/2 (a feather of about `radius` px each side of the edge). {MASK_OUT}"))
        .params(radius("feather radius, px"))
        .typed(morph::feather_mask)
        .register();
}

fn register_shapes(r: &mut Registry) {
    r.function(AREA, "findContours", "Outer contours of mask blobs with area, bbox, centroid, points")
        .description("cv2.findContours(RETR_EXTERNAL-like) on mask > threshold (default 127): one entry per 8-connected blob, largest first. Each: {area (pixel count), bbox [x,y,w,h], centroid [x,y], perimeter (px), points [[x,y],…] (outer border simplified with Douglas-Peucker to ≤ maxPoints, default 64)}. minArea filters specks; maxContours (default 100, ≤1000) caps the list. Returns {count (blobs ≥ minArea), contours, width, height}. Blobs inside holes are listed as their own contours.")
        .params(obj(
            json!({"mask": h("mask:N"), "minArea": int("px"), "maxContours": {"type": "integer", "minimum": 1, "maximum": 1000}, "maxPoints": {"type": "integer", "minimum": 3, "maximum": 256}, "threshold": {"type": "integer", "minimum": 0, "maximum": 254}}),
            &["mask"],
        ))
        .typed(shapes::find_contours)
        .register();
    r.function(AREA, "connectedComponents", "Label blobs; stats plus a mask of the kept ones")
        .description(&format!("cv2.connectedComponentsWithStats on mask > threshold. connectivity 4 | 8 (default). Keeps blobs with area ≥ minArea, the largest maxComponents (default 100). Use it to drop specks (minArea) or keep the main subject (maxComponents: 1). {MASK_OUT} of the kept blobs, plus {{count, components: [{{index, area, bbox, centroid}}]}}."))
        .params(obj(
            json!({"mask": h("mask:N"), "connectivity": {"type": "integer", "enum": [4, 8]}, "minArea": int("px"), "maxComponents": {"type": "integer", "minimum": 1, "maximum": 1000}, "threshold": {"type": "integer", "minimum": 0, "maximum": 254}}),
            &["mask"],
        ))
        .typed(shapes::connected_components)
        .register();
    r.function(AREA, "houghLines", "Straight lines in an edge mask (Hough transform)")
        .description("cv2.HoughLines on mask pixels > 127 (usually cv.canny output). rho px (default 1), theta in DEGREES step (default 1), threshold = min votes (default 30% of the strongest line), maxLines (default 20), minAngle/maxAngle filter on `angle`. Each line: {rho, theta (radians, OpenCV normal angle), angle (line direction in degrees, counterclockwise from horizontal on screen, -90..90), votes, x1,y1,x2,y2 (extent of the supporting edge pixels), support}. For horizon straightening just call cv.dominantAngle.")
        .params(obj(
            json!({"mask": h("mask:N"), "rho": num("px, 0.5..100"), "theta": num("degrees, 0.05..10"), "threshold": int("min votes"), "maxLines": {"type": "integer", "minimum": 1, "maximum": 200}, "minAngle": num("degrees"), "maxAngle": num("degrees")}),
            &["mask"],
        ))
        .typed(shapes::hough_lines)
        .register();
    r.function(AREA, "dominantAngle", "Tilt of the dominant horizon/vertical line, for straightening")
        .description("Finds the strongest near-horizontal line (mode horizontal, default), near-vertical line (vertical) or either (both) within ±maxTilt degrees (default 30) and measures its tilt to ~0.1°: edges on a ≤1024 px copy, Hough at 0.1°, then a least-squares fit. Returns {angle (tilt, counterclockwise positive), correction (= -angle: pass it to cv.rotate {angle: correction, crop: true} or to the document's rotate), lineAngle, kind, confidence (0..1, fraction of the image the line spans), candidates}. Check confidence before rotating; < 0.2 is a weak line.")
        .params(obj(json!({"img": img_any(), "maxTilt": num("degrees, 0.5..45"), "mode": {"type": "string", "enum": ["horizontal", "vertical", "both"]}}), &["img"]))
        .typed(shapes::dominant_angle)
        .register();
}

fn register_masks(r: &mut Registry) {
    r.function(AREA, "maskFromColor", "Mask of pixels close to a colour")
        .description(&format!("Select by colour. color: \"#rrggbb\" or [r,g,b]. space: lab (default; tolerance = ΔE76, ~2 just noticeable, default 15), rgb (Euclidean 0..441, default 40), hsv (hue only, tolerance in degrees, default 15; gray pixels excluded). softness ramps coverage to 0 over that many extra units. {MASK_OUT}"))
        .params(obj(
            json!({"img": h("img:N"), "color": {"description": "\"#rrggbb\" or [r, g, b]", "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}]}, "tolerance": num("distance in the space's units"), "softness": num("extra distance over which coverage fades"), "space": {"type": "string", "enum": ["lab", "rgb", "hsv"]}}),
            &["img", "color"],
        ))
        .typed(mask::mask_from_color)
        .register();
    r.function(AREA, "invertMask", "Invert a mask (255 - v)")
        .description(&format!("cv2.bitwise_not for masks: subject ↔ background. {MASK_OUT}"))
        .params(obj(json!({"mask": h("mask:N")}), &["mask"]))
        .typed(mask::invert_mask)
        .register();
    r.function(AREA, "combineMasks", "Union, intersect, subtract, xor, multiply two masks")
        .description(&format!("Soft-mask algebra (same size): union = max, intersect = min, subtract = a without b = min(a, 255-b), xor, multiply = a·b/255. On binary masks these are the usual set operations. {MASK_OUT}"))
        .params(obj(json!({"a": h("mask:N"), "b": h("mask:N"), "op": {"type": "string", "enum": ["union", "intersect", "subtract", "xor", "multiply"]}}), &["a", "b", "op"]))
        .typed(mask::combine_masks)
        .register();
    r.function(AREA, "maskBounds", "Bounding box, area and coverage of a mask")
        .description("Pixels > threshold (default 0). Returns {empty, x, y, w, h, area, coverage, width, height}; pass x,y,w,h to cv.crop to crop to the subject (add a margin).")
        .params(obj(json!({"mask": h("mask:N"), "threshold": {"type": "integer", "minimum": 0, "maximum": 254}}), &["mask"]))
        .typed(mask::mask_bounds)
        .register();
    r.function(AREA, "applyMask", "Use a mask as the image's alpha (cut out)")
        .description(&format!("alpha = alpha · mask / 255 (invert:true uses 255 - mask). Same size required. {IMG_OUT}"))
        .params(obj(json!({"img": h("img:N"), "mask": h("mask:N"), "invert": {"type": "boolean"}}), &["img", "mask"]))
        .typed(mask::apply_mask)
        .register();
    r.function(AREA, "composite", "Paste fg over bg at x, y (optional mask and opacity)")
        .description(&format!("Alpha 'over' compositing of fg onto a copy of bg, fg's top-left at (x, y) (may be negative or partly outside). mask (same size as fg) and opacity 0..1 scale fg's alpha. Output has bg's size. {IMG_OUT}"))
        .params(obj(json!({"bg": h("img:N"), "fg": h("img:N"), "mask": h("mask:N, fg-sized (optional)"), "x": int("px"), "y": int("px"), "opacity": num("0..1")}), &["bg", "fg"]))
        .typed(mask::composite)
        .register();
}

fn register_extra(r: &mut Registry) {
    r.function(AREA, "matchTemplate", "Find where a template best matches in an image")
        .description("cv2.matchTemplate + minMaxLoc on gray: method ccoeff_normed (default, robust to brightness), ccorr_normed, sqdiff_normed (lower = better). Large searches run coarse-to-fine on a pyramid. Returns {x, y (top-left), width, height, center, score (1 = perfect for ccoeff/ccorr), method, pyramidFactor}.")
        .params(obj(json!({"img": img_any(), "template": h("img:N or mask:N, smaller than img"), "method": {"type": "string", "enum": ["ccoeff_normed", "ccorr_normed", "sqdiff_normed"]}}), &["img", "template"]))
        .typed(extra::match_template)
        .register();
    r.function(AREA, "inpaint", "Fill a masked region from its surroundings (object/blemish removal)")
        .description(&format!("Fills mask ≥ 128. method patchmatch (default; content-aware patch synthesis, PatchMatch + multi-scale EM, copies texture) or membrane (smooth harmonic fill, like cv2.inpaint TELEA/NS; best for thin scratches and small spots). radius = patch radius 1..8 (default 3). Works on the hole's bounding box plus context (≤16 MP). Grow the mask a few px first (cv.growMask) so the object's fringe goes too. {IMG_OUT} plus method, region."))
        .params(obj(json!({"img": h("img:N"), "mask": h("mask:N, 255 = fill"), "radius": {"type": "integer", "minimum": 1, "maximum": 8}, "method": {"type": "string", "enum": ["patchmatch", "membrane"]}}), &["img", "mask"]))
        .typed(extra::inpaint)
        .register();
}

#[cfg(test)]
mod tests;

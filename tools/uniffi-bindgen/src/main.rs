//! `cargo run -p uniffi-bindgen -- generate --library <libimagelab_ffi.so> --language kotlin --out-dir …`
//! Run by the Android build (android/app/build.gradle.kts, task `uniffiBindgen`).
fn main() {
    uniffi::uniffi_bindgen_main()
}

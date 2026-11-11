#!/bin/sh
# Fetch Material Symbols Outlined glyphs as Android vector drawables.
#
#   android/scripts/material-symbols.sh home star mic ...
#   android/scripts/material-symbols.sh --fill home star
#
# Android cannot load the Material Symbols .woff2 as a font resource, and a ligature
# font would make every icon a `Text`; a vector drawable is what
# `Icon(painterResource(...))` takes. Google publishes each glyph as an Android
# vector in google/material-design-icons (Apache License 2.0), which is where this
# reads from. Same script as ARGO's argo-android, pointed at this app.
#
# Each glyph lands at app/src/main/res/drawable/ic_sym_<name>.xml (wght 400, GRAD 0,
# opsz 24, unfilled) and, with --fill, at ic_sym_<name>_fill.xml. The
# `android:tint` attribute is dropped: `Icon` tints anyway. Idempotent.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
out="$here/../app/src/main/res/drawable"
base="https://raw.githubusercontent.com/google/material-design-icons/master/symbols/android"

fill=0
if [ "${1:-}" = "--fill" ]; then
  fill=1
  shift
fi
[ "$#" -gt 0 ] || { echo "usage: $0 [--fill] <glyph>..." >&2; exit 2; }

mkdir -p "$out"
for name in "$@"; do
  if [ "$fill" = 1 ]; then
    src="$base/$name/materialsymbolsoutlined/${name}_fill1_24px.xml"
    dest="$out/ic_sym_${name}_fill.xml"
    what="FILL 1"
  else
    src="$base/$name/materialsymbolsoutlined/${name}_24px.xml"
    dest="$out/ic_sym_${name}.xml"
    what="FILL 0"
  fi
  body=$(curl -fsSL "$src") || { echo "no such glyph: $name ($src)" >&2; exit 1; }
  {
    printf '<?xml version="1.0" encoding="utf-8"?>\n'
    printf '<!-- Material Symbols Outlined "%s" (wght 400, GRAD 0, opsz 24, %s), Apache-2.0.\n' "$name" "$what"
    printf '     Fetched by android/scripts/material-symbols.sh; do not edit by hand. -->\n'
    # The attribute only: on Google's files it is the vector tag's last
    # attribute, and the line that carries it also closes the tag.
    printf '%s\n' "$body" | sed -e 's/ *android:tint="[^"]*"//'
  } > "$dest"
  echo "wrote $(basename "$dest")"
done

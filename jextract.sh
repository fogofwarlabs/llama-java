#!/usr/bin/env bash
# Regenerates the classes of org.fogofwarlabs.llama.bindings with jextract from the headers of the llama.cpp build in gradle.properties.
#
#   ./jextract.sh
#
# The script runs on Windows x86_64 (Git Bash): there size_t is unsigned long long, which jextract maps to a layout of 64 bits on every
# platform; on Linux and macOS it maps size_t to C long, whose size differs on Windows. It downloads the pinned jextract build and the
# headers to build/jextract/, generates the functions, structs, typedefs and constants listed in jextract/includes.txt, removes the C long
# layout that the filtered API does not use (its type differs between the platforms) and replaces the sources under src/main/java.
set -euo pipefail

JEXTRACT_URL=https://download.java.net/java/early_access/jextract/25/2/openjdk-25-jextract+2-4_windows-x64_bin.tar.gz
JEXTRACT_SHA256=b03533eb6b249a154752b7c7bf93cdb8c147cf2f9699422e615e84b7fb652872
HEADERS=(include/llama.h ggml/include/ggml.h ggml/include/ggml-alloc.h ggml/include/ggml-backend.h ggml/include/ggml-cpu.h
         ggml/include/ggml-opt.h ggml/include/gguf.h)
PACKAGE=org.fogofwarlabs.llama.bindings

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) ;;
    *) echo "Run this script on Windows x86_64 (Git Bash)." >&2
       exit 1 ;;
esac

cd "$(dirname "$0")"
root=$PWD
build=$(sed -n 's/^llamaBuild=//p' gradle.properties | tr -d '\r')
if [ -z "$build" ]; then
    echo "gradle.properties has no llamaBuild line." >&2
    exit 1
fi
work=$root/build/jextract
mkdir -p "$work"

# jextract, checked against its published SHA-256 digest.
if [ ! -x "$work/jextract-25/bin/jextract" ]; then
    curl -fsSL -o "$work/jextract.tar.gz" "$JEXTRACT_URL"
    echo "$JEXTRACT_SHA256  $work/jextract.tar.gz" | sha256sum -c --quiet -
    tar -xzf "$work/jextract.tar.gz" -C "$work"
    rm "$work/jextract.tar.gz"
fi

# The headers of the release tag.
headers=$work/headers-$build
rm -rf "$headers"
for header in "${HEADERS[@]}"; do
    mkdir -p "$headers/$(dirname "$header")"
    curl -fsSL -o "$headers/$header" "https://raw.githubusercontent.com/ggml-org/llama.cpp/$build/$header"
done

output=$work/output
rm -rf "$output"
mkdir -p "$output"
cmd //c "$(cygpath -w "$work/jextract-25/bin/jextract.bat")" \
    -I "$(cygpath -w "$headers/ggml/include")" -I "$(cygpath -w "$headers/include")" \
    --target-package "$PACKAGE" --header-class-name llama_h --output "$(cygpath -w "$output")" \
    @"$(cygpath -w "$root/jextract/includes.txt")" "$(cygpath -w "$headers/include/llama.h")"

generated=$output/${PACKAGE//./\/}
shared=$generated/llama_h\$shared.java
sed -i '/ C_LONG = /d' "$shared"
if grep -rqE '\bC_LONG\b' "$generated"; then
    echo "The generated classes use C long, whose size differs between the platforms." >&2
    exit 1
fi

target=$root/src/main/java/${PACKAGE//./\/}
rm -rf "$target"
mkdir -p "$target"
for file in "$generated"/*.java; do
    tr -d '\r' < "$file" > "$target/$(basename "$file")"
done
echo "Generated $(ls "$target" | wc -l) classes in $target."

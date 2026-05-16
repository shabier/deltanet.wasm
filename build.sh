#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"

# Check emsdk
if ! command -v emcc &>/dev/null; then
    if [ -n "$EMSDK" ]; then
        source "$EMSDK/emsdk_env.sh" 2>/dev/null
    else
        echo "Error: Emscripten not found. Install emsdk or set EMSDK env var."
        exit 1
    fi
fi

echo "Using Emscripten $(emcc --version | head -1)"

# Check submodule
if [ ! -f "$LLAMA_DIR/CMakeLists.txt" ]; then
    echo "Initializing llama.cpp submodule..."
    git -C "$SCRIPT_DIR" submodule update --init --depth 1
fi

# Apply the performance patch to the llama.cpp submodule. A fresh
# `git clone --recursive` checks out the pinned revision clean, so without
# this step the FMA change (patches/0002) is not present and the build is
# about 2% slower. Safe to run repeatedly: if the reverse-check succeeds the
# patch is already applied and is skipped. Only patches/0002 is applied.
# patches/0001 is the broken upstream PR #19590 (see docs/RESEARCH-LOG.md),
# kept for reference and never applied.
FMA_PATCH="$SCRIPT_DIR/patches/0002-wasm-relaxed-fma-simd-mappings.patch"
if [ -f "$FMA_PATCH" ]; then
    if git -C "$LLAMA_DIR" apply --reverse --check "$FMA_PATCH" 2>/dev/null; then
        echo "FMA patch already applied."
    elif git -C "$LLAMA_DIR" apply --check "$FMA_PATCH" 2>/dev/null; then
        echo "Applying FMA patch (patches/0002)..."
        git -C "$LLAMA_DIR" apply "$FMA_PATCH"
    else
        echo "Warning: patches/0002 does not apply cleanly to this llama.cpp" \
             "revision. Building without the FMA change (about 2% slower)."
    fi
fi

# Build llama.cpp WASM libraries. The shipped flag is `-mrelaxed-simd`. It
# measures equal (within noise) to strict simd128 on x86 and Node, and about
# 14% faster decode on Apple Silicon and Chrome (relaxed_madd becomes one
# NEON fused multiply-add). A strict build was prototyped for iOS but iOS
# cannot run a 500 MB model anyway (memory limit), so there is no strict
# variant. See docs/PERFORMANCE.md.
LLAMA_BUILD="$LLAMA_DIR/build-wasm"
if [ ! -f "$LLAMA_BUILD/src/libllama.a" ] || [ "$1" = "--rebuild" ]; then
    echo "Building llama.cpp WASM libraries..."
    rm -rf "$LLAMA_BUILD"
    mkdir -p "$LLAMA_BUILD"
    cd "$LLAMA_BUILD"
    # EMSCRIPTEN_SYSTEM_PROCESSOR=wasm: Emscripten.cmake defaults
    # CMAKE_SYSTEM_PROCESSOR to "x86", so ggml's arch detection skips
    # arch/wasm/quants.c and the Q4_0 dot compiles to generic scalar. Forcing
    # "wasm" makes the strict-simd128 kernel compile. This is the largest
    # single speedup.
    emcmake cmake .. \
        -DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-pthread -mrelaxed-simd" \
        -DCMAKE_CXX_FLAGS="-pthread -mrelaxed-simd" \
        -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
        -DGGML_OPENMP=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_HTML=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_WASM_MEM64=OFF
    cmake --build . -j$(nproc)
    echo "llama.cpp libraries built."
fi

BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$SCRIPT_DIR"

echo "Compiling deltanet-wasm bindings..."
em++ src/deltanet-wasm-bindings.cpp \
    -I "$LLAMA_DIR/include" -I "$LLAMA_DIR/ggml/include" \
    -L "$LLAMA_BUILD/src" -lllama \
    -L "$LLAMA_BUILD/common" -lcommon \
    -L "$LLAMA_BUILD/ggml/src" -lggml -lggml-cpu -lggml-base \
    -std=c++17 -O3 -mrelaxed-simd -pthread \
    -sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency \
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sINITIAL_MEMORY=256MB \
    -sFORCE_FILESYSTEM=1 -sEXPORTED_RUNTIME_METHODS='["FS"]' \
    -sENVIRONMENT=web,worker -sMODULARIZE=1 -sEXPORT_NAME=createDeltaNet \
    --bind -o "$BUILD_DIR/deltanet-wasm.js"

echo ""
echo "Build complete:"
ls -lh "$BUILD_DIR/deltanet-wasm.js" "$BUILD_DIR/deltanet-wasm.wasm"

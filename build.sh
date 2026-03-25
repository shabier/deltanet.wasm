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

# Build llama.cpp WASM libraries
LLAMA_BUILD="$LLAMA_DIR/build-wasm"
if [ ! -f "$LLAMA_BUILD/src/libllama.a" ] || [ "$1" = "--rebuild" ]; then
    echo "Building llama.cpp WASM libraries..."
    rm -rf "$LLAMA_BUILD"
    mkdir -p "$LLAMA_BUILD"
    cd "$LLAMA_BUILD"
    emcmake cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-pthread" \
        -DCMAKE_CXX_FLAGS="-pthread" \
        -DGGML_METAL=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_OPENMP=OFF \
        -DGGML_BLAS=OFF \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_HTML=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_WASM_MEM64=OFF
    cmake --build . -j$(nproc)
    echo "llama.cpp libraries built."
fi

# Build WASM bindings
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$SCRIPT_DIR"

echo "Compiling deltanet-wasm bindings..."
em++ src/deltanet-wasm-bindings.cpp \
    -I "$LLAMA_DIR/include" \
    -I "$LLAMA_DIR/ggml/include" \
    -L "$LLAMA_BUILD/src" -lllama \
    -L "$LLAMA_BUILD/common" -lcommon \
    -L "$LLAMA_BUILD/ggml/src" -lggml -lggml-cpu -lggml-base \
    -std=c++17 -O3 \
    -msimd128 \
    -pthread \
    -sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency \
    -sALLOW_MEMORY_GROWTH=1 \
    -sMAXIMUM_MEMORY=4GB \
    -sINITIAL_MEMORY=256MB \
    -sFORCE_FILESYSTEM=1 \
    -sEXPORTED_RUNTIME_METHODS='["FS"]' \
    -sENVIRONMENT=web,worker \
    -sMODULARIZE=1 \
    -sEXPORT_NAME=createDeltaNet \
    --bind \
    -o "$BUILD_DIR/deltanet-wasm.js"

echo ""
echo "Build complete:"
ls -lh "$BUILD_DIR/deltanet-wasm.js" "$BUILD_DIR/deltanet-wasm.wasm"

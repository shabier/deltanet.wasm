# deltanet.wasm

llama.cpp compiled to WebAssembly, with the SSM operations (`ggml_ssm_conv`, `ggml_ssm_scan`) tested and working. Runs Qwen 3.5 and other GGUF models client-side. Use it in a browser, in Node, or anywhere Emscripten targets.

I haven't found another public WASM build that runs a DeltaNet model end-to-end. The existing builds (wllama, llama-cpp-wasm, web-llm) don't currently support it.

## Background

Most ML frameworks depend on BLAS (OpenBLAS, Intel MKL) for matrix math. Those libraries are written in platform-specific assembly and don't compile to WASM. GGML sidesteps this entirely. Every tensor op uses self-contained SIMD intrinsics that Emscripten maps straight to WASM v128 opcodes. No external dependencies.

Qwen 3.5 uses DeltaNet, a hybrid architecture where 75% of layers are linear attention O(n) and 25% are standard quadratic attention. The linear layers use a delta rule state update instead of recomputing attention over the full history. A 0.8B DeltaNet model benchmarks close to a standard 1.7B transformer, at half the download size.

GGML's SSM ops (`ggml_ssm_conv`, `ggml_ssm_scan`) are portable C + SIMD by design. Emscripten remaps the intrinsics to WASM v128 opcodes, so the kernels are CPU-portable rather than architecture-specific. In principle they should work in WASM. This build confirms it.

## What I changed

The upstream llama.cpp `generate()` pattern runs the entire decode loop in C++ and returns a string. That blocks the worker thread for the full generation. I split it into discrete steps so the JavaScript side controls the loop:

1. `decodePrompt(text)`: tokenize and process the input in batches
2. `generateToken()`: sample one token, return it immediately
3. Repeat in a JS loop, posting each token to the main thread

This lets you stream tokens to the UI as they're generated and cancel between steps without killing the worker.

I also tuned the threading (pthread pool sized to `navigator.hardwareConcurrency`), enabled WASM SIMD, and set batch size to 2048 for faster prompt processing.

## Numbers

Qwen 3.5 0.8B, Q4_0 quantization (507 MB), Chrome on M4 MacBook Air:

| | |
|---|---|
| Generation | ~8 tok/s |
| Model load | ~560 ms |
| WASM binary | 2.1 MB |
| JS glue | 97 KB |
| Peak memory | ~1 GB |

Model downloads once, then lives in the browser's Cache API.

## Build

```bash
git clone --recursive https://github.com/shabier/deltanet.wasm.git
cd deltanet.wasm
./build.sh
```

Needs [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html), CMake 3.14+, and Git. Outputs `build/deltanet-wasm.js` and `build/deltanet-wasm.wasm`.

Submodule pinned to `llama.cpp@a970515`. Future upstream changes may require rebases.

## Usage

```js
// In a Web Worker
importScripts('deltanet-wasm.js');

const Module = await createDeltaNet({
  locateFile: (path) => '/wasm/' + path,
});

Module.FS.writeFile('/model.gguf', modelBytes);
Module.loadModel('/model.gguf', 4096);
Module.FS.unlink('/model.gguf');

Module.resetContext();
Module.decodePrompt('<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n');

for (let i = 0; i < 256; i++) {
  const token = Module.generateToken();
  if (!token) break;
  postMessage({ type: 'token', token });
}
```

## API

| Function | Description |
|----------|-------------|
| `loadModel(path, n_ctx)` | Load a GGUF model from the virtual filesystem |
| `tokenize(text)` | Count tokens without processing |
| `decodePrompt(text)` | Tokenize and run prefill in batches |
| `generateToken()` | Sample and decode one token, empty string on EOS |
| `resetContext()` | Clear KV cache |
| `freeModel()` | Release everything |

## Browser requirements

SharedArrayBuffer (needs COOP/COEP headers), WASM SIMD (Chrome 91+, Firefox 89+, Safari 16.4+), and about 1 GB of free RAM for the 0.8B model.

## Models

Anything llama.cpp supports in GGUF format works. Tested with Qwen 3.5 0.8B (DeltaNet). Standard transformer models (Qwen 3, Llama, Phi, Gemma) also work. They just don't use the SSM ops.

## Architecture

```
Web Worker
  → Emscripten module (createDeltaNet)
    → llama.cpp (model loading, tokenization, sampling)
      → GGML (tensor ops, compute graph)
        → ggml-cpu (matmul, attention, SSM ops)
          → WASM SIMD + pthread workers
```

The DeltaNet scheduler activates automatically when a DeltaNet model is loaded. No configuration needed.

## Acknowledgments

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp). My contribution is the WASM build, the per-token API, and confirming the SSM ops work in a browser.

## License

MIT

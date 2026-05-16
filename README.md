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

I also changed the build for speed. Emscripten's CMake reports the target as `x86`, so ggml's architecture detection compiled the quantized matmul kernels as plain scalar code; the SIMD build was not running SIMD for the Q4_0 dot product. The changes:

- force the wasm architecture (fixes the scalar problem)
- add relaxed-SIMD
- a fused multiply-add patch (`patches/0002`, applied to the submodule by `build.sh`)
- thread count from physical cores, plus one persistent threadpool
- batch size 2048 for faster prefill

Together this took decode from about 32 to about 54 tokens/sec (Node, Ryzen 5 7600). A fresh clone reproduces it.

The full method, the measured per-token decode cost, and deployment notes are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md). The research log, including the approaches that did not work and two conclusions that were later found wrong and corrected, is in [docs/RESEARCH-LOG.md](docs/RESEARCH-LOG.md).

## Numbers

Qwen 3.5 0.8B, Q4_0 (507 MB), steady-state decode (same prompt and method for both rows, see [docs/PERFORMANCE.md](docs/PERFORMANCE.md)):

| Machine | Engine | Decode | Model load |
|---|---|---|---|
| M4 MacBook Air (10-core) | Chrome 147 | ~173 tok/s | ~470 ms |
| Ryzen 5 7600 | Node 22 | ~54 tok/s | ~500 ms |

WASM binary 2.2 MB, JS glue 98 KB, peak memory ~1 GB. The model downloads once, then lives in the browser's Cache API. The generated text is identical on both machines (x86-64 Node and ARM64 Chrome): the `sha256` of the output matches.

The pre-rework build, from the initial commit, was benchmarked on the same M4 and Chrome with the same method: about 104 tok/s steady-state. So the rework is about 1.7x on the M4 (104 to 173) and about 1.6x on the Ryzen (34 to 54 steady-state), consistent across both machines. An earlier version of this file said the pre-rework speed was about 8 tok/s and the fix was about 20x; that 8 was a loose measurement, is wrong, and is dropped.

These are steady-state numbers. V8 runs WASM on a slow baseline compiler for the first 16 to 32 tokens and then switches to the optimizing compiler in under a second, so a short cold measurement reads lower than the real rate. The method, the cold versus steady detail, the cross-hardware table (which includes a thread-count heuristic that misfires on Apple Silicon), and the measured per-token cost are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Build

```bash
git clone --recursive https://github.com/shabier/deltanet.wasm.git
cd deltanet.wasm
./build.sh
```

Needs [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html), CMake 3.14+, and Git. Outputs `build/deltanet-wasm.js` and `build/deltanet-wasm.wasm`. `--rebuild` forces a lib rebuild.

Submodule pinned to `llama.cpp@a970515`. `build.sh` applies `patches/0002` (the FMA change) to the submodule on every run. The apply step is safe to run repeatedly and only that patch is applied. `patches/0001` is the broken upstream PR #19590, kept for reference and never applied. An upstream bump may require the patch to be regenerated.

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

Requirements:

- SharedArrayBuffer: COOP/COEP headers and a secure context (`https://` or `http://localhost`; a plain HTTP LAN or Tailscale IP will not work)
- WASM SIMD and threads
- about 1 GB free RAM
- a `-mrelaxed-simd` engine: Chrome 114+, Firefox 120+, Node 22+ (every current desktop engine)

Tested on desktop Chrome, Firefox, and Node. Desktop Safari is untested. Mobile is not supported: a roughly 500 MB model is over the iOS Safari per-tab memory limit and the tab crashes regardless of build. That is a memory limit, not a SIMD problem; the strict-build comparison is in [docs/PERFORMANCE.md](docs/PERFORMANCE.md), section "iOS / relaxed-SIMD".

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

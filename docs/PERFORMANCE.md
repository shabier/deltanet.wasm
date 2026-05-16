# Performance

Reference for the build settings that affect speed, the reproducible
benchmark, and the measured per-token decode cost. The story of how this was
found, including the approaches that did not work and two conclusions that
were later corrected, is in [RESEARCH-LOG.md](RESEARCH-LOG.md).

## What ships, and why

`build.sh` applies all of these automatically (the patch through the
`patches/0002` step, the flags inline). The model output stays correct, see
"Quality" below.

| Change | Where | Effect |
|---|---|---|
| `-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm` | `build.sh` cmake | Emscripten sets `CMAKE_SYSTEM_PROCESSOR` to `x86` by default, so ggml's architecture detection did not compile `arch/wasm/quants.c` and the Q4_0 dot product used the generic scalar code. Forcing `wasm` makes the SIMD kernel compile. This is the largest single effect. |
| `-mrelaxed-simd` (libs + bindings) | `build.sh` | Single shipped binary. About 0% versus strict `simd128` on x86/V8, but about 14% faster decode on Apple Silicon with Chrome (`relaxed_madd` becomes one NEON fused multiply-add, measured in the M4 A/B). Needs a relaxed-SIMD engine (Chrome 114+, Firefox 120+, Node 22+, which is every desktop target). Not iOS, which cannot run a 500 MB model anyway (memory). See "iOS / relaxed-SIMD". |
| `relaxed_madd` FMA | `patches/0002`, applied to the submodule | The WASM `GGML_F32x4_FMA` macro was a separate multiply then add; every other ggml backend fuses it. Now one `wasm_f32x4_relaxed_madd` under `__wasm_relaxed_simd__`. This is what makes decode faster on Apple Silicon. |
| Thread heuristic and persistent threadpool | `src/deltanet-wasm-bindings.cpp` | `pick_threads()` plus one persistent `ggml_threadpool_t` instead of one created per `llama_decode()` call. Note: the heuristic misfires on Apple Silicon with more than 8 cores, see the cross-hardware table. |

Measured pre-rework versus shipped, same method (steady-state) on each
machine: Ryzen and Node about 34 to about 54 tok/s, M4 and Chrome about 104
to about 173 tok/s. That is about 1.6x on x86 and about 1.7x on Apple
Silicon, output correct and deterministic. (The original 32.25 figure was
the pre-rework number under the old cold-blended measurement; 34 is the same
build measured the proper way.) The architecture-detection fix is the largest
effect on every machine. The relaxed-SIMD flag and FMA are noise on x86 but
about 14% on Apple Silicon (NEON FMA, measured, see "iOS / relaxed-SIMD").

## Reproducible benchmark

Everything runs inside the official `emscripten/emsdk` Docker image. Nothing
is installed on the host except Docker and the model file.

```bash
# 1. model
mkdir -p bench-cache
curl -L -o bench-cache/Qwen3.5-0.8B-Q4_0.gguf \
  'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_0.gguf?download=true'

# 2. build. build.sh ships -sENVIRONMENT=web,worker; the second command
#    relinks with `node` added so require() works from the harness.
docker run --rm -u $(id -u):$(id -g) -v "$PWD":/repo -w /repo -e HOME=/tmp \
  emscripten/emsdk:latest bash -c './build.sh'
docker run --rm -u $(id -u):$(id -g) -v "$PWD":/repo -w /repo -e HOME=/tmp \
  emscripten/emsdk:latest bash -c 'em++ src/deltanet-wasm-bindings.cpp \
    -I llama.cpp/include -I llama.cpp/ggml/include \
    -L llama.cpp/build-wasm/src -lllama -L llama.cpp/build-wasm/common -lcommon \
    -L llama.cpp/build-wasm/ggml/src -lggml -lggml-cpu -lggml-base \
    -std=c++17 -O3 -mrelaxed-simd -pthread \
    -sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency \
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sINITIAL_MEMORY=256MB \
    -sFORCE_FILESYSTEM=1 -sEXPORTED_RUNTIME_METHODS=["FS"] \
    -sENVIRONMENT=web,worker,node -sMODULARIZE=1 -sEXPORT_NAME=createDeltaNet \
    --bind -o build/deltanet-wasm.js'

# 3. run. See "Measure steady state, not a cold window" for why --no-liftoff.
docker run --rm -u $(id -u):$(id -g) -v "$PWD":/repo \
  -v "$PWD/bench-cache":/cache -w /cache -e HOME=/tmp \
  emscripten/emsdk:latest node --no-liftoff /cache/bench.mjs
```

The harness (`bench-cache/bench.mjs`, gitignored) reports `decode_tok_per_s`,
`prefill_tok_per_s`, `model_load_ms`, peak RSS, and `output_sha256_16`. The
sha is also a quality check: greedy sampling makes the output deterministic
given the same compute order, so a stable sha means the output did not change.

Reference machine: Arch Linux, Ryzen 5 7600 (6 physical, 12 logical), Node 22
in the container, Qwen3.5-0.8B Q4_0.

### In-browser bench (cross-hardware)

The browser numbers below were gathered with a small local harness: a
COOP/COEP static server (SharedArrayBuffer needs cross-origin isolation, so
`python -m http.server` is not enough) and a page that runs the same protocol
as the Node bench. Same fixed prompt, 256 tokens, greedy, threads from the
built-in heuristic, and steady-state decode (first 32 tokens discarded,
because a browser cannot pass `--no-liftoff`, so V8's automatic switch to the
optimizing compiler in under a second is what gets measured, which is the
real product behaviour). That harness is a local testing tool, not part of
the repo (same as `bench-cache/`, gitignored, rebuild as needed). Notes for
the problems we hit, recorded so they are not found again:

> SharedArrayBuffer needs a secure context and COOP/COEP. `http://localhost`
> and `http://127.0.0.1` qualify; a plain HTTP LAN or Tailscale IP does not
> (the headers are sent but the browser still blocks SAB). To bench another
> machine, run the server on that machine and open it via `localhost`, or put
> it behind real HTTPS (Tailscale Serve's `*.ts.net` certificate works). The
> WASM is portable, so artifacts built on any host run in any browser.

| Machine | Engine | Build | Threads | Steady decode | Cold | Prefill | Load |
|---|---|---|---|---|---|---|---|
| Ryzen 5 7600 (6P/12T) | Node 22, `--no-liftoff` | relaxed | 6 | 54.2 | n/a | ~49 | ~0.5 s |
| Ryzen 5 7600 (6P/12T) | Node 22, `--no-liftoff` | strict | 6 | 54.4 | n/a | ~49 | ~0.5 s |
| Ryzen 5 7600 (6P/12T) | Node 22, default | relaxed | 6 | ~54 | ~27 then ramps | ~49 | ~0.5 s |
| M4 MacBook Air (10-core) | Chrome 147 | relaxed | 5 | 173.1 | 51 | 62 | ~0.47 s |
| M4 MacBook Air (10-core) | Chrome 147 | strict | 5 | 152.2 | 50 | 76 | ~0.56 s |
| M4 MacBook Air (10-core) | Chrome 147 | pre-rework baseline | 8 | 103.6 | 38 | 50 | ~0.68 s |

The shipped build is relaxed (single binary). Main numbers: Ryzen and Node
about 54 tok/s, M4 and Chrome about 173 tok/s steady-state. The output is
identical across architectures (relaxed sha `348c6e9289611982`, strict
`a9133467260155da`, each matching its Node run; the pre-rework baseline is
`8f66e60b6a86e729`, also identical across Node and Chrome), so the quality
check holds across x86-64 Node and ARM64 Chrome. The pre-rework build (from
the initial commit: no arch fix, libs compiled scalar, `n_threads=8`, no
persistent threadpool) measured 103.6 tok/s steady-state on the same M4 with
the same method, so the rework is about 1.7x on the M4 (104 to 173) and about
1.6x on the Ryzen (34 to 54 steady-state). An earlier version of this file
said the pre-rework speed was about 8 tok/s; that was a loose measurement and
is wrong (the real pre-rework steady-state is about 104). The arch-detection
fix is most of the difference. The strict rows are why a strict-only build
was rejected, see
"iOS / relaxed-SIMD".

> Thread heuristic misfire on Apple Silicon. `pick_threads()` uses
> `hc > 8 ? hc/2 : hc`, assuming more than 8 logical cores means SMT (2 times
> physical). The M4 reports `hardware_concurrency = 10` with no SMT (10
> physical cores), so it ran on 5 threads with 5 cores idle, and still
> reached 170 tok/s. The assumption the heuristic was built on, that Apple
> Silicon has 8 or fewer logical cores and no SMT, is no longer true for
> M3 and M4 parts. A fix would detect no-SMT (or Apple Silicon) and use the
> full core count instead of halving above 8. Decode is limited by the
> overall budget so the decode gain may be small, but prefill is
> compute-bound and would benefit, as would lower-core machines. It needs an
> A/B with a forced thread count. This is an open item.

## Measure steady state, not a cold window

V8 compiles WASM with the Liftoff baseline compiler first and switches to the
optimizing TurboFan compiler in the background. The switch finishes within
the first 16 to 32 generated tokens (under a second); only those run at the
slow Liftoff rate. A single 128-token window started cold therefore mixes
about 16 slow tokens with about 112 fast ones and reports about 48 tok/s,
while the real steady-state rate is about 54.

- Benchmarking: use `node --no-liftoff` (forces TurboFan from the first
  token), or discard the first 32 to 64 tokens. Otherwise you measure the
  baseline compiler and every comparison is contaminated.
- Node deployment: `--no-liftoff` is worth setting (the CLI flag, or
  `require('v8').setFlagsFromString('--no-liftoff')` before instantiation).
  It does not raise the ceiling, it removes the short cold period at the
  start.
- Browser tab: same V8, it switches automatically in under a second. A real
  session runs at the steady rate after that. Report cold and warm
  separately; do not quote the cold number as the product's speed.

## Measured decode budget

Per-token decode time. Each share except Q4_0 was measured by removing that
component in a throwaway build and reading the change in decode speed (not
estimated). Q4_0 is a probe estimate, see the note after the table. Reference
machine at about 54 tok/s (18.5 ms per token).

| Component | Time | Share | How measured | Notes |
|---|---|---|---|---|
| Q4_0 matmuls (129 tensors, FFN and projections) | ~6.3 ms | ~34% | byte budget / kernel probe (TurboFan) | bandwidth-bound, no source-level change left that helps |
| lm_head Q6_K (1 tensor, full-vocab projection) | ~5.5 ms | ~30% | no-op, decode 54 to 77 | bandwidth-bound, no source-level change left that helps |
| Other (small f32 ops across 24 layers, the ~1377-node graph and its barriers, sampling) | ~4.9 ms | ~26% | residual | spread out, no single expensive part; needs graph-level work |
| Q5_K (18, ssm_out, mostly L3-resident) | ~0.8 ms | ~4% | byte budget | negligible |
| DeltaNet f32 recurrence | ~0.5 ms | ~3% | no-op, decode 54 to ~55.8 | negligible |
| Activation requant (q8_0/q8_K) | ~0.5 ms | ~3% | no-op, decode 54 to ~55.5 | negligible |

The lm_head, DeltaNet, and requant rows are direct in-situ measurements
(remove the component, read the decode delta, under `--no-liftoff`). Q4_0 is
not: it is the GGUF byte budget (about 252 MB of Q4_0 weights per token)
divided by the standalone Q4_0 kernel throughput from the real-kernel probe
under `--no-liftoff` (about 40 GB/s). That is a TurboFan number, not a cold
or Liftoff one, but it is a lower bound on the in-situ cost (the isolated
probe does not include the per-row dispatch and graph overhead around the
129 real matmuls), so the true Q4_0 share is probably a little higher and the
"Other" residual absorbs the difference.

What this means: the two large matmuls (about 64% together) are
bandwidth-bound and have no source-level change left that helps (checked
across about 12 experiments under clean TurboFan, including bit-exact kernel
rewrites). The DeltaNet and activation-requant suspects were measured at
about 3% each, so they are not where the time goes. There is no
quality-preserving quick change left in contained source. Remaining options:

- `patches/0001` (upstream PR #19590, relaxed i8 dot) is incorrect: x86
  lowers `relaxed_dot_i8x16_i7x16` through unsigned `pmaddubsw`, which does
  not compute Q4_0 by Q8_0 correctly. Kept for reference, not applied.
- Quality-lossy: requantize the lm_head from Q6_K to Q4_0 (about 209 MB to
  about 143 MB, plus the faster kernel, so roughly +11%, about 54 to about
  60), only after a check that logits and argmax do not change meaningfully.
- Graph-level: the spread-out 26% is upstream graph-fusion work (fewer than
  1377 nodes per token), not a contained change.

## Quality

Every shipped change is either bit-exact (the architecture and threadpool
work) or the same floating-point class as FMA on x86 and ARM (the
`relaxed_madd` patch is single-rounding versus double-rounding, the same
reassociation every FMA-enabled ggml platform already does). Output is
correct and deterministic across runs; the generation sha is stable. No
quality regression was accepted at any step. Candidate changes that made the
output incorrect (for example the relaxed i8 dot) were rejected, not shipped.

## iOS / relaxed-SIMD

`build.sh` passes `-mrelaxed-simd`. That lets LLVM emit relaxed-SIMD opcodes
anywhere in the module, and an engine that does not implement the
relaxed-SIMD proposal rejects the whole module when it parses it. There is no
per-instruction fallback:

```
CompileError: WebAssembly.Module doesn't parse at byte 57:
relaxed simd instructions not supported, in function at index 253
```

Seen on iOS Safari (WebKit added relaxed-SIMD around Safari and iOS 18; the
older installed base fails). Desktop Chrome 114+, Firefox 120+, Node 22+ are
fine.

A strict `-msimd128` build was prototyped to fix this. The `patches/0002`
FMA macro is guarded by `__wasm_relaxed_simd__`, so a strict build emits no
relaxed opcodes and uses the strict multiply-add fallback with no source
change. It was measured both ways before any decision:

| Machine and engine | relaxed decode | strict decode | strict vs relaxed |
|---|---|---|---|
| Ryzen 5 7600, Node (x86, `--no-liftoff`, 5 trials) | 54.15 | 54.42 | about 0% (noise) |
| M4 MacBook Air, Chrome 147 (ARM) | 173.1 | 152.2 | relaxed about 14% faster |

The value of relaxed-SIMD depends on the architecture, it is not a constant.
On x86 with V8 the two are equal (the large effect is the strict `simd128`
architecture-detection kernel; relaxed adds nothing). On Apple Silicon,
`wasm_f32x4_relaxed_madd` becomes one NEON fused multiply-add where strict is
multiply then add (two operations), which is about 14% faster decode on the
M4. (M4 prefill is the other way around, strict 76 versus relaxed 62 t/s, but
prefill is about 0.2 s on a 16-token prompt, and decode over hundreds of
tokens is what matters for generation.) Both builds are correct and produce
identical output per architecture (relaxed sha `348c6e9289611982`, strict
`a9133467260155da`, each matching its Node run).

The earlier conclusion, "relaxed is worth about 0%, ship strict only", was
based on x86 only. That is the same kind of mistake corrected elsewhere in
this document. The ARM A/B disproved it.

### Decision: relaxed-only, mobile not supported

iOS Safari is not a target, independent of SIMD. A roughly 500 MB model is
over WebKit's per-tab memory limit (the tab is killed at about 1 GB) and the
tab crashes after the module loads. This was confirmed on a device and
matches earlier projects that did not support mobile for this model size. So
the only thing strict was good for (loading on iOS) does not help, and strict
is slower on Apple Silicon for decode and equal on x86. There is no platform
where strict is the better choice. The strict variant, the `--both` and
`--strict` build modes, and the feature-detect loader were removed:

- Ship the relaxed build only (`-mrelaxed-simd`, single binary, the
  historical default, unchanged). Fastest on ARM, equal on x86, simplest
  repo.
- `patches/0002` is kept. The relaxed FMA is what makes decode faster on
  Apple Silicon (it is now measured-valuable on ARM, not pointless).
  `build.sh` applies it as before.
- Mobile and iOS are documented as not supported (a memory limit, not a SIMD
  problem). Desktop targets (Chrome 114+, Firefox 120+, Node 22+) all have
  relaxed-SIMD, so the narrower browser floor does not matter for anyone who
  can run a 0.8B model in a tab.

Lesson: the value of a SIMD flag does not carry across architectures. Measure
on the target. Here the ARM A/B reversed the x86 conclusion.

## Caveats

- The Ryzen numbers are Node in a container on a 6-physical, 12-logical CPU.
  Apple Silicon with more than 8 cores hits the thread heuristic misfire
  above; it still gets the architecture-detection win, and the FMA win when
  relaxed-SIMD is used.
- Only V8 has been profiled. The shape of the budget (bandwidth-bound
  matmuls) does not depend on the engine, but the absolute numbers on
  SpiderMonkey and JavaScriptCore are not verified.

# deltanet.wasm research log

This is the work log: what was tried, what did not work, and two conclusions
that were later found wrong and corrected. For the reference (what ships, the
reproducible benchmark, the measured budget) see
[PERFORMANCE.md](PERFORMANCE.md).

## Starting point

The request was to benchmark the current build and make it faster, because
the current speed was too slow. The project is llama.cpp compiled to
WebAssembly with Qwen3.5-Next (Gated Delta Net) support. Constraints set by
the user over the session: nothing is installed on the host (every build and
bench runs in the official `emscripten/emsdk` Docker image with only the repo
and model mounted), no git commands without asking, no risky commands while
the user was away, and no quality regression (any speedup had to keep the
model output correct).

Model under test: Qwen3.5-0.8B Q4_0. 24 blocks, `full_attention_interval=4`
so 18 Gated-DeltaNet layers and 6 full-attention layers, embedding 1024, FFN
3584, SSM state size 128, 16 delta-net heads. Hardware: Arch Linux, Ryzen 5
7600 (6 physical, 12 logical), Node 22 in the container.

Decode started at 32.25 tok/s and ended at about 54 tok/s steady-state, output
correct and deterministic. The headline number changed twice as the
measurement itself was corrected (Phases 9 to 13). The durable result that
applies on every machine is the compile-time changes; the rest of this log is
how that became clear.

---

## Phase 1: containerized bench harness

Goal: a reproducible decode and prefill number with nothing installed on the
host.

Approach: a Node harness (`bench-cache/bench.mjs`) that loads the GGUF into
MEMFS, runs prefill on a fixed prompt, decodes `N_GEN` tokens, and prints
JSON: `module_init_ms`, `model_load_ms`, `prefill_tok_per_s`,
`decode_tok_per_s`, `peak_rss_mb`, and `output_sha256_16` (the first 16 hex
of a sha256 over the generated text). Greedy sampling makes that hash
deterministic given the same compute order, so it also detects quality
changes across every experiment.

Problem found: `build.sh` ships `-sENVIRONMENT=web,worker`, which removes
Node's pthread shim and breaks `require()` from Node. The bench needs a
`web,worker,node` relink, which is one linker flag different from the shipped
browser artifact and otherwise identical.

Result: a one-command Docker bench. The sha was used on every change after
this: each one was judged on tok/s and on whether the output stayed correct.

---

## Phase 2: thread oversubscription

Goal: stop running more matmul threads than physical cores.

Findings: `loadModel()` hardcoded `n_threads = 8`. On a 6-physical-core Ryzen
that reports 12 logical, 8 matmul threads run on SMT siblings. Swept the
count:

| n_threads | Decode (tok/s) | Note |
|-----------|----------------|------|
| 4 | 26.05 | under-used |
| 6 | 32.42 | best, equals physical core count |
| 8 | 30.59 | the old default, SMT oversubscription |
| 10 | 29.07 | contention |
| 12 | 28.53 | thrashing (prefill drops to 11.6) |

Fix: a `pick_threads()` heuristic. If `hardware_concurrency() > 8`, use
`hc/2` (assume an SMT desktop), otherwise use `hc`. That keeps the original 8
on Apple Silicon and mobile (8 or fewer logical, no SMT, at the time) so it
does not regress a platform that could not be tested here. Added
`setThreads()` and `setPoll()` JS overrides; kept the `loadModel(path,
n_ctx)` signature so the README example still works.

Result: output bit-identical. Combined with Phase 3, 32.25 to 35.81 tok/s.

---

## Phase 3: per-token threadpool churn

Goal: stop creating and destroying a threadpool every token.

Findings: with no threadpool attached to the `llama_context`, ggml builds a
temporary one inside every `llama_decode()` call, which is `pthread_create`
times (N-1) then `pthread_join` times (N-1) per generated token. In WASM over
Web Workers that is real overhead, 128 times for a 128-token decode.

Fix: create one persistent `ggml_threadpool_t` in `loadModel()`, attach it
with `llama_attach_threadpool()`, free it in `freeModel()`.

Result: included in the 35.81 tok/s above. Bit-identical.

---

## Phase 4: the Emscripten arch-detection problem (the large one)

Goal: find out why a SIMD build was this slow.

Investigation: ggml's CMake picks an architecture-specific `quants.c` (NEON
for arm64, AVX2 for x86, `wasm_i32x4_dot_i16x8` for wasm) by matching
`CMAKE_SYSTEM_PROCESSOR`. Emscripten sets that to `x86`. So ggml's
`MATCHES "wasm"` check never matched and the build used the generic scalar
Q4_0 dot product. The WASM SIMD kernel was never compiled. The SIMD build had
been running scalar code the whole time.

Did not work: `-DCMAKE_SYSTEM_PROCESSOR=wasm`. The Emscripten toolchain file
resets it after it is set.

Fix: `-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm`, an Emscripten.cmake override
applied after the toolchain file runs. ggml's wasm branch matches and
`arch/wasm/quants.c` compiles in.

Result: 35.81 to 46.12 tok/s, about +29% from one CMake flag. The output sha
changed (different floating-point accumulation order than scalar) but the
text stayed correct and on-topic, which is the expected and accepted
SIMD-versus-scalar reassociation.

---

## Phase 5: relaxed-SIMD compile mode

Goal: use the relaxed-SIMD code paths the kernels already contain.

Findings: `arch/wasm/quants.c` has paths guarded by `__wasm_relaxed_simd__`
(F16 multiply-add and similar) that the strict-SIMD fallback was running
instead.

Fix: `-msimd128` to `-mrelaxed-simd` on the lib build and the bindings link.

Result: 46.12 to 47.56 tok/s, about +3%. Deterministic across trials. (Note:
Phase 11 later showed this +3% was a measurement artifact, see there.)

---

## Phase 6: FMA contraction in `simd-mappings.h`

Goal: a real fused multiply-add on the WASM path.

Findings: `GGML_F32x4_FMA(a,b,c)` for WASM was
`wasm_f32x4_add(wasm_f32x4_mul(b,c), a)`, a separate multiply then add. Every
other ggml backend fuses it (AVX2 `_mm256_fmadd_ps`, NEON `vfmaq_f32`). This
macro is the inner operation of `ggml_vec_dot_f32` and `ggml_vec_mad_f32`,
which the Gated-DeltaNet kernel (18 of 24 layers) and the F16 attention dot
products use.

Fix: under `__wasm_relaxed_simd__`, `GGML_F32x4_FMA(a,b,c)` becomes
`wasm_f32x4_relaxed_madd(b,c,a)`, one instruction, the same accuracy class as
FMA on x86 and ARM. The strict multiply-add is kept as the fallback. Shipped
as `patches/0002-wasm-relaxed-fma-simd-mappings.patch`, applied to the
submodule by `build.sh`.

Result: 47.56 to 48.47 tok/s, about +1.9%, tight variance. sha
`348c6e9289611982`, correct and deterministic. This became the shipped
configuration. About +50% over the original. (Phase 11 later showed +1.9% was
also a measurement artifact on x86; Phase, the M4 A/B, showed it is real on
Apple Silicon.)

---

## Phase 7: profiling, where decode time goes

Goal: before doing more, find the actual bottleneck.

Findings: read the model hyperparameters from the GGUF. The Gated-DeltaNet
gate is scalar per head (the kernel's `else` branch: one `expf` then a SIMD
scale), so `expf` is not significant (about 288 scalar `expf` per token).
The flop budget is dominated by the Q4_0 weight matmuls (FFN and
projections), about 10 times the delta-net recurrence. During autoregressive
decode every matmul is a matrix-vector product: each weight byte is read once
and used once, no reuse.

Tested the bottleneck instead of assuming it:

- 4-accumulator unroll of `ggml_vec_dot_q4_0_q8_0` to break the loop-carried
  dependency: neutral (47.66). A latency-bound kernel would have sped up;
  this did not.
- Exact algebraic fusion of the DeltaNet kernel. The updated state times q
  equals the old state times q plus delta scaled by (k dot q), so the S-times-k
  and S-times-q reads can share one pass over the 64 KB per-head state matrix
  and the post-update pass disappears. Verified the identity, output correct
  and deterministic (`a5ca62e6c9436cab`). Result: neutral, +0.85%. The
  no-change result is the finding: removing a whole state pass changed
  nothing, so the DeltaNet kernel is not the decode bottleneck (its working
  set is L2-resident). 75% of layers is not 75% of decode time.

Result: both reverted. I concluded, too early as Phase 8 shows, that decode
was memory-bandwidth bound on the Q4_0 weights at a WASM practical ceiling,
and documented that. The user's instruction to keep looking for a
double-digit gain correctly refused that conclusion.

---

## Phase 8: the bandwidth probe, disproving my own ceiling

Goal: stop inferring the ceiling and measure it.

Approach: a standalone WASM SIMD memory-read probe with the same
`-mrelaxed-simd -pthread` flags, a 512 MB buffer (much larger than the 32 MB
L3), 4 independent accumulators, partitioned across threads like ggml's
mul_mat chunks.

Result, which overturned the Phase 7 conclusion:

| Threads | Read bandwidth |
|---------|----------------|
| 1 | 51.80 GB/s |
| 3 | 53.31 GB/s |
| 6 | 51.89 GB/s |
| 12 | 51.13 GB/s |

WASM sustains about 52 GB/s. Decode runs at about 24 GB/s effective (507 MB
model times about 48 tok/s), which is about 46% of the achievable bandwidth,
so about 2 times of room. Decode was not bandwidth-bound. The Phase 7
conclusion was wrong. The gap was being spent somewhere between the loads,
and there was real room to find it.

---

## Phase 9: V8's Liftoff tier

Goal: find what spends the 2x gap, given that seven source-level kernel
changes were all neutral, broken, or worse (Phase 7 plus the failures in
"What didn't work").

Reasoning: the trivial probe loop reached 52 GB/s; every complex-SIMD
dequant variant was neutral. The thing that treats a trivial loop and a
complex loop differently, and that had not been touched, is the JIT. V8
compiles WASM with the Liftoff baseline compiler first and switches to
TurboFan in the background. Liftoff's code for interacting SIMD is much
worse, and for large ggml functions the switch did not seem to complete
within a run. Source micro-optimization cannot help because Liftoff compiles
every variant equally poorly.

Measurement:

| `node` flag | Decode (tok/s) | Note |
|-------------|----------------|------|
| `--liftoff-only` (force baseline) | 37.25 | the slow path, isolated |
| default (Liftoff plus background switch) | 43 to 48 | seems to not finish switching, even at N_GEN=256 |
| `--no-liftoff` (force TurboFan) | 54.53 (median of 5) | +12.5%, sha `348c6e9289611982` unchanged |

Result as first reported (partly wrong): 54.53 tok/s, "+12.5% over default
Node, +69% over the original." I described `--no-liftoff` as the largest gain
after the arch fix. Phase 10 shows that was a measurement artifact: the
"default = 48" baseline it was compared against was itself a cold-blended
average, not steady state. The flag is real and works (the in-process
`v8.setFlagsFromString('--no-liftoff')` route works too on plain `node`,
about 54, bit-identical, but it is a Node-only API), but what it actually
buys is in Phase 10.

---

## Phase 10: the tab question, measuring the ramp, correcting myself

Goal: it ships in a browser tab, and tabs cannot pass `--no-liftoff`. So: do
real users get the speed, or was the win a Node-only benchmark effect?

Approach: from a cold instantiation, measure decode tok/s in 16-token chunks
on default `node` (the closest proxy for a tab, no flags).

| Cumulative tokens | Decode (tok/s) |
|---|---|
| 0 to 16 | ~27 (cold, Liftoff) |
| 16 to 32 | ~54 (switched) |
| 32+ | ~51 to 54, steady, indefinitely |

What this showed, and what I had wrong: V8's background switch finishes
within the first 16 to 32 tokens, well under a second. Only the first 16 or
so tokens are slow. The clean default-Node steady state (first 64 tokens
discarded) is about 51 tok/s, the same class as forced TurboFan. The Phase 9
"+12.5% / +69%" and the earlier "switch never completes even at 256 tokens"
were both the same mistake: a 128-token window started cold mixes about 16
Liftoff tokens with about 112 TurboFan tokens and averages about 48 (128 /
(16/27 + 112/54) is about 48). The bench had been reporting a cold-blended
average the whole time, not steady state.

Result: the honest picture. The durable result that applies everywhere is
the compile-time changes (Phases 4 to 6). Steady-state decode on any V8 (Node
default or a browser tab) is about 51 to 54 tok/s, reached automatically
within about 32 tokens, no flag, no API. `--no-liftoff` and the in-process
route do not raise that, they only remove the short cold period from a short
measurement. For a tab: full speed automatically after under a second. An
optional 32-token warmup behind the load screen (about 0.5 s, in our control)
removes even that. The "+69%" headline became "+50% durable, true steady
state about 51 to 54, and stop quoting cold-blended averages."

---

## Phase 11: re-validating under clean TurboFan

Goal: every "neutral" verdict in Phases 7 to 9 was measured on
Liftoff-contaminated cold-blended windows. If Liftoff makes all SIMD equally
slow, a real TurboFan win would have looked neutral. So: re-run the six under
`node --no-liftoff`, N_GEN=256, 5 trials, against an identically measured
baseline (FMA = 54.15 tok/s, range 54.08 to 54.84). The relaxed i8 dot stayed
excluded because its failure is x86 `pmaddubsw` semantics, which does not
depend on the tier.

| Experiment | Old verdict | TurboFan | Change | Quality | Outcome |
|---|---|---|---|---|---|
| extmul | neutral | 53.54 | -1.1% | bit-identical | holds |
| 4-acc ILP | neutral | 52.95 | -2.2% | coherent | holds |
| branchless fp16 | -18% | 25.0 | -54% | bit-identical | holds, worse (the scalar conversion dominates once the rest is fast) |
| DeltaNet fusion | neutral | 54.42 | +0.5% | coherent | holds, kernel confirmed not the bottleneck |
| direct dispatch | "neutral" | 54.93 | +1.4% | bit-identical | changed verdict |
| Q8_0 K-cache | "neutral" | 56.04 | +3.5% | coherent, lossy | changed verdict |

Result: the measurement debt was real and it cost two wins. Four verdicts
held (and fp16 was confirmed much worse, which shows the re-run was not just
confirming what it wanted to). Two changed: the contaminated measurement had
written off about +1.4% bit-identical (direct dispatch) and about +3.5%
coherent but lossy (Q8_0 K-cache), about +5% of real, stackable speed. The
user was right to refuse "this is the ceiling." Direct-dispatch is the clean
one to ship (no quality risk); Q8_0 K-cache is larger but needs a
multi-prompt quality check first because it is the only kept change that is
not bit-exact.

---

## Phase 12: the realistic ceiling probe

Goal: settle "is this the ceiling?" with a measurement, not inference. The
earlier 52 GB/s came from a trivial sum loop, the easiest possible pattern.
Build a probe that links the real `ggml_vec_dot_q4_0_q8_0` (from
`libggml-cpu.a`, not reimplemented) and runs it in the exact decode pattern:
a DRAM-sized Q4_0 region streamed once per pass, one resident Q8_0 row, rows
split across 6 threads, `--no-liftoff`.

| | GB/s | tok/s-equivalent |
|---|---|---|
| Trivial sum loop | ~52 | raw ceiling, not reachable by real work |
| Real Q4_0 kernel, K=3584 | ~38 | ~87 |
| Real Q4_0 kernel, K=1024 | ~42 | ~96 |
| Actual decode | ~28 effective | ~54 |

Thread scaling (K=3584): 1T 7.8, 3T 23.6, 6T 45.3, 12T 44 GB/s, clean up to
physical cores.

Result, which inverts the question: the Q4_0 kernel alone runs about 1.6 to
2 times faster than decode produces tokens (about 87 to 96 versus about 54).
Decode is at about 55 to 62% of the dominant operation's own ceiling. So this
is not the ceiling, decode is not Q4_0-kernel-bound, and every one of the ten
or so kernel micro-optimizations was neutral because the project had been
tuning a path with about 1.6 times spare capacity. The reachable room is the
40% of each token the probe excludes by construction: the non-Q4_0 quant
kernels (Q6_K lm_head, about 23% of bytes per token, and 18 Q5_K tensors) and
the per-token orchestration (about 1377 graph nodes, each with a
`ggml_barrier`, about 74k syncs per second). The DeltaNet f32 path is ruled
out (the fusion was neutral under clean TurboFan).

---

## Phase 13: the budget, measured fully

Goal: stop saying "the room is somewhere in the other 40%." Measure exactly
where.

Step 1, byte budget from the GGUF. Per-token weight traffic: Q4_0 252 MB
(51%), Q6_K 209 MB (42%, and it is one tensor, the tied `token_embd.weight`
used as the lm_head), Q5_K 26 MB (5%). The project had optimized the Q4_0
path and never looked at an equally large single tensor.

Step 2, real-kernel ceilings (probe, 6 threads, TurboFan): Q4_0 about 40,
Q6_K about 33.5, Q5_K about 31 GB/s. The Q5_K total (26 MB) is under the L3
size so it is mostly resident, not a lever.

Step 3, measure in situ instead of modelling. No-op the Q6_K dot (throwaway
build, garbage output, only timing): decode 54 to about 77 tok/s. So the
lm_head is about 5.5 ms per token, about 30% of decode, which matches the
byte-budget prediction. Budget confirmed by measurement.

Step 4, is the Q6_K kernel improvable? Its 6-bit unpack was a scalar byte
loop (the same pattern as the Q4_0 scalar problem). Vectorized it.
Bit-exact, model sha stayed `348c6e9289611982`, result neutral (+0.4%). A
clean-TurboFan poll re-sweep was also neutral. So Q6_K is like Q4_0:
bandwidth-bound, no source-level change left that helps. Every dominant quant
matmul is now confirmed not improvable in source.

Step 5, profile the "other" part by no-op'ing each suspect. Activation
requant (q8_0 plus q8_K `from_float`) no-op: 54.15 to about 55.5, about 3%.
DeltaNet f32 recurrence no-op: 54.15 to about 55.8, about 3%. Both suspects
measured negligible. The DeltaNet result also explains Phase 11: the fusion
was neutral because the whole kernel is about 0.5 ms per token.

Result, the full measured budget: Q4_0 about 34%, lm_head Q6_K about 30%,
spread-out "other" about 26%, Q5_K about 4%, DeltaNet about 3%, requant about
3%. The two suspected quality-free targets were measured and are not it. The
26% "other" is genuinely spread out (small f32 ops across 24 layers plus the
1377-node graph and its barriers), no single expensive part, no contained
quality-free source change (it is upstream graph-fusion work). No
quality-preserving quick change remains. The one sizable lever is
quality-lossy: lm_head Q6_K to Q4_0, about +11% (about 54 to about 60), only
after a logit and argmax check. End state: the decode budget is fully
understood, and that understanding says further speed is either lossy or
needs graph-level work.

(Note: the Q4_0 34% is the one row that is a probe estimate, not an in-situ
no-op like the others. See the note in PERFORMANCE.md.)

---

## Phase 14: the pre-rework baseline, measured properly on the M4

Goal: the README claimed the pre-rework speed on M4 and Chrome was about 8
tok/s, so the rework looked like about 20x. That 8 was a loose old number.
Measure the pre-rework build with the same method now used for everything
else.

Approach: reconstruct the build from the initial commit (a6b5ffb): the
original `build.sh` flags (no `-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm`, libs
compiled with only `-pthread` so ggml is scalar), the original bindings
(`n_threads = 8`, no persistent threadpool), pristine submodule, no patch.
Built into separate paths so the current build was not disturbed. Node
sanity check first: about 34 tok/s steady-state on the Ryzen, sha
`8f66e60b6a86e729`, which is the original scalar output. Then the same build
on the M4 through the browser harness, steady-state, same prompt.

| Machine | Build | Steady decode |
|---|---|---|
| Ryzen 5 7600, Node, `--no-liftoff` | pre-rework baseline | ~34 |
| Ryzen 5 7600, Node, `--no-liftoff` | shipped | ~54 |
| M4, Chrome | pre-rework baseline | 103.6 |
| M4, Chrome | shipped | ~173 |

Result: the rework is about 1.6x on x86 (34 to 54) and about 1.7x on Apple
Silicon (104 to 173), measured the same way on each machine. The "about 8
tok/s before, about 20x" figure was wrong; the real pre-rework steady-state
on the M4 is about 104. Corrected in the README and PERFORMANCE.md, and the 8
figure was dropped. The pre-rework output is bit-identical across Node and
Chrome (`8f66e60b6a86e729`), the same kind of cross-architecture
determinism check as the shipped build.

Lesson, again: an old loose number is not a baseline. Re-measure the "before"
with the same method as the "after", or the improvement ratio is fiction.

---

## What didn't work

| Approach | Why it failed |
|----------|---------------|
| `-DCMAKE_SYSTEM_PROCESSOR=wasm` | The Emscripten toolchain file resets it; use `-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm` |
| Upstream PR #19590 (`relaxed_dot_i8x16_i7x16` on Q4_0) | Feeds the full-i8 q8_0 activation (plus or minus 127) into the i7 operand, producing garbage ("His His His...") |
| "Correctness-fixed" relaxed i8 by i7 dot (Q4_0 nibble minus 8 in range -8 to 7 as i7, q8_0 as i8) | +5% but garbage output ("EDOensureあた..."). Deeper than an i7-range bug: x86 lowers `relaxed_dot_i8x16_i7x16` to `pmaddubsw`-class operations that treat one operand as unsigned, so it cannot do signed by signed when the i8 operand is ever negative (q8_0 always is). Not usable for Q4_0 by Q8_0 on mainstream hardware, with any operand assignment |
| 4-accumulator unroll of the Q4_0 dot | Neutral, decode was not ALU-latency bound |
| Exact DeltaNet kernel fusion (updated state times q = state times q + delta times (k dot q)) | Correct and coherent but neutral, the DeltaNet state is L2-resident and is not the decode bottleneck |
| `cparams.type_k = GGML_TYPE_Q8_0` (withdrawn) | Not a failure. Looked neutral under Liftoff; Phase 11 under TurboFan was +3.5%, coherent. Pending a multi-prompt quality check |
| Direct (non-indirect) dispatch of the Q4_0 GEMV (withdrawn) | Not a failure. Looked neutral under Liftoff; Phase 11 was +1.4%, bit-identical, no quality risk. Ship candidate |
| Branchless bit-twiddle fp16 to fp32 (drop the 256 KB table) | -18% blended, -54% under clean TurboFan, bit-identical. The table is L2-resident with low-entropy access; a cached load beats about 15 scalar ops, and the gap grows once TurboFan makes everything else fast |
| `extmul` plus `extadd_pairwise` op-count reduction in the Q4_0 dot | Neutral, bit-identical. Fewer WASM-SIMD ops on paper is not fewer machine instructions after V8 lowers them |
| `-flto` on libs and bindings | Neutral (and tested before the arch fix; not revisited because Liftoff, not codegen, was the limiter) |
| `-sALLOW_MEMORY_GROWTH=0 -sINITIAL_MEMORY=2GB` | Neutral or slightly worse, costs 2 GB upfront |
| Threadpool `poll` sweep (0, 1, 10, 50, 100, 1000) | All within noise; kept ggml's default (50) |
| Concluding "memory-bandwidth ceiling" from neutral kernel experiments (Phase 7) | Wrong. The direct bandwidth probe (Phase 8) showed about 2x of room. Infer less, measure more |
| Concluding "+12.5% / +69% from `--no-liftoff`" and "switch never completes" (Phase 9) | Wrong. The ramp probe (Phase 10) showed the switch finishes in 16 to 32 tokens; the baseline was a cold-blended 128-token average. The flag removes a measurement artifact, not a real throughput gap |
| Trusting a single cold 128-token window as "the number" | The worst bug of the project. It mixed the JIT-switch ramp into every result, understated steady state by about 10 to 12%, and mislabeled two real wins (direct-dispatch +1.4%, Q8_0 K-cache +3.5%) as "neutral." A flawed measurement does not only mislead the write-up, it discards working code |
| Strict-only build (post-iOS) | Rejected. Equal on x86, about 14% slower decode on Apple Silicon, and cannot help iOS (the model is over the iOS memory limit regardless). No platform where it is the better choice |

---

## What I would keep in mind next time

- The deterministic output sha was the most useful tool. Judge every change
  on tok/s and on whether the output is still correct. A fast but wrong
  kernel is an obvious failure; a fast bit-identical kernel is a safe win.
- Check that the toolchain did what you think. The largest single gain
  (+29%) was one CMake flag that revealed the SIMD build had been scalar the
  whole time.
- When a toolchain variable does not stick, look for the
  toolchain-file-specific override (`EMSCRIPTEN_SYSTEM_PROCESSOR`, not
  `CMAKE_SYSTEM_PROCESSOR`).
- `relaxed_dot_i8x16_i7x16` cannot do signed by signed where either operand
  can be negative, because x86 lowers it through unsigned `pmaddubsw`.
  Deterministic gibberish is the signature of this kind of bug.
- A neutral result from an exact optimization is a measurement, not a waste.
  It tells you where the bottleneck is not. The DeltaNet fusion no-change
  result located the cost (the Q4_0 GEMV, not the SSM kernel).
- Do not infer a hardware ceiling from neutral micro-optimizations. Build the
  trivial probe (a 30-line bandwidth loop) before concluding "this is the
  limit." It cost one wrong conclusion here.
- 75% of layers is not 75% of time. Whether the working set is in L2 or DRAM
  decides where the cycles go, not the layer count.
- The JIT is part of the system under test. Under V8, benchmark WASM with
  `--no-liftoff`, or you measure the baseline compiler and every source
  conclusion is suspect.
- Fewer SIMD intrinsics on paper is not fewer machine instructions after the
  JIT lowers them. `extadd_pairwise` is not one x86 op.
- A 256 KB lookup table with low-entropy access beats branchless arithmetic
  when the loads are cache-resident and the ALU is the contended resource.
- The cheapest large win needed zero source changes. Check the runtime and
  the toolchain before rewriting kernels.
- Separate the cold ramp from steady state before reporting a number. A
  JIT-tiered runtime has a ramp; a single cold window averages the ramp into
  the result and is wrong in both directions (it understated the real speed
  and inflated a "flag win" that was only the ramp removed).
- Measure the deployment, not a proxy you can flag. A tab cannot pass
  `--no-liftoff`; the real question was what V8 does on its own, and it
  switches in under a second anyway.
- The measurement was wrong for longer than any single hypothesis. Build the
  measurement harness as carefully as the optimization.
- A contaminated benchmark throws away working code. Two real wins (+1.4%
  bit-identical, +3.5% coherent) were filed under "didn't work" because the
  measurement floor was noise. Re-validate "neutral" verdicts whenever the
  measurement method changes.
- "Neutral" is a statement about the instrument as much as the change. Ask
  whether a real effect of that size would even be visible through the noise
  or ramp it was measured in.
- A kernel with spare capacity is not the bottleneck, no matter how busy it
  looks. Ten micro-optimizations were neutral because the Q4_0 dot had about
  1.6 times the spare capacity relative to end-to-end decode, which one isolated-component
  probe would have shown in an hour. Build that probe before the optimization
  work, not after.
- Isolate by linking the real component, not by reimplementing it. The
  ceiling probe called the production `ggml_vec_dot_q4_0_q8_0` straight from
  the `.a`, so there was no risk of measuring a kernel that is not the
  shipped one.
- No profiler available: no-op the component. A one-line early `return` in a
  throwaway build, and the decode change is that component's in-situ cost. It
  measured lm_head at 30%, DeltaNet at 3%, requant at 3%, each a hard number.
  Garbage output does not matter when you are only timing.
- A suspect is not a cost until measured. "DeltaNet is 75% of layers" and
  "the residual is requantized 5 times per layer" both sounded like the
  bottleneck; both measured about 3%. Guessing the budget is how the project
  spent its time on something that was not the bottleneck.
- The value of a SIMD flag does not carry across architectures. Relaxed-SIMD
  was about 0% on x86 and about +14% on Apple Silicon. Measure on the target.

---

## Conclusions

1. The durable speedup is the compile-time changes: the Emscripten
   arch-detection fix (`-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm`, the large one),
   `-mrelaxed-simd`, and the `relaxed_madd` FMA patch. They are in
   `build.sh` and the tree, apply on every runtime including a browser tab,
   need no flags, and keep the output correct.
2. Steady-state decode is about 51 to 54 tok/s on default V8, reached
   automatically within the first 16 to 32 tokens (under a second). Only the
   first 16 or so tokens are at the slow Liftoff rate (about 27).
3. `--no-liftoff` (CLI) and `v8.setFlagsFromString('--no-liftoff')`
   (in-process, Node-only) are not throughput wins. Default V8 reaches the
   same steady state on its own. They remove the cold period from a short
   measurement and slightly help time-to-first-token. Useful for honest
   benchmarking, not for end-user tok/s.
4. The browser tab gets full speed automatically: same V8 behaviour, about
   51 to 54 tok/s after under a second, no flag or API. An optional 32-token
   warmup behind the load screen removes the short slow start; not required.
5. The benchmark method was the longest-standing and most expensive bug. A
   single cold 128-token window understated steady state by about 10 to 12%
   and mislabeled two working changes as "neutral" (Phase 11). Future
   numbers: discard the first 32 to 64 tokens (or use `--no-liftoff`) and
   report steady state.
6. Re-validation under clean TurboFan (Phase 11) recovered two changes:
   direct-dispatch +1.4% (bit-identical, clean to ship) and Q8_0 K-cache
   +3.5% (coherent, lossy, ship after a multi-prompt quality check). About
   +5% of stackable speed was hidden behind the bad measurement.
7. Is this the ceiling? No (Phase 12). The real-kernel probe shows the Q4_0
   dot can sustain about 87 to 96 tok/s-equivalent versus about 54 actual, so
   decode is at about 55 to 62% of the dominant operation's own ceiling, not
   kernel-bound. The room is outside the Q4_0 path: the non-Q4_0 quant
   kernels (Q6_K lm_head, Q5_K) and the per-token graph orchestration. The
   DeltaNet f32 path is excluded. The kernel-micro-optimization work
   (Phases 7 to 11) was the wrong target, shown by measurement, not guessed.
8. Budget fully measured (Phase 13), each share by component no-op except
   Q4_0: Q4_0 about 34%, lm_head Q6_K about 30%, spread-out "other" about
   26%, Q5_K about 4%, DeltaNet about 3%, requant about 3%. The two suspected
   quality-free targets (DeltaNet, requant) were measured at about 3% each.
   The 26% "other" has no single expensive part (small f32 ops across 24
   layers plus the 1377-node graph and its barriers), so it is graph-level
   work.
9. No quality-free quick change remains. Both large matmuls (about 64%) are
   bandwidth-bound with no source-level change left that helps. Ship order:
   direct-dispatch (clean, +1.4%); Q8_0 K-cache (+3.5%, after a quality
   check); lm_head Q6_K to Q4_0 (about +11%, the only sizable lever,
   quality-lossy, after a logit and argmax check); a WASM tiled GEMM for
   prefill. Further decode speed is now either lossy or needs upstream
   graph-level work, not a contained source change.
10. Measured pre-rework versus shipped, same method on each machine (Phase
    14): about 34 to about 54 tok/s on Ryzen and Node, about 104 to about
    173 tok/s on M4 and Chrome. The rework is about 1.6x on x86 and about
    1.7x on Apple Silicon. The earlier "about 8 tok/s before, about 20x"
    claim was a loose measurement and was wrong; it has been corrected and
    dropped.

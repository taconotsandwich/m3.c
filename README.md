# m3.c

`m3.c` is a native C11 inference runtime for the official
[MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
model. It runs the complete caption-and-lyrics-to-stereo-waveform pipeline on
Apple Metal without Python, PyTorch, Diffusers, or a CPU inference fallback.

The public API is intentionally small: open a trusted local model snapshot,
generate an owned waveform, read either planar channel, or write a 44.1 kHz
stereo IEEE-float WAVE file.

## Status

Current version: `0.1.0`

Verified on an Apple M4 Mac with 32 GiB unified memory:

- Strict suite: 174 cases, 2,746 checks, 0 skipped.
- ASan and UBSan suite: 174 cases, 2,746 checks, 0 skipped.
- Official pinned-weight Metal E2E: passed every generation phase.
- E2E smoke request: one generated frame, 1,536 samples per channel.
- Official model revision:
  `bd348f9c49ea3c1b39f33ace3436f8fad435f24e`.

The runtime implements:

- Official model-directory, config, tensor-name, dtype, rank, and shape checks.
- Music3 prompt normalization and tokenizer loading.
- Qwen semantic generation with classifier-free guidance.
- Seven-book RVQ depth decoding and complete-frame Qwen feedback.
- Condition encoding and the 36-layer flow-matching transformer.
- Thirty-step Euler flow synthesis with official chunk overlap behavior.
- Weight-normalized stereo vocoder decoding.
- Exact post-decode crop, planar assembly, clamping, and WAVE interleaving.
- Transactional outputs, deterministic RNG state, progress, and cancellation.

Neural operations run on the Metal GPU. The Apple Neural Engine is not used.
CPU work is limited to orchestration, validation, sampling, weight staging, and
bounded storage transfers.

## Requirements

- macOS with a Metal-capable GPU.
- Apple Clang with C11 and Objective-C ARC support.
- GNU Make.
- Enough local storage for the selected official model snapshot.
- Substantial unified memory or swap for official-weight inference.

The full official pipeline was verified on a 32 GiB M4. Smaller-memory systems
may fail the runtime's allocation preflight or experience heavy swap use.

## Build and test

Build the static library:

```sh
make -j8
```

Run the strict real-Metal test suite:

```sh
make test -j8
```

Run the AddressSanitizer and UndefinedBehaviorSanitizer suite:

```sh
make test-sanitize -j8
```

Remove generated build products:

```sh
make clean
```

The Apple arm64 driver does not accept standalone `-fsanitize=leak`, but its
ASan runtime supports integrated leak checking. Run the same sanitizer suite
with the leak detector explicitly enabled when that additional gate is needed:

```sh
ASAN_OPTIONS='detect_leaks=1:leak_check_at_exit=1' make test-sanitize -j8
```

The tests also assert backend allocation baselines on ownership-sensitive
paths; LeakSanitizer covers CPU heap ownership, not Metal driver resources.

## Official model weights

Model binaries are local runtime data and are intentionally excluded from Git.
Use this repository-local layout:

```text
models/
└── MiniMax-Music3-bd348/
    ├── modular_model_index.json
    ├── scheduler/
    ├── tokenizer/
    ├── language_model/
    │   ├── model-00001-of-00004.safetensors
    │   ├── model-00002-of-00004.safetensors
    │   ├── model-00003-of-00004.safetensors
    │   └── model-00004-of-00004.safetensors
    ├── rvq_depth_decoder/
    │   └── diffusion_pytorch_model.safetensors
    ├── condition_encoder/
    │   └── diffusion_pytorch_model.safetensors
    ├── transformer/
    │   ├── diffusion_pytorch_model-00001-of-00002.safetensors
    │   └── diffusion_pytorch_model-00002-of-00002.safetensors
    └── vocoder/
        └── diffusion_pytorch_model.safetensors
```

Install the current Hugging Face CLI, then download the exact engine-required
snapshot:

```sh
hf download MiniMaxAI/MiniMax-Music3 \
  --revision bd348f9c49ea3c1b39f33ace3436f8fad435f24e \
  --local-dir models/MiniMax-Music3-bd348 \
  --include modular_model_index.json \
  --include 'scheduler/*' \
  --include 'tokenizer/*' \
  --include 'language_model/*' \
  --include 'rvq_depth_decoder/*' \
  --include 'condition_encoder/*' \
  --include 'transformer/*' \
  --include 'vocoder/*'
```

This is 22 engine-consumed files and approximately 27 GiB on disk, including
nine weight files. Verify the pinned files before opening the model root:

```sh
uvx --from huggingface_hub hf cache verify \
  MiniMaxAI/MiniMax-Music3 \
  --revision bd348f9c49ea3c1b39f33ace3436f8fad435f24e \
  --local-dir models/MiniMax-Music3-bd348
```

The filtered snapshot intentionally omits 66 unrelated remote files, so the
verifier warns that they are missing. A successful required-file verification
reports `checked=22` with no checksum mismatch; local `.cache/huggingface`
metadata and lock files may also be reported as remote-repository extras.

The Hub repository also contains an older/reference packaging family,
including `qwen_7B/`, `flowmatching_vae.pth`, and `dav.pth`, plus scripts and
assets. A complete archival mirror is 88 files and about 57.4 GB:

```sh
hf download MiniMaxAI/MiniMax-Music3 \
  --revision bd348f9c49ea3c1b39f33ace3436f8fad435f24e \
  --local-dir models/MiniMax-Music3-bd348-full
```

Those alternate artifacts are not inputs to this runtime. The largest single
checkpoint is not a complete Music3 pipeline: semantic, RVQ, condition, flow,
and vocoder weights are independent required components.

`m3_music3_engine_open` validates the official structure and tensor contracts,
but the model root is caller-trusted. It does not authenticate the Hub revision
or hash file contents, so use the checksum verification command above.

## Minimal generation example

```c
#include "m3.h"

#include <stdio.h>

int main(void)
{
    static const char caption[] = "Warm electronic instrumental music.";
    static const char lyrics[] = "[verse]\nHello world.";
    m3_music3_engine *engine = NULL;
    m3_music3_output *output = NULL;
    m3_music3_request request = {
        {caption, sizeof(caption) - 1U},
        {lyrics, sizeof(lyrics) - 1U},
        1U,
        20260815U,
        0U,
    };
    m3_error error;
    m3_status status;

    m3_error_reset(&error);
    status = m3_music3_engine_open(
        &engine, "models/MiniMax-Music3-bd348", &error);
    if (status == M3_STATUS_OK) {
        status = m3_music3_generate(
            engine, &request, NULL, NULL, &output, &error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_output_write_wav(
            output, "music3.wav", &error);
    }
    if (status != M3_STATUS_OK) {
        (void)fprintf(stderr, "%s\n", m3_error_message(&error));
    }

    m3_music3_output_free(output);
    m3_music3_engine_free(engine);
    return status == M3_STATUS_OK ? 0 : 1;
}
```

After `make`, compile the example against the static library:

```sh
clang -std=c11 -O3 -Iinclude example.c build/libm3.a -lm \
  -framework Foundation \
  -framework Metal \
  -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph \
  -framework Accelerate \
  -o example
```

`maximum_frames` must be between 1 and 9,000. A value of 1 is suitable only
for a smoke test. Generation is synchronous, the engine is non-reentrant and
not thread-safe, and every output must be freed before its owning engine.

## Official-model benchmark and diagnostics

Build the public-API benchmark harness:

```sh
make benchmark -j8
```

The repository includes a macOS monitor that records phase timing, CPU and GPU
utilization, memory pressure, swap, and process RSS. It cancels gracefully if a
configured safety limit is reached and never retries a stochastic generation:

```sh
uv run tools/music3_debug.py
```

Defaults use `models/MiniMax-Music3-bd348`, one maximum frame, seed `20260815`,
and sequence `0`. The full log is written to
`/private/tmp/m3_music3_debug.log`; a successful run retains the validated WAVE
at `/private/tmp/m3_music3_debug.wav`. Use `--delete-wave` to remove it after
the run. Inspect the machine without loading the model with:

```sh
uv run tools/music3_debug.py --sample-only
```

Every threshold, prompt, output path, seed, sequence, and frame cap is exposed
through `--help`; `--quiet` writes only the log. The C harness uses only the
public `include/m3.h` API and
validates phase monotonicity, both complete planar channels, finite/clamped
samples, exact WAVE metadata and interleaving, and deterministic output hashes.

## Measured performance

The current software optimization pass was measured on the same fanless M4
Mac, exact pinned weights, one-frame request, seed `20260815`, and sequence `0`.
Each retained change preserved the two output hashes bit-for-bit.

| Revision | Total | Semantic | Flow | Decode |
| --- | ---: | ---: | ---: | ---: |
| Before this pass (`0c367e7`) | 169.40 s | 37.17 s | 63.84 s | 55.26 s |
| Current | 65.57 s | 17.78 s | 32.83 s | 1.30 s |

That is a 2.58x end-to-end speedup, or 61.3% lower wall time. The retained
changes remove repeated convolution/dense/attention address work, tile dense
kernels, share attention probabilities across output channels, and reduce
flow command-buffer submissions. Larger dense blocks, a small-row tile variant,
Qwen-wide command batching, and typed common loads were measured and rejected
because they were slower on this workload.

A fresh post-integration run through `tools/music3_debug.py` completed in
66.60 seconds with the same two hashes; the 1.6% difference from the table is
normal run-to-run variation on the fanless machine.

The runtime accepts F16 and BF16 tensors for supported operators, but the
official pinned Music3 snapshot is F32 and its schema loader intentionally
requires those exact official dtypes. Running a separately converted FP16
checkpoint would require one deliberate model-schema/numerical-contract change;
it is not silently accepted as the official model.

## Public API

The complete public surface is declared in [`include/m3.h`](include/m3.h):

- `m3_music3_engine_open` and `m3_music3_engine_free`
- `m3_music3_generate`
- `m3_music3_output_get_info`
- `m3_music3_output_read_channel_f32`
- `m3_music3_output_write_wav`
- `m3_music3_output_free`

Generation progress is reported as monotonic phase-local counters covering
preparation, semantic weights, semantic decoding, flow weights, flow solving,
vocoder weights, materialization, decoding, and assembly. Returning `false`
from the callback cancels generation without replacing the caller's prior
output.

## License

`m3.c` is licensed under GPL-2.0-only. See [`COPYING`](COPYING). Bundled
Unicode data is covered by [`LICENSES/Unicode-3.0.txt`](LICENSES/Unicode-3.0.txt).

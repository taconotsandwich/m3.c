/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_semantic_internal.h"

#include "m3_tokenizer.h"

#include <stdint.h>
#include <string.h>

#define M3_SEMANTIC_CACHE_POSITION_BYTES UINT64_C(294912)
#define M3_SEMANTIC_ROPE_POSITION_BYTES UINT64_C(256)
#define M3_SEMANTIC_PREFILL_TOKEN_BYTES UINT64_C(241664)
#define M3_SEMANTIC_RESULT_BYTES UINT64_C(213004)
#define M3_SEMANTIC_FRAME_BYTES UINT64_C(65536)
#define M3_SEMANTIC_RVQ_PHASE_BYTES UINT64_C(1937460)
#define M3_SEMANTIC_CACHE_STORAGE_POSITION_BYTES UINT64_C(4096)
#define M3_SEMANTIC_PREFILL_STORAGE_TOKEN_BYTES UINT64_C(49152)
#define M3_SEMANTIC_RVQ_LARGEST_STORAGE_BYTES UINT64_C(196608)

static bool m3_semantic_multiply(uint64_t left, uint64_t right,
                                 uint64_t *product)
{
    if (left != 0U && right > UINT64_MAX / left) {
        return false;
    }
    *product = left * right;
    return true;
}

static bool m3_semantic_add(uint64_t left, uint64_t right,
                            uint64_t *sum)
{
    if (right > UINT64_MAX - left) {
        return false;
    }
    *sum = left + right;
    return true;
}

static void m3_semantic_maximum(uint64_t value, uint64_t *maximum)
{
    if (value > *maximum) {
        *maximum = value;
    }
}

m3_status m3_semantic_plan_build(uint64_t prompt_tokens,
                                 uint64_t frame_limit,
                                 m3_semantic_plan *plan,
                                 m3_error *error)
{
    m3_semantic_plan built;
    uint64_t activation;
    uint64_t cache;
    uint64_t cache_storage;
    uint64_t output;
    uint64_t persistent;
    uint64_t prefill_storage;
    uint64_t progress;
    uint64_t rope;

    (void)memset(&built, 0, sizeof(built));
    if (plan == NULL || prompt_tokens < 3U ||
        prompt_tokens > M3_TOKENIZER_MAX_PROMPT_IDS ||
        frame_limit == 0U || frame_limit > M3_SEMANTIC_MAX_FRAMES ||
        frame_limit > UINT64_MAX - prompt_tokens) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "semantic generation plan is out of range");
    }
    built.prompt_tokens = prompt_tokens;
    built.frame_limit = frame_limit;
    built.cache_capacity = prompt_tokens + frame_limit;
    if (built.cache_capacity > M3_QWEN_MAX_CAPACITY ||
        !m3_semantic_multiply(
            built.cache_capacity, M3_SEMANTIC_CACHE_POSITION_BYTES,
            &cache) ||
        !m3_semantic_multiply(
            built.cache_capacity, M3_SEMANTIC_ROPE_POSITION_BYTES,
            &rope) ||
        !m3_semantic_add(cache, rope, &persistent) ||
        !m3_semantic_multiply(
            prompt_tokens, M3_SEMANTIC_PREFILL_TOKEN_BYTES,
            &activation) ||
        !m3_semantic_add(persistent, activation,
                         &built.prefill_added_bytes) ||
        !m3_semantic_add(built.prefill_added_bytes,
                         M3_SEMANTIC_RESULT_BYTES,
                         &built.prefill_added_bytes) ||
        !m3_semantic_multiply(frame_limit, M3_SEMANTIC_FRAME_BYTES,
                              &output) ||
        !m3_semantic_add(persistent, output,
                         &built.frame_added_bytes) ||
        !m3_semantic_add(built.frame_added_bytes,
                         M3_SEMANTIC_RVQ_PHASE_BYTES,
                         &built.frame_added_bytes) ||
        !m3_semantic_multiply(
            frame_limit, M3_SEMANTIC_FULL_ATTEMPT_PROGRESS, &progress) ||
        !m3_semantic_add(progress,
                         M3_SEMANTIC_QWEN_PROGRESS +
                             M3_SEMANTIC_FINAL_ATTEMPT_PROGRESS,
                         &built.progress_total)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "semantic generation plan overflows");
    }
    built.maximum_added_bytes = built.prefill_added_bytes;
    m3_semantic_maximum(built.frame_added_bytes,
                        &built.maximum_added_bytes);
    if (output > SIZE_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "semantic output size overflows size_t");
    }
    built.output_bytes = (size_t)output;
    if (!m3_semantic_multiply(
            built.cache_capacity,
            M3_SEMANTIC_CACHE_STORAGE_POSITION_BYTES, &cache_storage) ||
        !m3_semantic_multiply(
            prompt_tokens, M3_SEMANTIC_PREFILL_STORAGE_TOKEN_BYTES,
            &prefill_storage)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "semantic storage plan overflows");
    }
    built.largest_storage_bytes = output;
    m3_semantic_maximum(cache_storage, &built.largest_storage_bytes);
    m3_semantic_maximum(prefill_storage, &built.largest_storage_bytes);
    m3_semantic_maximum(M3_SEMANTIC_RVQ_LARGEST_STORAGE_BYTES,
                        &built.largest_storage_bytes);
    *plan = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_semantic_preflight(m3_backend *backend,
                                const m3_semantic_plan *plan,
                                m3_error *error)
{
    m3_backend_allocation_stats stats;
    m3_backend_info info;
    uint64_t live;
    uint64_t peak;
    m3_status status;

    if (backend == NULL || plan == NULL || plan->output_bytes == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "semantic allocation plan is invalid");
    }
    status = m3_backend_get_info(backend, &info, error);
    if (status == M3_STATUS_OK) {
        status = m3_backend_get_allocation_stats(backend, &stats, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    live = (uint64_t)stats.live_allocated_bytes;
    if ((size_t)live != stats.live_allocated_bytes ||
        plan->maximum_added_bytes > UINT64_MAX - live) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "semantic working-set plan overflows");
    }
    if (plan->largest_storage_bytes > info.maximum_storage_bytes) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "semantic tensor exceeds backend storage limit");
    }
    peak = live + plan->maximum_added_bytes;
    if (info.recommended_working_set_bytes != 0U &&
        peak > info.recommended_working_set_bytes) {
        return m3_error_set(
            error, M3_STATUS_OUT_OF_MEMORY,
            "semantic runtime exceeds backend recommended working set");
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

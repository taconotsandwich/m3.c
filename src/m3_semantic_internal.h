/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_SEMANTIC_INTERNAL_H
#define M3_SEMANTIC_INTERNAL_H

#include "m3_guided_sampling.h"
#include "m3_qwen_runtime.h"
#include "m3_rvq_condition.h"
#include "m3_semantic_runtime.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_SEMANTIC_QWEN_PROGRESS 39U
#define M3_SEMANTIC_RVQ_PROGRESS 7U
#define M3_SEMANTIC_FULL_ATTEMPT_PROGRESS 49U
#define M3_SEMANTIC_FINAL_ATTEMPT_PROGRESS 9U

typedef struct {
    uint64_t prompt_tokens;
    uint64_t frame_limit;
    uint64_t cache_capacity;
    uint64_t progress_total;
    uint64_t prefill_added_bytes;
    uint64_t frame_added_bytes;
    uint64_t maximum_added_bytes;
    uint64_t largest_storage_bytes;
    size_t output_bytes;
} m3_semantic_plan;

typedef struct {
    void *context;
    m3_backend *backend;
    m3_status (*start)(void *context, uint64_t cache_capacity,
                       m3_error *error);
    void (*finish)(void *context);
    m3_status (*prefill)(void *context,
                         const m3_tensor_view *prompt_ids,
                         m3_progress_callback progress,
                         void *progress_context, m3_qwen_state *state,
                         m3_error *error);
    m3_status (*semantic_embedding)(void *context, uint32_t code,
                                    m3_tensor_view *embedding,
                                    m3_error *error);
    m3_status (*decode_frame)(
        void *context, const m3_tensor_view *last_hidden,
        const m3_tensor_view *semantic_embedding, const float *uniforms,
        size_t uniform_count, m3_progress_callback progress,
        void *progress_context, m3_rvq_frame *frame, m3_error *error);
    m3_status (*build_feedback)(
        void *context, const m3_tensor_view *semantic_embedding,
        const m3_rvq_frame *frame, m3_rvq_feedback *feedback,
        m3_error *error);
    m3_status (*advance)(void *context,
                         const m3_tensor_view *feedback,
                         m3_progress_callback progress,
                         void *progress_context, m3_qwen_state *state,
                         m3_error *error);
} m3_semantic_operations;

m3_status m3_semantic_plan_build(uint64_t prompt_tokens,
                                 uint64_t frame_limit,
                                 m3_semantic_plan *plan,
                                 m3_error *error);
m3_status m3_semantic_preflight(m3_backend *backend,
                                const m3_semantic_plan *plan,
                                m3_error *error);
m3_status m3_semantic_validate_request(
    m3_backend *backend, const m3_tensor_view *prompt_ids,
    uint64_t frame_limit, const m3_rng *rng,
    const m3_semantic_output *output, m3_semantic_plan *plan,
    m3_error *error);
m3_status m3_semantic_generate_core(
    const m3_semantic_operations *operations,
    const m3_tensor_view *prompt_ids, uint64_t frame_limit, m3_rng *rng,
    m3_progress_callback progress, void *progress_context,
    m3_semantic_output *output, m3_error *error);

#endif

/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_QWEN_RUNTIME_H
#define M3_QWEN_RUNTIME_H

#include "m3_progress.h"
#include "m3_tensor.h"
#include "m3_weight_stage.h"

#include <stddef.h>
#include <stdint.h>

#define M3_QWEN_VOCAB_SIZE 200000U
#define M3_QWEN_HIDDEN_SIZE 4096U
#define M3_QWEN_LAYER_COUNT 36U
#define M3_QWEN_QUERY_HEADS 32U
#define M3_QWEN_KEY_VALUE_HEADS 8U
#define M3_QWEN_HEAD_DIMENSION 128U
#define M3_QWEN_INTERMEDIATE_SIZE 12288U
#define M3_QWEN_MAX_CAPACITY 14000U
#define M3_QWEN_EOS_TOKEN_ID 151670U
#define M3_QWEN_SEMANTIC_TOKEN_START 151675U
#define M3_QWEN_SEMANTIC_TOKEN_COUNT 16384U
#define M3_QWEN_WEIGHT_COUNT 399U

typedef struct m3_qwen_runtime m3_qwen_runtime;

/* Logits preserve [conditional, unconditional] row order. EOS is F32 [2,1]
 * and semantics are F32 [2,16384], mapped to IDs 151675..168058. */
typedef struct {
    const m3_tensor_view *eos_logits;
    const m3_tensor_view *semantic_logits;
} m3_qwen_logits;

typedef struct {
    /* BF16 [4096], shared by the two identical step-token rows. */
    const m3_tensor_view *token_embedding;
    /* Final normalized BF16 [2,4096] current-token hidden rows. */
    const m3_tensor_view *hidden;
    const m3_tensor_view *eos_logits;
    const m3_tensor_view *semantic_logits;
} m3_qwen_step_result;

/* The immutable staged language model and its backend are borrowed and must
 * outlive the runtime. A successful call's result views borrow runtime-owned
 * storage and remain valid until the next successful call or runtime free.
 * Failed and cancelled calls leave the published token count and result views
 * unchanged. */
m3_status m3_qwen_runtime_create(m3_qwen_runtime **runtime,
                                 const m3_weight_stage *language_model,
                                 uint64_t cache_capacity,
                                 m3_error *error);
void m3_qwen_runtime_free(m3_qwen_runtime *runtime);

/* Prompt IDs are borrowed for the call, reside on the runtime backend, and
 * must be contiguous I32 [2,T] in [conditional, unconditional] row order.
 * Prefill is legal only while the published token count is zero. */
m3_status m3_qwen_runtime_prefill(m3_qwen_runtime *runtime,
                                  const m3_tensor_view *prompt_ids,
                                  m3_progress_callback progress,
                                  void *progress_context,
                                  m3_qwen_logits *logits,
                                  m3_error *error);

/* token_id is an absolute official semantic token ID. The runtime embeds the
 * same token for both rows at the next position. On success it publishes the
 * raw row-independent BF16 token embedding [4096], final normalized BF16
 * hidden rows [2,4096], and next-token F32 logits. */
m3_status m3_qwen_runtime_step(m3_qwen_runtime *runtime,
                               uint32_t token_id,
                               m3_progress_callback progress,
                               void *progress_context,
                               m3_qwen_step_result *result,
                               m3_error *error);

uint64_t m3_qwen_runtime_token_count(const m3_qwen_runtime *runtime);
uint64_t m3_qwen_runtime_cache_capacity(const m3_qwen_runtime *runtime);
size_t m3_qwen_runtime_cache_bytes(const m3_qwen_runtime *runtime);

#endif

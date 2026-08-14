/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_QWEN_INTERNAL_H
#define M3_QWEN_INTERNAL_H

#include "m3_qwen_runtime.h"
#include "m3_runtime_workspace.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_QWEN_WEIGHT_NAME_CAPACITY 128U

typedef struct {
    uint64_t vocab_size;
    uint64_t hidden_size;
    uint64_t layer_count;
    uint64_t query_heads;
    uint64_t key_value_heads;
    uint64_t head_dimension;
    uint64_t intermediate_size;
    uint64_t eos_token_id;
    uint64_t semantic_token_start;
    uint64_t semantic_token_count;
    float rms_epsilon;
    float rope_theta;
} m3_qwen_dimensions;

typedef struct {
    const m3_tensor_view *input_norm;
    const m3_tensor_view *query_weight;
    const m3_tensor_view *key_weight;
    const m3_tensor_view *value_weight;
    const m3_tensor_view *output_weight;
    const m3_tensor_view *query_norm;
    const m3_tensor_view *key_norm;
    const m3_tensor_view *post_attention_norm;
    const m3_tensor_view *gate_weight;
    const m3_tensor_view *up_weight;
    const m3_tensor_view *down_weight;
} m3_qwen_layer_weights;

typedef struct {
    const m3_tensor_view *embedding;
    const m3_tensor_view *final_norm;
    const m3_tensor_view *head;
    m3_qwen_layer_weights layers[M3_QWEN_LAYER_COUNT];
} m3_qwen_weights;

typedef struct {
    m3_weight_requirement requirements[M3_QWEN_WEIGHT_COUNT];
    char names[M3_QWEN_WEIGHT_COUNT][M3_QWEN_WEIGHT_NAME_CAPACITY];
    size_t count;
} m3_qwen_weight_plan;

typedef struct {
    m3_runtime_workspace workspace;
    uint64_t capacity;
    size_t byte_count;
} m3_qwen_cache;

typedef struct {
    const m3_qwen_dimensions *dimensions;
    const m3_qwen_weights *weights;
    m3_backend *backend;
    m3_command_executor *executor;
    m3_tensor_view *cache_views;
    const m3_tensor_view *cosines;
    const m3_tensor_view *sines;
    uint64_t cache_capacity;
    uint64_t token_count;
    m3_runtime_workspace published;
} m3_qwen_forward_state;

typedef enum {
    M3_QWEN_FORWARD_PREFILL = 0,
    M3_QWEN_FORWARD_STEP
} m3_qwen_forward_kind;

typedef struct {
    const m3_tensor_view *token_embedding;
    const m3_tensor_view *hidden;
    const m3_tensor_view *eos_logits;
    const m3_tensor_view *semantic_logits;
} m3_qwen_forward_result;

struct m3_qwen_runtime {
    const m3_weight_stage *stage;
    m3_qwen_dimensions dimensions;
    m3_qwen_weights weights;
    m3_qwen_cache cache;
    m3_runtime_workspace rope;
    m3_command_executor executor;
    m3_qwen_forward_state forward;
};

const m3_qwen_dimensions *m3_qwen_official_dimensions(void);
m3_status m3_qwen_weight_plan_init(m3_qwen_weight_plan *plan,
                                   m3_error *error);
m3_status m3_qwen_weights_bind(m3_qwen_weights *weights,
                               const m3_weight_stage *stage,
                               m3_error *error);

m3_status m3_qwen_cache_spec(const m3_qwen_dimensions *dimensions,
                             uint64_t capacity, size_t index,
                             m3_runtime_tensor_spec *spec,
                             m3_error *error);
m3_status m3_qwen_cache_measure(const m3_qwen_dimensions *dimensions,
                                uint64_t capacity, size_t *storage_count,
                                size_t *byte_count, m3_error *error);
void m3_qwen_cache_init(m3_qwen_cache *cache);
void m3_qwen_cache_dispose(m3_qwen_cache *cache);
m3_status m3_qwen_cache_build(m3_qwen_cache *cache, m3_backend *backend,
                              const m3_qwen_dimensions *dimensions,
                              uint64_t capacity, m3_error *error);
m3_status m3_qwen_rope_build(m3_runtime_workspace *rope,
                             m3_backend *backend,
                             const m3_qwen_dimensions *dimensions,
                             uint64_t capacity, m3_error *error);

void m3_qwen_forward_state_init(m3_qwen_forward_state *state);
void m3_qwen_forward_state_dispose(m3_qwen_forward_state *state);
m3_status m3_qwen_forward_execute(m3_qwen_forward_state *state,
                                  m3_qwen_forward_kind kind,
                                  const m3_tensor_view *ids,
                                  m3_progress_callback progress,
                                  void *progress_context,
                                  m3_qwen_forward_result *result,
                                  m3_error *error);

typedef struct {
    m3_tensor_view *hidden;
    m3_tensor_view *normalized;
    m3_tensor_view *query;
    m3_tensor_view *key;
    m3_tensor_view *value;
    m3_tensor_view *rotated_query;
    m3_tensor_view *rotated_key;
    m3_tensor_view *attention;
    m3_tensor_view *attention_reordered;
    m3_tensor_view *projection;
    m3_tensor_view *gate;
    m3_tensor_view *up;
    m3_tensor_view *down;
} m3_qwen_layer_workspace;

m3_status m3_qwen_layer_execute(
    m3_command_executor *executor, const m3_qwen_dimensions *dimensions,
    const m3_qwen_layer_weights *weights,
    m3_qwen_layer_workspace *workspace, m3_tensor_view *key_cache,
    m3_tensor_view *value_cache, const m3_tensor_view *cosines,
    const m3_tensor_view *sines, uint64_t position,
    m3_error *error);

#endif

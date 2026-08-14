/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_RVQ_CONDITION_INTERNAL_H
#define M3_RVQ_CONDITION_INTERNAL_H

#include "m3_rvq_condition.h"
#include "m3_runtime_workspace.h"

#include <stdbool.h>

typedef struct {
    uint32_t hidden_size;
    uint32_t layer_count;
    uint32_t attention_head_count;
    uint32_t intermediate_size;
    uint32_t position_count;
    m3_dtype dtype;
} m3_rvq_config;

typedef struct {
    uint32_t hidden_size;
    uint32_t layer_count;
    uint32_t output_size;
    uint32_t resize_numerator;
    uint32_t resize_denominator;
} m3_condition_config;

typedef struct {
    char name_storage[47][128];
    m3_weight_requirement requirements[47];
    size_t count;
} m3_rvq_requirement_set;

typedef struct {
    char name_storage[4][128];
    m3_weight_requirement requirements[4];
    size_t count;
} m3_condition_requirement_set;

m3_status m3_rvq_requirements(m3_rvq_requirement_set *set,
                              m3_error *error);
m3_status m3_condition_requirements(m3_condition_requirement_set *set,
                                    m3_error *error);

m3_status m3_rvq_condition_preflight(
    m3_backend *backend, size_t output_bytes,
    const m3_runtime_tensor_spec *specs, size_t spec_count,
    m3_error *error);
m3_status m3_rvq_condition_cancel(m3_progress_callback progress,
                                  void *context, uint64_t completed,
                                  uint64_t total, const char *operation,
                                  m3_error *error);
m3_status m3_rvq_condition_check_view(
    m3_backend *backend, const m3_tensor_view *view, m3_dtype dtype,
    uint8_t rank, const uint64_t *shape, const char *name,
    m3_error *error);

#define M3_RVQ_WORKSPACE_COUNT 16U

typedef enum {
    M3_RVQ_WS_SEQUENCE = 0,
    M3_RVQ_WS_POSITION_IDS,
    M3_RVQ_WS_CODE_IDS,
    M3_RVQ_WS_TOKEN,
    M3_RVQ_WS_POSITIONS,
    M3_RVQ_WS_HIDDEN,
    M3_RVQ_WS_HIDDEN_TEMP,
    M3_RVQ_WS_NORM,
    M3_RVQ_WS_QUERY,
    M3_RVQ_WS_KEY,
    M3_RVQ_WS_VALUE,
    M3_RVQ_WS_ATTENTION,
    M3_RVQ_WS_REORDER,
    M3_RVQ_WS_GATE,
    M3_RVQ_WS_UP,
    M3_RVQ_WS_LOGITS
} m3_rvq_workspace_index;

m3_status m3_rvq_validate(
    m3_backend *backend, const m3_rvq_config *config,
    const m3_rvq_weights *weights, const m3_tensor_view *last_hidden,
    const m3_tensor_view *semantic_embedding, const float *uniforms,
    size_t uniform_count, const m3_rvq_frame *frame,
    size_t *output_bytes, m3_error *error);
void m3_rvq_workspace_specs(const m3_rvq_config *config,
                            m3_runtime_tensor_spec *specs);
m3_status m3_rvq_decode_frame_core(
    m3_backend *backend, const m3_rvq_config *config,
    const m3_rvq_weights *weights, const m3_tensor_view *last_hidden,
    const m3_tensor_view *semantic_embedding, const float *uniforms,
    size_t uniform_count, m3_progress_callback progress,
    void *progress_context, m3_rvq_frame *frame, m3_error *error);

size_t m3_rvq_sequence_length(size_t head_index);
bool m3_rvq_next_embedding_id(size_t head_index, uint32_t code,
                              int32_t *embedding_id);

#define M3_CONDITION_WORKSPACE_COUNT 6U

typedef enum {
    M3_CONDITION_WS_LAYER_WEIGHTS = 0,
    M3_CONDITION_WS_CAST_LAYER,
    M3_CONDITION_WS_WEIGHTED_LAYER,
    M3_CONDITION_WS_MIX,
    M3_CONDITION_WS_CONVOLUTION,
    M3_CONDITION_WS_RESIZED
} m3_condition_workspace_index;

m3_status m3_condition_output_length(
    uint64_t frames, uint32_t numerator, uint32_t denominator,
    uint64_t *length, m3_error *error);
m3_status m3_condition_validate(
    m3_backend *backend, const m3_condition_config *config,
    const m3_condition_weights *weights, const m3_tensor_view *frames,
    const m3_condition_output *output, uint64_t *output_length,
    size_t *output_bytes, m3_error *error);
void m3_condition_workspace_specs(
    const m3_condition_config *config, uint64_t frames,
    uint64_t output_length, m3_runtime_tensor_spec *specs);
m3_status m3_condition_encode_core(
    m3_backend *backend, const m3_condition_config *config,
    const m3_condition_weights *weights, const m3_tensor_view *frames,
    m3_progress_callback progress, void *progress_context,
    m3_condition_output *output, m3_error *error);

#endif

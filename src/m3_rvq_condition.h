/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_RVQ_CONDITION_H
#define M3_RVQ_CONDITION_H

#include "m3_backend.h"
#include "m3_progress.h"
#include "m3_tensor.h"
#include "m3_weight_stage.h"

#include <stddef.h>
#include <stdint.h>

#define M3_RVQ_LAYER_CAPACITY 4U
#define M3_RVQ_CODEBOOK_COUNT 8U
#define M3_RVQ_RESIDUAL_COUNT 7U
#define M3_RVQ_CODEBOOK_SIZE 1024U

typedef struct {
    const m3_tensor_view *input_norm;
    const m3_tensor_view *query;
    const m3_tensor_view *key;
    const m3_tensor_view *value;
    const m3_tensor_view *attention_out;
    const m3_tensor_view *post_attention_norm;
    const m3_tensor_view *gate;
    const m3_tensor_view *up;
    const m3_tensor_view *down;
} m3_rvq_layer_weights;

typedef struct {
    const m3_tensor_view *audio_embeddings;
    const m3_tensor_view *projection;
    const m3_tensor_view *position_embeddings;
    const m3_tensor_view *norm;
    m3_rvq_layer_weights layers[M3_RVQ_LAYER_CAPACITY];
    const m3_tensor_view *heads[M3_RVQ_RESIDUAL_COUNT];
} m3_rvq_weights;

typedef struct {
    m3_storage *storage;
    m3_tensor_view conditioning;
    uint32_t codes[M3_RVQ_RESIDUAL_COUNT];
} m3_rvq_frame;

void m3_rvq_frame_init(m3_rvq_frame *frame);
void m3_rvq_frame_dispose(m3_rvq_frame *frame);

m3_status m3_rvq_weights_bind(const m3_weight_stage *stage,
                              m3_rvq_weights *weights, m3_error *error);
m3_status m3_rvq_decode_frame(
    m3_backend *backend, const m3_rvq_weights *weights,
    const m3_tensor_view *last_hidden,
    const m3_tensor_view *semantic_embedding, const float *uniforms,
    size_t uniform_count, m3_progress_callback progress,
    void *progress_context, m3_rvq_frame *frame, m3_error *error);

typedef struct {
    const m3_tensor_view *layer_weight_logits;
    const m3_tensor_view *layer_scale;
    const m3_tensor_view *projection;
    const m3_tensor_view *bias;
} m3_condition_weights;

typedef struct {
    m3_storage *storage;
    m3_tensor_view tensor;
} m3_condition_output;

void m3_condition_output_init(m3_condition_output *output);
void m3_condition_output_dispose(m3_condition_output *output);

m3_status m3_condition_weights_bind(
    const m3_weight_stage *stage, m3_condition_weights *weights,
    m3_error *error);
m3_status m3_condition_encode(
    m3_backend *backend, const m3_condition_weights *weights,
    const m3_tensor_view *frames, m3_progress_callback progress,
    void *progress_context, m3_condition_output *output, m3_error *error);

#endif

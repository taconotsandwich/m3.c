/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_FLOW_RUNTIME_H
#define M3_FLOW_RUNTIME_H

#include "m3_progress.h"
#include "m3_rng.h"
#include "m3_rvq_condition.h"
#include "m3_tensor.h"
#include "m3_weight_stage.h"

#include <stddef.h>
#include <stdint.h>

#define M3_FLOW_WEIGHT_COUNT 441U
#define M3_FLOW_LAYER_COUNT 36U
#define M3_FLOW_LATENT_CHANNELS 128U
#define M3_FLOW_CONDITION_DIMENSION 2048U
#define M3_FLOW_INNER_DIMENSION 2048U
#define M3_FLOW_ATTENTION_HEADS 32U
#define M3_FLOW_HEAD_DIMENSION 64U
#define M3_FLOW_FEED_FORWARD_DIMENSION 8192U
#define M3_FLOW_ROTARY_DIMENSION 32U
#define M3_FLOW_FOURIER_DIMENSION 256U
#define M3_FLOW_CHUNK_FRAMES 200U
#define M3_FLOW_CHUNK_HOP 100U
#define M3_FLOW_CARRY_LENGTH 172U
#define M3_FLOW_INFERENCE_STEPS 30U
#define M3_FLOW_MAX_FRAMES 9000U

typedef struct {
    const m3_tensor_view *norm1_scale;
    const m3_tensor_view *norm1_bias;
    const m3_tensor_view *norm2_scale;
    const m3_tensor_view *norm2_bias;
    const m3_tensor_view *query;
    const m3_tensor_view *key;
    const m3_tensor_view *value;
    const m3_tensor_view *attention_out;
    const m3_tensor_view *feed_forward_in;
    const m3_tensor_view *feed_forward_in_bias;
    const m3_tensor_view *feed_forward_out;
    const m3_tensor_view *feed_forward_out_bias;
} m3_flow_layer_weights;

typedef struct {
    const m3_tensor_view *time_projection;
    const m3_tensor_view *time_linear_in;
    const m3_tensor_view *time_linear_in_bias;
    const m3_tensor_view *time_linear_out;
    const m3_tensor_view *time_linear_out_bias;
    const m3_tensor_view *preprocess_convolution;
    const m3_tensor_view *input_projection;
    const m3_tensor_view *output_projection;
    const m3_tensor_view *postprocess_convolution;
    m3_flow_layer_weights layers[M3_FLOW_LAYER_COUNT];
} m3_flow_weights;

/* Each chunk is an owned contiguous F32 tensor [1,128,L]. Flow publishes
 * uncropped chunks; the vocoder owns overlap cropping and concatenation. */
typedef struct {
    m3_storage **storages;
    m3_tensor_view *chunks;
    size_t chunk_count;
} m3_flow_output;

void m3_flow_output_init(m3_flow_output *output);
void m3_flow_output_dispose(m3_flow_output *output);

m3_status m3_flow_weights_bind(const m3_weight_stage *stage,
                               m3_flow_weights *weights,
                               m3_error *error);

/* The staged flow and condition weights, backend, and frame input are
 * borrowed for the call. Frames are BF16 [1,F,8,4096] and may be strided.
 * On failure or cancellation, output and rng are unchanged. */
m3_status m3_flow_synthesize(
    m3_backend *backend, const m3_flow_weights *weights,
    const m3_condition_weights *condition_weights,
    const m3_tensor_view *frame_hiddens, m3_rng *rng,
    m3_progress_callback progress, void *progress_context,
    m3_flow_output *output, m3_error *error);

#endif

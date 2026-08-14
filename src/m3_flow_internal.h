/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_FLOW_INTERNAL_H
#define M3_FLOW_INTERNAL_H

#include "m3_flow_runtime.h"
#include "m3_runtime_workspace.h"
#include "m3_rvq_condition_internal.h"

#include <stdbool.h>

typedef struct {
    uint32_t latent_channels;
    uint32_t condition_dimension;
    uint32_t layer_count;
    uint32_t attention_heads;
    uint32_t head_dimension;
    uint32_t feed_forward_dimension;
    uint32_t rotary_dimension;
    uint32_t fourier_dimension;
    uint32_t chunk_frames;
    uint32_t chunk_hop;
    uint32_t carry_length;
    uint32_t inference_steps;
    uint32_t maximum_frames;
    float guidance_scale;
    float layer_norm_epsilon;
    float rotary_theta;
    m3_condition_config condition;
} m3_flow_config;

typedef struct {
    char names[M3_FLOW_WEIGHT_COUNT][128];
    m3_weight_requirement requirements[M3_FLOW_WEIGHT_COUNT];
    size_t count;
} m3_flow_requirement_set;

#define M3_FLOW_WORKSPACE_COUNT 32U

typedef enum {
    M3_FLOW_WS_CONCATENATED = 0,
    M3_FLOW_WS_PREPROCESSED,
    M3_FLOW_WS_PROJECTED,
    M3_FLOW_WS_FOURIER,
    M3_FLOW_WS_TIME_HIDDEN,
    M3_FLOW_WS_TIME_ACTIVATED,
    M3_FLOW_WS_TIME_EMBEDDING,
    M3_FLOW_WS_TIME_ONES,
    M3_FLOW_WS_HIDDEN,
    M3_FLOW_WS_HIDDEN_TEMP,
    M3_FLOW_WS_NORMALIZED,
    M3_FLOW_WS_QUERY,
    M3_FLOW_WS_KEY,
    M3_FLOW_WS_VALUE,
    M3_FLOW_WS_QUERY_ROTARY,
    M3_FLOW_WS_KEY_ROTARY,
    M3_FLOW_WS_ATTENTION,
    M3_FLOW_WS_REORDER,
    M3_FLOW_WS_FEED_FORWARD,
    M3_FLOW_WS_GATED,
    M3_FLOW_WS_COSINES,
    M3_FLOW_WS_SINES,
    M3_FLOW_WS_OUTPUT_SEQUENCE,
    M3_FLOW_WS_POSTPROCESSED,
    M3_FLOW_WS_VELOCITY,
    M3_FLOW_WS_LATENT,
    M3_FLOW_WS_NOISE_CHANNELS,
    M3_FLOW_WS_NOISE_PROMPT,
    M3_FLOW_WS_PREVIOUS_LATENT,
    M3_FLOW_WS_PREVIOUS_CONDITION,
    M3_FLOW_WS_SCALAR,
    M3_FLOW_WS_ARITHMETIC
} m3_flow_workspace_index;

typedef struct {
    m3_backend *backend;
    const m3_flow_config *config;
    const m3_flow_weights *weights;
    m3_runtime_workspace workspace;
    m3_command_executor executor;
    uint64_t maximum_length;
} m3_flow_run;

void m3_flow_config_init(m3_flow_config *config);
m3_status m3_flow_requirements(m3_flow_requirement_set *requirements,
                               m3_error *error);
m3_status m3_flow_validate(
    m3_backend *backend, const m3_flow_config *config,
    const m3_flow_weights *weights,
    const m3_condition_weights *condition_weights,
    const m3_tensor_view *frame_hiddens, const m3_rng *rng,
    const m3_flow_output *output, size_t *chunk_count,
    uint64_t *maximum_length, uint64_t *progress_total,
    m3_error *error);

m3_status m3_flow_chunk_count(const m3_flow_config *config,
                              uint64_t frame_count, size_t *count,
                              m3_error *error);
m3_status m3_flow_chunk_window(const m3_flow_config *config,
                               uint64_t frame_count, size_t index,
                               uint64_t *start, uint64_t *length,
                               m3_error *error);
m3_status m3_flow_carry_window(const m3_flow_config *config,
                               uint64_t latent_length, uint64_t *start,
                               uint64_t *length, m3_error *error);
float m3_flow_timestep(uint32_t step, uint32_t step_count);
float m3_flow_timestep_delta(uint32_t step, uint32_t step_count);
void m3_flow_blend_coefficients(float timestep, float *noise,
                                float *previous);

void m3_flow_workspace_specs(const m3_flow_config *config,
                             uint64_t maximum_length,
                             m3_runtime_tensor_spec *specs);
m3_status m3_flow_preflight(
    m3_backend *backend, const m3_flow_config *config,
    const m3_tensor_view *frame_hiddens, size_t chunk_count,
    uint64_t maximum_length, const m3_runtime_tensor_spec *flow_specs,
    m3_error *error);
m3_status m3_flow_view(m3_flow_run *run, size_t slot, m3_dtype dtype,
                       uint8_t rank, const uint64_t *shape,
                       m3_tensor_view *view, m3_error *error);
m3_status m3_flow_fill_f32(m3_tensor_view *view, float value,
                           m3_error *error);
m3_status m3_flow_zero_storage(m3_storage *storage, m3_error *error);

m3_status m3_flow_prepare_tables(m3_flow_run *run, uint64_t length,
                                 float timestep, m3_error *error);
m3_status m3_flow_transformer_layer(
    m3_flow_run *run, const m3_flow_layer_weights *weights,
    uint64_t length, m3_error *error);
m3_status m3_flow_forward(m3_flow_run *run,
                          const m3_tensor_view *latent,
                          const m3_tensor_view *condition,
                          uint64_t length, float timestep,
                          m3_tensor_view *velocity, m3_error *error);
m3_status m3_flow_guided_velocity(m3_flow_run *run,
                                  m3_tensor_view *velocity,
                                  uint64_t length, m3_error *error);
m3_status m3_flow_euler_step(m3_flow_run *run,
                             m3_tensor_view *latent,
                             const m3_tensor_view *velocity,
                             uint64_t length, float delta,
                             m3_error *error);
m3_status m3_flow_blend_overlap(
    m3_flow_run *run, m3_tensor_view *latent, uint64_t length,
    uint64_t overlap, float timestep, m3_error *error);
m3_status m3_flow_preserve_carry(
    m3_flow_run *run, const m3_tensor_view *latent,
    const m3_tensor_view *condition, uint64_t length,
    uint64_t *carry_length, m3_error *error);

m3_status m3_flow_synthesize_core(
    m3_backend *backend, const m3_flow_config *config,
    const m3_flow_weights *weights,
    const m3_condition_weights *condition_weights,
    const m3_tensor_view *frame_hiddens, m3_rng *rng,
    m3_progress_callback progress, void *progress_context,
    m3_flow_output *output, m3_error *error);

#endif

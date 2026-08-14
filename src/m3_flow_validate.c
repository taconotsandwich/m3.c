/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_flow_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static m3_status m3_flow_check_view(
    m3_backend *backend, const m3_tensor_view *view, uint8_t rank,
    const uint64_t *shape, const char *name, m3_error *error)
{
    const void *data = NULL;
    uint8_t axis;
    m3_status status;

    if (backend == NULL || view == NULL || view->storage == NULL ||
        shape == NULL || name == NULL ||
        m3_storage_backend(view->storage) != backend) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s is not on the flow backend", name);
    }
    status = m3_tensor_const_data(view, &data, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (view->metadata.dtype != M3_DTYPE_F32 ||
        view->metadata.rank != rank) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s has the wrong dtype or rank", name);
    }
    for (axis = 0U; axis < rank; ++axis) {
        if (view->metadata.shape[axis] != shape[axis]) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "%s has the wrong shape", name);
        }
    }
    (void)data;
    return M3_STATUS_OK;
}

static m3_status m3_flow_validate_config(const m3_flow_config *config,
                                         uint64_t *inner,
                                         uint64_t *concatenated,
                                         m3_error *error)
{
    uint64_t computed_inner;
    uint64_t computed_concat;

    if (config == NULL || inner == NULL || concatenated == NULL ||
        config->latent_channels == 0U ||
        config->condition_dimension == 0U || config->layer_count == 0U ||
        config->layer_count > M3_FLOW_LAYER_COUNT ||
        config->attention_heads == 0U || config->head_dimension == 0U ||
        config->feed_forward_dimension == 0U ||
        config->rotary_dimension == 0U ||
        (config->rotary_dimension & 1U) != 0U ||
        config->rotary_dimension > config->head_dimension ||
        config->fourier_dimension == 0U ||
        (config->fourier_dimension & 1U) != 0U ||
        config->chunk_frames == 0U || config->chunk_hop == 0U ||
        config->chunk_hop >= config->chunk_frames ||
        config->carry_length == 0U || config->inference_steps == 0U ||
        config->maximum_frames == 0U ||
        !isfinite(config->guidance_scale) ||
        !isfinite(config->layer_norm_epsilon) ||
        config->layer_norm_epsilon <= 0.0F ||
        !isfinite(config->rotary_theta) || config->rotary_theta <= 0.0F ||
        config->condition.hidden_size == 0U ||
        config->condition.layer_count == 0U ||
        config->condition.layer_count > 8U ||
        config->condition.output_size != config->condition_dimension ||
        config->condition.resize_numerator == 0U ||
        config->condition.resize_denominator == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow runtime configuration is invalid");
    }
    computed_inner = (uint64_t)config->attention_heads *
                     config->head_dimension;
    computed_concat = (uint64_t)config->latent_channels * 2U;
    if (computed_inner > UINT32_MAX ||
        config->condition_dimension > UINT64_MAX - computed_concat) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "flow runtime dimensions overflow");
    }
    computed_concat += config->condition_dimension;
    *inner = computed_inner;
    *concatenated = computed_concat;
    return M3_STATUS_OK;
}

static m3_status m3_flow_validate_roots(
    m3_backend *backend, const m3_flow_config *config,
    const m3_flow_weights *weights, uint64_t inner,
    uint64_t concatenated, m3_error *error)
{
    const uint64_t time_projection[] = {
        config->fourier_dimension / 2U, 1U
    };
    const uint64_t time_in[] = {inner, config->fourier_dimension};
    const uint64_t inner_vector[] = {inner};
    const uint64_t time_out[] = {inner, inner};
    const uint64_t preprocess[] = {concatenated, concatenated, 1U};
    const uint64_t input_projection[] = {inner, concatenated};
    const uint64_t output_projection[] = {
        config->latent_channels, inner
    };
    const uint64_t postprocess[] = {
        config->latent_channels, config->latent_channels, 1U
    };
    m3_status status;

    if (weights == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow weights are required");
    }
    status = m3_flow_check_view(
        backend, weights->time_projection, 2U, time_projection,
        "flow time projection", error);
#define M3_FLOW_ROOT(member, rank, shape, label)                            \
    if (status == M3_STATUS_OK) {                                           \
        status = m3_flow_check_view(backend, weights->member, rank, shape,  \
                                    label, error);                           \
    }
    M3_FLOW_ROOT(time_linear_in, 2U, time_in, "flow time input linear")
    M3_FLOW_ROOT(time_linear_in_bias, 1U, inner_vector,
                 "flow time input bias")
    M3_FLOW_ROOT(time_linear_out, 2U, time_out,
                 "flow time output linear")
    M3_FLOW_ROOT(time_linear_out_bias, 1U, inner_vector,
                 "flow time output bias")
    M3_FLOW_ROOT(preprocess_convolution, 3U, preprocess,
                 "flow preprocess convolution")
    M3_FLOW_ROOT(input_projection, 2U, input_projection,
                 "flow input projection")
    M3_FLOW_ROOT(output_projection, 2U, output_projection,
                 "flow output projection")
    M3_FLOW_ROOT(postprocess_convolution, 3U, postprocess,
                 "flow postprocess convolution")
#undef M3_FLOW_ROOT
    return status;
}

static m3_status m3_flow_validate_layer(
    m3_backend *backend, const m3_flow_config *config,
    const m3_flow_layer_weights *weights, uint64_t inner, size_t index,
    m3_error *error)
{
    const uint64_t vector[] = {inner};
    const uint64_t square[] = {inner, inner};
    const uint64_t feed_forward_in[] = {
        (uint64_t)config->feed_forward_dimension * 2U, inner
    };
    const uint64_t feed_forward_in_bias[] = {
        (uint64_t)config->feed_forward_dimension * 2U
    };
    const uint64_t feed_forward_out[] = {
        inner, config->feed_forward_dimension
    };
    char name[96];
    m3_status status;

#define M3_FLOW_LAYER(member, rank, shape, suffix)                          \
    (void)snprintf(name, sizeof(name), "flow layer %zu %s", index, suffix); \
    status = m3_flow_check_view(backend, weights->member, rank, shape, name, \
                                error);                                      \
    if (status != M3_STATUS_OK) {                                            \
        return status;                                                       \
    }
    M3_FLOW_LAYER(norm1_scale, 1U, vector, "norm1 scale")
    M3_FLOW_LAYER(norm1_bias, 1U, vector, "norm1 bias")
    M3_FLOW_LAYER(norm2_scale, 1U, vector, "norm2 scale")
    M3_FLOW_LAYER(norm2_bias, 1U, vector, "norm2 bias")
    M3_FLOW_LAYER(query, 2U, square, "query")
    M3_FLOW_LAYER(key, 2U, square, "key")
    M3_FLOW_LAYER(value, 2U, square, "value")
    M3_FLOW_LAYER(attention_out, 2U, square, "attention output")
    M3_FLOW_LAYER(feed_forward_in, 2U, feed_forward_in,
                  "feed-forward input")
    M3_FLOW_LAYER(feed_forward_in_bias, 1U, feed_forward_in_bias,
                  "feed-forward input bias")
    M3_FLOW_LAYER(feed_forward_out, 2U, feed_forward_out,
                  "feed-forward output")
    M3_FLOW_LAYER(feed_forward_out_bias, 1U, vector,
                  "feed-forward output bias")
#undef M3_FLOW_LAYER
    return M3_STATUS_OK;
}

static bool m3_flow_output_valid(const m3_flow_output *output)
{
    return output != NULL &&
           ((output->chunk_count == 0U && output->storages == NULL &&
             output->chunks == NULL) ||
            (output->chunk_count != 0U && output->storages != NULL &&
             output->chunks != NULL));
}

static m3_status m3_flow_lengths(
    const m3_flow_config *config, uint64_t frames, size_t *chunk_count,
    uint64_t *maximum_length, uint64_t *progress_total, m3_error *error)
{
    size_t count = 0U;
    size_t index;
    uint64_t maximum = 0U;
    m3_status status = m3_flow_chunk_count(
        config, frames, &count, error);

    for (index = 0U; index < count && status == M3_STATUS_OK; ++index) {
        uint64_t start = 0U;
        uint64_t frame_length = 0U;
        uint64_t latent_length = 0U;

        status = m3_flow_chunk_window(
            config, frames, index, &start, &frame_length, error);
        if (status == M3_STATUS_OK) {
            status = m3_condition_output_length(
                frame_length, config->condition.resize_numerator,
                config->condition.resize_denominator, &latent_length,
                error);
        }
        if (status == M3_STATUS_OK && latent_length > maximum) {
            maximum = latent_length;
        }
        (void)start;
    }
    if (status == M3_STATUS_OK &&
        (uint64_t)count > UINT64_MAX / config->inference_steps) {
        status = m3_error_set(error, M3_STATUS_OVERFLOW,
                              "flow progress total overflows");
    }
    if (status == M3_STATUS_OK) {
        *chunk_count = count;
        *maximum_length = maximum;
        *progress_total = (uint64_t)count * config->inference_steps;
    }
    return status;
}

m3_status m3_flow_validate(
    m3_backend *backend, const m3_flow_config *config,
    const m3_flow_weights *weights,
    const m3_condition_weights *condition_weights,
    const m3_tensor_view *frame_hiddens, const m3_rng *rng,
    const m3_flow_output *output, size_t *chunk_count,
    uint64_t *maximum_length, uint64_t *progress_total,
    m3_error *error)
{
    m3_condition_output unused_output;
    uint64_t unused_length = 0U;
    size_t unused_bytes = 0U;
    uint64_t inner = 0U;
    uint64_t concatenated = 0U;
    size_t layer;
    m3_status status;

    if (backend == NULL || frame_hiddens == NULL || rng == NULL ||
        chunk_count == NULL || maximum_length == NULL ||
        progress_total == NULL || !m3_flow_output_valid(output) ||
        (rng->increment & UINT64_C(1)) == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow synthesis arguments are invalid");
    }
    status = m3_flow_validate_config(
        config, &inner, &concatenated, error);
    if (status == M3_STATUS_OK) {
        status = m3_flow_validate_roots(
            backend, config, weights, inner, concatenated, error);
    }
    for (layer = 0U; status == M3_STATUS_OK &&
                    layer < config->layer_count; ++layer) {
        status = m3_flow_validate_layer(
            backend, config, &weights->layers[layer], inner, layer,
            error);
    }
    m3_condition_output_init(&unused_output);
    if (status == M3_STATUS_OK) {
        status = m3_condition_validate(
            backend, &config->condition, condition_weights,
            frame_hiddens, &unused_output, &unused_length, &unused_bytes,
            error);
    }
    if (status == M3_STATUS_OK &&
        frame_hiddens->metadata.shape[1] > config->maximum_frames) {
        status = m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                              "flow frame count exceeds capacity");
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_lengths(
            config, frame_hiddens->metadata.shape[1], chunk_count,
            maximum_length, progress_total, error);
    }
    if (status == M3_STATUS_OK) {
        m3_error_reset(error);
    }
    return status;
}

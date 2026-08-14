/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_rvq_condition_internal.h"

#include <stdint.h>
#include <string.h>

m3_status m3_condition_output_length(
    uint64_t frames, uint32_t numerator, uint32_t denominator,
    uint64_t *length, m3_error *error)
{
    uint64_t scaled;

    if (frames == 0U || numerator == 0U || denominator == 0U ||
        length == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "condition resize ratio is invalid");
    }
    if (frames > UINT64_MAX / numerator) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "condition output length overflows");
    }
    scaled = frames * numerator;
    *length = scaled / denominator;
    if (*length == 0U) {
        *length = 1U;
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

static m3_status m3_condition_validate_config(
    const m3_condition_config *config, m3_error *error)
{
    if (config == NULL || config->hidden_size == 0U ||
        config->layer_count == 0U || config->layer_count > 8U ||
        config->output_size == 0U || config->resize_numerator == 0U ||
        config->resize_denominator == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "condition runtime configuration is invalid");
    }
    return M3_STATUS_OK;
}

static m3_status m3_condition_validate_weights(
    m3_backend *backend, const m3_condition_config *config,
    const m3_condition_weights *weights, m3_error *error)
{
    const uint64_t layers[] = {config->layer_count};
    const uint64_t scale[] = {1U};
    const uint64_t projection[] = {
        config->output_size, config->hidden_size, 3U
    };
    const uint64_t bias[] = {config->output_size};
    m3_status status;

    if (weights == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "condition weights are required");
    }
    status = m3_rvq_condition_check_view(
        backend, weights->layer_weight_logits, M3_DTYPE_F32, 1U, layers,
        "condition layer logits", error);
    if (status == M3_STATUS_OK) {
        status = m3_rvq_condition_check_view(
            backend, weights->layer_scale, M3_DTYPE_F32, 1U, scale,
            "condition layer scale", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_condition_check_view(
            backend, weights->projection, M3_DTYPE_F32, 3U, projection,
            "condition projection", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_condition_check_view(
            backend, weights->bias, M3_DTYPE_F32, 1U, bias,
            "condition bias", error);
    }
    return status;
}

m3_status m3_condition_validate(
    m3_backend *backend, const m3_condition_config *config,
    const m3_condition_weights *weights, const m3_tensor_view *frames,
    const m3_condition_output *output, uint64_t *output_length,
    size_t *output_bytes, m3_error *error)
{
    m3_tensor_metadata output_metadata;
    uint64_t input_shape[4];
    uint64_t output_shape[3];
    m3_status status;

    if (backend == NULL || config == NULL || frames == NULL ||
        output == NULL ||
        output_length == NULL || output_bytes == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "condition runtime arguments are invalid");
    }
    status = m3_condition_validate_config(config, error);
    if (status == M3_STATUS_OK &&
        (frames->metadata.dtype != M3_DTYPE_BF16 ||
         frames->metadata.rank != 4U ||
         frames->metadata.shape[0] != 1U ||
         frames->metadata.shape[1] == 0U ||
         frames->metadata.shape[2] != config->layer_count ||
         frames->metadata.shape[3] != config->hidden_size)) {
        status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                              "condition frames must be [1,F,layers,H]");
    }
    if (status == M3_STATUS_OK) {
        input_shape[0] = 1U;
        input_shape[1] = frames->metadata.shape[1];
        input_shape[2] = config->layer_count;
        input_shape[3] = config->hidden_size;
        status = m3_rvq_condition_check_view(
            backend, frames, frames->metadata.dtype, 4U, input_shape,
            "condition frames", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_condition_validate_weights(
            backend, config, weights, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_condition_output_length(
            frames->metadata.shape[1], config->resize_numerator,
            config->resize_denominator, output_length, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    output_shape[0] = 1U;
    output_shape[1] = *output_length;
    output_shape[2] = config->output_size;
    status = m3_tensor_metadata_init(&output_metadata, M3_DTYPE_F32, 3U,
                                     output_shape, error);
    if (status == M3_STATUS_OK) {
        *output_bytes = output_metadata.byte_count;
        m3_error_reset(error);
    }
    return status;
}

static void m3_condition_spec(m3_runtime_tensor_spec *spec,
                              uint8_t rank, const uint64_t *shape)
{
    uint8_t axis;

    (void)memset(spec, 0, sizeof(*spec));
    spec->dtype = M3_DTYPE_F32;
    spec->rank = rank;
    spec->alignment = 64U;
    for (axis = 0U; axis < rank; ++axis) {
        spec->shape[axis] = shape[axis];
    }
}

void m3_condition_workspace_specs(
    const m3_condition_config *config, uint64_t frames,
    uint64_t output_length, m3_runtime_tensor_spec *specs)
{
    const uint64_t layer_weights[] = {config->layer_count};
    const uint64_t layer[] = {1U, 1U, config->hidden_size, frames};
    const uint64_t mix[] = {1U, config->hidden_size, frames};
    const uint64_t convolution[] = {1U, config->output_size, frames};
    const uint64_t resized[] = {
        1U, config->output_size, output_length
    };

    m3_condition_spec(&specs[M3_CONDITION_WS_LAYER_WEIGHTS], 1U,
                      layer_weights);
    m3_condition_spec(&specs[M3_CONDITION_WS_CAST_LAYER], 4U, layer);
    m3_condition_spec(&specs[M3_CONDITION_WS_WEIGHTED_LAYER], 4U, layer);
    m3_condition_spec(&specs[M3_CONDITION_WS_MIX], 3U, mix);
    m3_condition_spec(&specs[M3_CONDITION_WS_CONVOLUTION], 3U,
                      convolution);
    m3_condition_spec(&specs[M3_CONDITION_WS_RESIZED], 3U, resized);
}

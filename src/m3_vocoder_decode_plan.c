/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_vocoder_internal.h"

#include <string.h>

void m3_vocoder_output_init(m3_vocoder_output *output)
{
    if (output != NULL) {
        (void)memset(output, 0, sizeof(*output));
    }
}

void m3_vocoder_output_dispose(m3_vocoder_output *output)
{
    if (output == NULL) {
        return;
    }
    m3_storage_free(output->storage);
    m3_vocoder_output_init(output);
}

static m3_status m3_vocoder_decode_config_validate(
    const m3_vocoder_plan_config *config, m3_error *error)
{
    size_t block;
    m3_status status = m3_vocoder_plan_config_validate(config, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (config->block_count != M3_VOCODER_BLOCK_COUNT ||
        config->residual_count != M3_VOCODER_RESIDUAL_COUNT) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "vocoder decoder topology is invalid");
    }
    for (block = 0U; block < M3_VOCODER_BLOCK_COUNT; ++block) {
        if ((config->strides[block] & 1U) != 0U) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "vocoder decoder block is invalid");
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_vocoder_decode_record(
    m3_vocoder_decode_measurement *measurement, size_t slot,
    uint64_t channels, uint64_t length, m3_error *error)
{
    const uint64_t shape[] = {2U, channels, length};
    m3_tensor_metadata metadata;
    m3_status status = m3_tensor_metadata_init(
        &metadata, M3_DTYPE_F32, 3U, shape, error);

    if (status == M3_STATUS_OK &&
        metadata.byte_count > measurement->buffer_bytes[slot]) {
        measurement->buffer_bytes[slot] = metadata.byte_count;
    }
    return status;
}

static m3_status m3_vocoder_decode_multiply_length(
    uint64_t *length, uint64_t stride, m3_error *error)
{
    if (*length > UINT64_MAX / stride) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "vocoder waveform length overflows");
    }
    *length *= stride;
    return M3_STATUS_OK;
}

m3_status m3_vocoder_decode_measure(
    const m3_vocoder_plan_config *config, uint64_t latent_length,
    m3_vocoder_decode_measurement *measurement, m3_error *error)
{
    m3_vocoder_decode_measurement built;
    uint64_t channels;
    uint64_t length = latent_length;
    size_t hidden = 1U;
    size_t block;
    m3_status status;

    if (measurement == NULL || config == NULL || latent_length == 0U ||
        latent_length > config->maximum_latent_length) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "vocoder latent length is invalid");
    }
    status = m3_vocoder_decode_config_validate(config, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    (void)memset(&built, 0, sizeof(built));
    status = m3_vocoder_decode_record(
        &built, 0U, config->decoder_output_channels, length, error);
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_decode_record(
            &built, 1U, config->initial_channels, length, error);
    }
    channels = config->initial_channels;
    for (block = 0U; block < M3_VOCODER_BLOCK_COUNT &&
                         status == M3_STATUS_OK; ++block) {
        size_t first = (hidden + 1U) % M3_VOCODER_DECODE_BUFFER_COUNT;
        size_t second = (hidden + 2U) % M3_VOCODER_DECODE_BUFFER_COUNT;
        size_t residual;

        status = m3_vocoder_decode_record(
            &built, first, channels, length, error);
        if (status == M3_STATUS_OK) {
            status = m3_vocoder_decode_multiply_length(
                &length, config->strides[block], error);
        }
        channels /= 2U;
        if (status == M3_STATUS_OK) {
            status = m3_vocoder_decode_record(
                &built, second, channels, length, error);
        }
        hidden = second;
        for (residual = 0U; residual < M3_VOCODER_RESIDUAL_COUNT &&
                            status == M3_STATUS_OK; ++residual) {
            first = (hidden + 1U) % M3_VOCODER_DECODE_BUFFER_COUNT;
            second = (hidden + 2U) % M3_VOCODER_DECODE_BUFFER_COUNT;
            status = m3_vocoder_decode_record(
                &built, first, channels, length, error);
            if (status == M3_STATUS_OK) {
                status = m3_vocoder_decode_record(
                    &built, second, channels, length, error);
            }
            hidden = second;
        }
    }
    if (status == M3_STATUS_OK) {
        size_t final = (hidden + 1U) % M3_VOCODER_DECODE_BUFFER_COUNT;
        const uint64_t output_shape[] = {1U, 2U, length};
        m3_tensor_metadata output_metadata;
        size_t slot;

        status = m3_vocoder_decode_record(
            &built, final, channels, length, error);
        if (status == M3_STATUS_OK) {
            status = m3_tensor_metadata_init(
                &output_metadata, M3_DTYPE_F32, 3U, output_shape, error);
        }
        if (status == M3_STATUS_OK) {
            built.output_bytes = output_metadata.byte_count;
            built.output_length = length;
        }
        for (slot = 0U; slot < M3_VOCODER_DECODE_BUFFER_COUNT &&
                        status == M3_STATUS_OK; ++slot) {
            if (built.buffer_bytes[slot] >
                SIZE_MAX - built.workspace_bytes) {
                status = m3_error_set(
                    error, M3_STATUS_OVERFLOW,
                    "vocoder workspace byte count overflows");
            } else {
                built.workspace_bytes += built.buffer_bytes[slot];
            }
        }
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    *measurement = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

static m3_status m3_vocoder_decode_plan_add(
    const m3_backend_info *info, size_t bytes, uint64_t *planned,
    m3_error *error)
{
    uint64_t amount = (uint64_t)bytes;

    if ((size_t)amount != bytes) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "vocoder allocation size overflows");
    }
    if (amount > info->maximum_storage_bytes) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "vocoder tensor exceeds backend storage limit");
    }
    if (amount > UINT64_MAX - *planned) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "vocoder allocation plan overflows");
    }
    *planned += amount;
    return M3_STATUS_OK;
}

static m3_status m3_vocoder_decode_preflight(
    m3_backend *backend, const m3_vocoder_decode_measurement *measurement,
    m3_error *error)
{
    m3_backend_allocation_stats stats;
    m3_backend_info info;
    uint64_t planned;
    size_t slot;
    m3_status status = m3_backend_get_info(backend, &info, error);

    if (status == M3_STATUS_OK) {
        status = m3_backend_get_allocation_stats(backend, &stats, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    planned = (uint64_t)stats.live_allocated_bytes;
    if ((size_t)planned != stats.live_allocated_bytes) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "vocoder live allocation count overflows");
    }
    for (slot = 0U; slot < M3_VOCODER_DECODE_BUFFER_COUNT &&
                    status == M3_STATUS_OK; ++slot) {
        status = m3_vocoder_decode_plan_add(
            &info, measurement->buffer_bytes[slot], &planned, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_decode_plan_add(
            &info, measurement->output_bytes, &planned, error);
    }
    if (status == M3_STATUS_OK &&
        info.recommended_working_set_bytes != 0U &&
        planned > info.recommended_working_set_bytes) {
        status = m3_error_set(
            error, M3_STATUS_OUT_OF_MEMORY,
            "vocoder decode exceeds backend recommended working set");
    }
    return status;
}

m3_status m3_vocoder_decode_prepare(
    m3_vocoder_runtime *runtime, const m3_tensor_view *latents,
    m3_vocoder_output *output,
    m3_vocoder_decode_measurement *measurement, m3_error *error)
{
    const void *latent_data = NULL;
    m3_status status;

    if (runtime == NULL || runtime->backend == NULL || latents == NULL ||
        output == NULL || latents->storage == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "vocoder decode arguments are invalid");
    }
    status = m3_tensor_const_data(latents, &latent_data, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (latents->metadata.dtype != M3_DTYPE_F32 ||
        latents->metadata.rank != 3U ||
        latents->metadata.shape[0] != 1U ||
        !m3_tensor_is_contiguous(latents) ||
        m3_storage_backend(latents->storage) != runtime->backend ||
        runtime->bound.block_count != M3_VOCODER_BLOCK_COUNT ||
        runtime->bound.residual_count != M3_VOCODER_RESIDUAL_COUNT) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "vocoder decode arguments are invalid");
    }
    if (latents->metadata.shape[1] != runtime->config.latent_channels) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "vocoder latent channel shape is invalid");
    }
    status = m3_vocoder_decode_measure(
        &runtime->config, latents->metadata.shape[2], measurement, error);
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_decode_preflight(
            runtime->backend, measurement, error);
    }
    return status;
}

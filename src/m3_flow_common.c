/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_flow_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void m3_flow_config_init(m3_flow_config *config)
{
    if (config == NULL) {
        return;
    }
    *config = (m3_flow_config){
        M3_FLOW_LATENT_CHANNELS,
        M3_FLOW_CONDITION_DIMENSION,
        M3_FLOW_LAYER_COUNT,
        M3_FLOW_ATTENTION_HEADS,
        M3_FLOW_HEAD_DIMENSION,
        M3_FLOW_FEED_FORWARD_DIMENSION,
        M3_FLOW_ROTARY_DIMENSION,
        M3_FLOW_FOURIER_DIMENSION,
        M3_FLOW_CHUNK_FRAMES,
        M3_FLOW_CHUNK_HOP,
        M3_FLOW_CARRY_LENGTH,
        M3_FLOW_INFERENCE_STEPS,
        M3_FLOW_MAX_FRAMES,
        1.7F,
        1.0e-5F,
        10000.0F,
        {4096U, 8U, M3_FLOW_CONDITION_DIMENSION, 441U, 128U}
    };
}

void m3_flow_output_init(m3_flow_output *output)
{
    if (output != NULL) {
        (void)memset(output, 0, sizeof(*output));
    }
}

void m3_flow_output_dispose(m3_flow_output *output)
{
    size_t index;

    if (output == NULL) {
        return;
    }
    if (output->storages != NULL) {
        for (index = 0U; index < output->chunk_count; ++index) {
            size_t later;

            for (later = index + 1U; later < output->chunk_count;
                 ++later) {
                if (output->storages[later] == output->storages[index]) {
                    output->storages[later] = NULL;
                }
            }
            m3_storage_free(output->storages[index]);
        }
    }
    free(output->storages);
    free(output->chunks);
    m3_flow_output_init(output);
}

m3_status m3_flow_chunk_count(const m3_flow_config *config,
                              uint64_t frame_count, size_t *count,
                              m3_error *error)
{
    uint64_t extra;
    uint64_t chunks;

    if (config == NULL || count == NULL || frame_count == 0U ||
        config->chunk_frames == 0U || config->chunk_hop == 0U ||
        config->chunk_hop >= config->chunk_frames) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow chunk configuration is invalid");
    }
    if (frame_count <= config->chunk_frames) {
        *count = 1U;
        m3_error_reset(error);
        return M3_STATUS_OK;
    }
    extra = frame_count - config->chunk_frames;
    chunks = 1U + extra / config->chunk_hop;
    if (extra % config->chunk_hop != 0U) {
        ++chunks;
    }
    if (chunks > SIZE_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "flow chunk count overflows size_t");
    }
    *count = (size_t)chunks;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_flow_chunk_window(const m3_flow_config *config,
                               uint64_t frame_count, size_t index,
                               uint64_t *start, uint64_t *length,
                               m3_error *error)
{
    size_t count = 0U;
    uint64_t begin;
    uint64_t remaining;
    m3_status status = m3_flow_chunk_count(
        config, frame_count, &count, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (start == NULL || length == NULL || index >= count ||
        (uint64_t)index > UINT64_MAX / config->chunk_hop) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow chunk window is invalid");
    }
    begin = (uint64_t)index * config->chunk_hop;
    if (begin >= frame_count) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "flow chunk begins beyond the input");
    }
    remaining = frame_count - begin;
    *start = begin;
    *length = remaining < config->chunk_frames
                  ? remaining
                  : config->chunk_frames;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_flow_carry_window(const m3_flow_config *config,
                               uint64_t latent_length, uint64_t *start,
                               uint64_t *length, m3_error *error)
{
    uint64_t twice;
    uint64_t begin;
    uint64_t end;

    if (config == NULL || start == NULL || length == NULL ||
        config->carry_length == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow carry configuration is invalid");
    }
    twice = (uint64_t)config->carry_length * 2U;
    begin = latent_length > twice ? latent_length - twice : 0U;
    end = latent_length > config->carry_length
              ? latent_length - config->carry_length
              : 0U;
    if (end < begin) {
        end = begin;
    }
    *start = begin;
    *length = end - begin;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

float m3_flow_timestep(uint32_t step, uint32_t step_count)
{
    double sigma;
    float sigma_f32;
    volatile float timestep;

    if (step_count == 0U || step >= step_count) {
        return 0.0F;
    }
    sigma = 1.0 - (double)step / (double)step_count;
    sigma_f32 = (float)sigma;
    timestep = 1.0F - sigma_f32;
    return timestep;
}

float m3_flow_timestep_delta(uint32_t step, uint32_t step_count)
{
    float current;
    float next;
    volatile float delta;

    if (step_count == 0U || step >= step_count) {
        return 0.0F;
    }
    current = m3_flow_timestep(step, step_count);
    next = step + 1U == step_count
               ? 1.0F
               : m3_flow_timestep(step + 1U, step_count);
    delta = next - current;
    return delta;
}

void m3_flow_blend_coefficients(float timestep, float *noise,
                                float *previous)
{
    volatile float product = 0.999999F * timestep;

    if (noise != NULL) {
        *noise = 1.0F - product;
    }
    if (previous != NULL) {
        *previous = timestep;
    }
}

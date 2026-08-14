/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_flow_internal.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static m3_status m3_flow_fourier(m3_flow_run *run, float timestep,
                                 m3_error *error)
{
    const uint32_t dimension = run->config->fourier_dimension;
    const uint32_t half = dimension / 2U;
    const uint64_t shape[] = {2U, dimension};
    const unsigned char *weight_data = NULL;
    m3_tensor_view fourier;
    float *output = NULL;
    const float two_pi =
        (float)6.2831853071795864769252867665590057683943387987502;
    volatile float scaled_timestep = two_pi * timestep;
    uint32_t index;
    m3_status status = m3_flow_view(
        run, M3_FLOW_WS_FOURIER, M3_DTYPE_F32, 2U, shape, &fourier,
        error);

    if (status == M3_STATUS_OK) {
        status = m3_tensor_const_data(
            run->weights->time_projection,
            (const void **)&weight_data, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_data(&fourier, (void **)&output, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (weight_data == NULL || output == NULL) {
        return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                            "flow Fourier storage is not host visible");
    }
    for (index = 0U; index < half; ++index) {
        float weight;
        volatile float angle;
        size_t offset = (size_t)index *
                        run->weights->time_projection->byte_strides[0];

        (void)memcpy(&weight, weight_data + offset, sizeof(weight));
        angle = scaled_timestep * weight;
        output[index] = cosf(angle);
        output[half + index] = sinf(angle);
        output[dimension + index] = output[index];
        output[dimension + half + index] = output[half + index];
    }
    if (status == M3_STATUS_OK) {
        m3_error_reset(error);
    }
    return status;
}

static m3_status m3_flow_rotary(m3_flow_run *run, uint64_t length,
                                m3_error *error)
{
    const uint64_t sequence = length + 1U;
    const uint32_t half = run->config->rotary_dimension / 2U;
    const uint64_t shape[] = {sequence, half};
    m3_tensor_view cosines;
    m3_tensor_view sines;
    float *cosine_data = NULL;
    float *sine_data = NULL;
    uint64_t position;
    m3_status status = m3_flow_view(
        run, M3_FLOW_WS_COSINES, M3_DTYPE_F32, 2U, shape, &cosines,
        error);

    if (status == M3_STATUS_OK) {
        status = m3_flow_view(
            run, M3_FLOW_WS_SINES, M3_DTYPE_F32, 2U, shape, &sines,
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_data(&cosines, (void **)&cosine_data, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_data(&sines, (void **)&sine_data, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (cosine_data == NULL || sine_data == NULL) {
        return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                            "flow rotary storage is not host visible");
    }
    for (position = 0U; position < sequence; ++position) {
        uint32_t pair;

        for (pair = 0U; pair < half; ++pair) {
            volatile float exponent =
                (float)(pair * 2U) /
                (float)run->config->rotary_dimension;
            volatile float divisor =
                powf(run->config->rotary_theta, exponent);
            volatile float inverse = 1.0F / divisor;
            volatile float frequency = (float)position * inverse;
            size_t offset = (size_t)position * half + pair;

            cosine_data[offset] = cosf(frequency);
            sine_data[offset] = sinf(frequency);
        }
    }
    if (status == M3_STATUS_OK) {
        m3_error_reset(error);
    }
    return status;
}

m3_status m3_flow_prepare_tables(m3_flow_run *run, uint64_t length,
                                 float timestep, m3_error *error)
{
    m3_status status;

    if (run == NULL || run->config == NULL || run->weights == NULL ||
        length == 0U || length > run->maximum_length ||
        !isfinite(timestep) || timestep < 0.0F || timestep > 1.0F) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow table request is invalid");
    }
    status = m3_flow_fourier(run, timestep, error);
    if (status == M3_STATUS_OK) {
        status = m3_flow_rotary(run, length, error);
    }
    return status;
}

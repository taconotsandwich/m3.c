/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_flow_internal.h"

#include <string.h>

static m3_status m3_flow_math_view(
    m3_flow_run *run, size_t slot, uint64_t length,
    m3_tensor_view *view, m3_error *error)
{
    const uint64_t shape[] = {
        1U, length, run->config->latent_channels
    };

    return m3_flow_view(
        run, slot, M3_DTYPE_F32, 3U, shape, view, error);
}

static m3_status m3_flow_math_execute(
    m3_flow_run *run, m3_op_kind kind,
    const m3_tensor_view *left, const m3_tensor_view *right,
    m3_tensor_view *output, m3_error *error)
{
    m3_command command = {0};

    command.kind = kind;
    if (kind == M3_OP_ADD) {
        command.descriptor.add = (m3_op_binary){left, right, output};
    } else {
        command.descriptor.mul = (m3_op_binary){left, right, output};
    }
    return m3_command_executor_execute(
        &run->executor, &command, 1U, error);
}

m3_status m3_flow_guided_velocity(m3_flow_run *run,
                                  m3_tensor_view *velocity,
                                  uint64_t length, m3_error *error)
{
    m3_tensor_view conditional;
    m3_tensor_view unconditional;
    m3_tensor_view scalar;
    m3_tensor_view arithmetic;
    m3_status status;

    if (run == NULL || velocity == NULL ||
        velocity->metadata.rank != 3U ||
        velocity->metadata.shape[0] != 2U ||
        velocity->metadata.shape[1] != length ||
        velocity->metadata.shape[2] != run->config->latent_channels) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow CFG velocity is invalid");
    }
    m3_tensor_view_init(&conditional);
    m3_tensor_view_init(&unconditional);
    status = m3_tensor_slice(
        velocity, 0U, 0U, 1U, &conditional, error);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_slice(
            velocity, 0U, 1U, 1U, &unconditional, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_view(
            run, M3_FLOW_WS_SCALAR, length, &scalar, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_view(
            run, M3_FLOW_WS_ARITHMETIC, length, &arithmetic, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_fill_f32(&scalar, -1.0F, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_execute(
            run, M3_OP_MUL, &unconditional, &scalar, &arithmetic, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_execute(
            run, M3_OP_ADD, &conditional, &arithmetic, &arithmetic,
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_fill_f32(
            &scalar, run->config->guidance_scale, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_execute(
            run, M3_OP_MUL, &arithmetic, &scalar, &arithmetic, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_execute(
            run, M3_OP_ADD, &unconditional, &arithmetic, &arithmetic,
            error);
    }
    if (status == M3_STATUS_OK) {
        *velocity = arithmetic;
        m3_error_reset(error);
    }
    return status;
}

m3_status m3_flow_euler_step(m3_flow_run *run,
                             m3_tensor_view *latent,
                             const m3_tensor_view *velocity,
                             uint64_t length, float delta,
                             m3_error *error)
{
    m3_tensor_view scalar;
    m3_tensor_view arithmetic;
    m3_status status;

    if (run == NULL || latent == NULL || velocity == NULL ||
        delta <= 0.0F) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow Euler step is invalid");
    }
    status = m3_flow_math_view(
        run, M3_FLOW_WS_SCALAR, length, &scalar, error);
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_view(
            run, M3_FLOW_WS_ARITHMETIC, length, &arithmetic, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_fill_f32(&scalar, delta, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_execute(
            run, M3_OP_MUL, velocity, &scalar, &arithmetic, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_execute(
            run, M3_OP_ADD, latent, &arithmetic, latent, error);
    }
    return status;
}

m3_status m3_flow_blend_overlap(
    m3_flow_run *run, m3_tensor_view *latent, uint64_t length,
    uint64_t overlap, float timestep, m3_error *error)
{
    m3_tensor_view latent_overlap;
    m3_tensor_view noise_prompt;
    m3_tensor_view previous_latent;
    m3_tensor_view scalar;
    m3_tensor_view arithmetic;
    float noise_coefficient;
    float previous_coefficient;
    m3_status status;

    if (overlap == 0U) {
        m3_error_reset(error);
        return M3_STATUS_OK;
    }
    if (run == NULL || latent == NULL || overlap > length ||
        overlap > run->config->carry_length) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow overlap blend is invalid");
    }
    m3_tensor_view_init(&latent_overlap);
    status = m3_tensor_slice(
        latent, 1U, 0U, overlap, &latent_overlap, error);
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_view(
            run, M3_FLOW_WS_NOISE_PROMPT, overlap, &noise_prompt,
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_view(
            run, M3_FLOW_WS_PREVIOUS_LATENT, overlap,
            &previous_latent, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_view(
            run, M3_FLOW_WS_SCALAR, overlap, &scalar, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_view(
            run, M3_FLOW_WS_ARITHMETIC, overlap, &arithmetic, error);
    }
    m3_flow_blend_coefficients(
        timestep, &noise_coefficient, &previous_coefficient);
    if (status == M3_STATUS_OK) {
        status = m3_flow_fill_f32(
            &scalar, noise_coefficient, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_execute(
            run, M3_OP_MUL, &noise_prompt, &scalar, &arithmetic, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_fill_f32(
            &scalar, previous_coefficient, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_execute(
            run, M3_OP_MUL, &previous_latent, &scalar,
            &latent_overlap, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_math_execute(
            run, M3_OP_ADD, &arithmetic, &latent_overlap,
            &latent_overlap, error);
    }
    return status;
}

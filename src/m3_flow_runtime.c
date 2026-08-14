/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_flow_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static m3_status m3_flow_cancel(m3_progress_callback progress,
                                void *context, uint64_t completed,
                                uint64_t total, m3_error *error)
{
    if (progress != NULL && !progress(context, completed, total)) {
        return m3_error_set(error, M3_STATUS_CANCELLED,
                            "flow synthesis was cancelled");
    }
    return M3_STATUS_OK;
}

static m3_status m3_flow_allocate_output(
    m3_backend *backend, const m3_flow_config *config,
    uint64_t frame_count, size_t chunk_count, m3_flow_output *built,
    m3_error *error)
{
    size_t index;
    m3_status status = M3_STATUS_OK;

    if (chunk_count > SIZE_MAX / sizeof(*built->storages) ||
        chunk_count > SIZE_MAX / sizeof(*built->chunks)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "flow output state allocation overflows");
    }
    built->storages = calloc(chunk_count, sizeof(*built->storages));
    built->chunks = calloc(chunk_count, sizeof(*built->chunks));
    if (built->storages == NULL || built->chunks == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate flow output state");
    }
    built->chunk_count = chunk_count;
    for (index = 0U; index < chunk_count && status == M3_STATUS_OK;
         ++index) {
        uint64_t start = 0U;
        uint64_t frames = 0U;
        uint64_t length = 0U;
        uint64_t shape[3];
        m3_tensor_metadata metadata;

        status = m3_flow_chunk_window(
            config, frame_count, index, &start, &frames, error);
        if (status == M3_STATUS_OK) {
            status = m3_condition_output_length(
                frames, config->condition.resize_numerator,
                config->condition.resize_denominator, &length, error);
        }
        shape[0] = 1U;
        shape[1] = config->latent_channels;
        shape[2] = length;
        if (status == M3_STATUS_OK) {
            status = m3_tensor_metadata_init(
                &metadata, M3_DTYPE_F32, 3U, shape, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_storage_allocate(
                backend, metadata.byte_count, 64U,
                &built->storages[index], error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_tensor_view_contiguous(
                &built->chunks[index], built->storages[index],
                M3_DTYPE_F32, 3U, shape, 0U, error);
        }
        (void)start;
    }
    return status;
}

static m3_status m3_flow_copy(m3_flow_run *run,
                              const m3_tensor_view *input,
                              m3_tensor_view *output,
                              m3_error *error)
{
    m3_command command = {0};

    command.kind = M3_OP_COPY;
    command.descriptor.copy = (m3_op_unary){input, output};
    return m3_command_executor_execute(
        &run->executor, &command, 1U, error);
}

static m3_status m3_flow_condition_chunk(
    m3_flow_run *run, const m3_condition_weights *weights,
    const m3_tensor_view *frames, uint64_t start, uint64_t frame_count,
    uint64_t previous_length, m3_condition_output *condition,
    uint64_t *overlap, m3_error *error)
{
    m3_tensor_view frame_chunk;
    m3_status status;

    m3_tensor_view_init(&frame_chunk);
    status = m3_tensor_slice(
        frames, 1U, start, frame_count, &frame_chunk, error);

    if (status == M3_STATUS_OK) {
        status = m3_condition_encode_core(
            run->backend, &run->config->condition, weights,
            &frame_chunk, NULL, NULL, condition, error);
    }
    if (status == M3_STATUS_OK) {
        uint64_t length = condition->tensor.metadata.shape[1];

        *overlap = previous_length < length ? previous_length : length;
    }
    if (status == M3_STATUS_OK && *overlap != 0U) {
        const uint64_t previous_shape[] = {
            1U, *overlap, run->config->condition_dimension
        };
        m3_tensor_view prefix;
        m3_tensor_view previous;

        m3_tensor_view_init(&prefix);
        status = m3_tensor_slice(
            &condition->tensor, 1U, 0U, *overlap, &prefix, error);
        if (status == M3_STATUS_OK) {
            status = m3_flow_view(
                run, M3_FLOW_WS_PREVIOUS_CONDITION, M3_DTYPE_F32, 3U,
                previous_shape, &previous, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_flow_copy(run, &previous, &prefix, error);
        }
    }
    return status;
}

static m3_status m3_flow_prepare_noise(
    m3_flow_run *run, m3_rng *rng, uint64_t length, uint64_t overlap,
    m3_tensor_view *latent, m3_error *error)
{
    const uint64_t noise_shape[] = {
        1U, run->config->latent_channels, length
    };
    const uint64_t latent_shape[] = {
        1U, length, run->config->latent_channels
    };
    const uint8_t to_sequence[] = {0U, 2U, 1U};
    m3_tensor_view noise;
    m3_tensor_view noise_sequence;
    float *noise_data = NULL;
    m3_status status = m3_flow_view(
        run, M3_FLOW_WS_NOISE_CHANNELS, M3_DTYPE_F32, 3U,
        noise_shape, &noise, error);

    m3_tensor_view_init(&noise_sequence);
    if (status == M3_STATUS_OK) {
        status = m3_flow_view(
            run, M3_FLOW_WS_LATENT, M3_DTYPE_F32, 3U, latent_shape,
            latent, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_data(&noise, (void **)&noise_data, error);
    }
    if (status == M3_STATUS_OK && noise_data == NULL) {
        status = m3_error_set(error, M3_STATUS_UNSUPPORTED,
                              "flow noise storage is not host visible");
    }
    if (status == M3_STATUS_OK) {
        status = m3_rng_normal_f32_fill(
            rng, noise_data, noise.metadata.element_count, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(
            &noise, to_sequence, &noise_sequence, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_copy(run, &noise_sequence, latent, error);
    }
    if (status == M3_STATUS_OK && overlap != 0U) {
        const uint64_t prompt_shape[] = {
            1U, overlap, run->config->latent_channels
        };
        m3_tensor_view prefix;
        m3_tensor_view prompt;

        m3_tensor_view_init(&prefix);
        status = m3_tensor_slice(
            latent, 1U, 0U, overlap, &prefix, error);
        if (status == M3_STATUS_OK) {
            status = m3_flow_view(
                run, M3_FLOW_WS_NOISE_PROMPT, M3_DTYPE_F32, 3U,
                prompt_shape, &prompt, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_flow_copy(run, &prefix, &prompt, error);
        }
    }
    return status;
}

static m3_status m3_flow_restore_overlap(
    m3_flow_run *run, m3_tensor_view *latent, uint64_t overlap,
    m3_error *error)
{
    const uint64_t shape[] = {
        1U, overlap, run->config->latent_channels
    };
    m3_tensor_view prefix;
    m3_tensor_view previous;
    m3_status status;

    if (overlap == 0U) {
        m3_error_reset(error);
        return M3_STATUS_OK;
    }
    m3_tensor_view_init(&prefix);
    status = m3_tensor_slice(
        latent, 1U, 0U, overlap, &prefix, error);
    if (status == M3_STATUS_OK) {
        status = m3_flow_view(
            run, M3_FLOW_WS_PREVIOUS_LATENT, M3_DTYPE_F32, 3U, shape,
            &previous, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_copy(run, &previous, &prefix, error);
    }
    return status;
}

m3_status m3_flow_preserve_carry(
    m3_flow_run *run, const m3_tensor_view *latent,
    const m3_tensor_view *condition, uint64_t length,
    uint64_t *carry_length, m3_error *error)
{
    uint64_t carry_start = 0U;
    uint64_t carry = 0U;
    m3_status status;

    if (run == NULL || latent == NULL || condition == NULL ||
        carry_length == NULL || length == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow carry state is invalid");
    }
    status = m3_flow_carry_window(
        run->config, length, &carry_start, &carry, error);
    if (status == M3_STATUS_OK && carry != 0U) {
        const uint64_t latent_shape[] = {
            1U, carry, run->config->latent_channels
        };
        const uint64_t condition_shape[] = {
            1U, carry, run->config->condition_dimension
        };
        m3_tensor_view latent_source;
        m3_tensor_view latent_previous;
        m3_tensor_view condition_source;
        m3_tensor_view condition_previous;

        m3_tensor_view_init(&latent_source);
        m3_tensor_view_init(&condition_source);
        status = m3_tensor_slice(
            latent, 1U, carry_start, carry, &latent_source, error);
        if (status == M3_STATUS_OK) {
            status = m3_flow_view(
                run, M3_FLOW_WS_PREVIOUS_LATENT, M3_DTYPE_F32, 3U,
                latent_shape, &latent_previous, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_flow_copy(
                run, &latent_source, &latent_previous, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_tensor_slice(
                condition, 1U, carry_start, carry,
                &condition_source, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_flow_view(
                run, M3_FLOW_WS_PREVIOUS_CONDITION, M3_DTYPE_F32, 3U,
                condition_shape, &condition_previous, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_flow_copy(
                run, &condition_source, &condition_previous, error);
        }
    }
    if (status == M3_STATUS_OK) {
        *carry_length = carry;
    }
    return status;
}

static m3_status m3_flow_store_chunk(
    m3_flow_run *run, const m3_tensor_view *latent,
    const m3_condition_output *condition, uint64_t length,
    m3_tensor_view *output, uint64_t *carry_length, m3_error *error)
{
    const uint8_t to_channels[] = {0U, 2U, 1U};
    m3_tensor_view latent_channels;
    m3_status status;

    m3_tensor_view_init(&latent_channels);
    status = m3_tensor_permute(
        latent, to_channels, &latent_channels, error);

    if (status == M3_STATUS_OK) {
        status = m3_flow_copy(run, &latent_channels, output, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_preserve_carry(
            run, latent, &condition->tensor, length, carry_length, error);
    }
    return status;
}

static m3_status m3_flow_denoise_chunk(
    m3_flow_run *run, m3_tensor_view *latent,
    const m3_tensor_view *condition, uint64_t length, uint64_t overlap,
    m3_progress_callback progress, void *progress_context,
    uint64_t *completed, uint64_t total, m3_error *error)
{
    uint32_t step;
    m3_status status = M3_STATUS_OK;

    for (step = 0U; step < run->config->inference_steps &&
                    status == M3_STATUS_OK; ++step) {
        float timestep = m3_flow_timestep(
            step, run->config->inference_steps);
        float delta = m3_flow_timestep_delta(
            step, run->config->inference_steps);
        m3_tensor_view velocity;

        if (overlap != 0U) {
            status = m3_flow_blend_overlap(
                run, latent, length, overlap, timestep, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_flow_forward(
                run, latent, condition, length, timestep, &velocity,
                error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_flow_guided_velocity(
                run, &velocity, length, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_flow_euler_step(
                run, latent, &velocity, length, delta, error);
        }
        if (status == M3_STATUS_OK) {
            ++*completed;
            status = m3_flow_cancel(
                progress, progress_context, *completed, total, error);
        }
    }
    return status;
}

static m3_status m3_flow_run_chunks(
    m3_flow_run *run, const m3_condition_weights *condition_weights,
    const m3_tensor_view *frame_hiddens, m3_rng *rng,
    m3_progress_callback progress, void *progress_context,
    m3_flow_output *built, uint64_t total, m3_error *error)
{
    m3_condition_output condition;
    uint64_t previous_length = 0U;
    uint64_t completed = 0U;
    size_t index;
    m3_status status = M3_STATUS_OK;

    m3_condition_output_init(&condition);
    for (index = 0U; index < built->chunk_count &&
                    status == M3_STATUS_OK; ++index) {
        uint64_t start = 0U;
        uint64_t frame_count = 0U;
        uint64_t overlap = 0U;
        uint64_t length;
        m3_tensor_view latent;

        status = m3_flow_chunk_window(
            run->config, frame_hiddens->metadata.shape[1], index,
            &start, &frame_count, error);
        if (status == M3_STATUS_OK) {
            status = m3_flow_condition_chunk(
                run, condition_weights, frame_hiddens, start,
                frame_count, previous_length, &condition, &overlap,
                error);
        }
        length = status == M3_STATUS_OK
                     ? condition.tensor.metadata.shape[1]
                     : 0U;
        if (status == M3_STATUS_OK) {
            status = m3_flow_prepare_noise(
                run, rng, length, overlap, &latent, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_flow_denoise_chunk(
                run, &latent, &condition.tensor, length, overlap,
                progress, progress_context, &completed, total, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_flow_restore_overlap(
                run, &latent, overlap, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_flow_store_chunk(
                run, &latent, &condition, length,
                &built->chunks[index], &previous_length, error);
        }
        m3_condition_output_dispose(&condition);
    }
    m3_condition_output_dispose(&condition);
    return status;
}

m3_status m3_flow_synthesize_core(
    m3_backend *backend, const m3_flow_config *config,
    const m3_flow_weights *weights,
    const m3_condition_weights *condition_weights,
    const m3_tensor_view *frame_hiddens, m3_rng *rng,
    m3_progress_callback progress, void *progress_context,
    m3_flow_output *output, m3_error *error)
{
    m3_runtime_tensor_spec specs[M3_FLOW_WORKSPACE_COUNT];
    m3_flow_output built;
    m3_flow_run run;
    m3_rng local_rng;
    size_t chunk_count = 0U;
    uint64_t maximum_length = 0U;
    uint64_t total = 0U;
    m3_status status;

    m3_flow_output_init(&built);
    (void)memset(&run, 0, sizeof(run));
    run.backend = backend;
    run.config = config;
    run.weights = weights;
    m3_runtime_workspace_init(&run.workspace);
    m3_command_executor_init(&run.executor, backend);
    status = m3_flow_validate(
        backend, config, weights, condition_weights, frame_hiddens, rng,
        output, &chunk_count, &maximum_length, &total, error);
    if (status == M3_STATUS_OK) {
        local_rng = *rng;
        run.maximum_length = maximum_length;
        m3_flow_workspace_specs(config, maximum_length, specs);
        status = m3_flow_preflight(
            backend, config, frame_hiddens, chunk_count, maximum_length,
            specs, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_cancel(
            progress, progress_context, 0U, total, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_allocate_output(
            backend, config, frame_hiddens->metadata.shape[1],
            chunk_count, &built, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_runtime_workspace_build(
            &run.workspace, backend, specs, M3_FLOW_WORKSPACE_COUNT,
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_run_chunks(
            &run, condition_weights, frame_hiddens, &local_rng,
            progress, progress_context, &built, total, error);
    }
    m3_runtime_workspace_dispose(&run.workspace);
    m3_command_executor_dispose(&run.executor);
    if (status != M3_STATUS_OK) {
        m3_flow_output_dispose(&built);
        return status;
    }
    m3_flow_output_dispose(output);
    *output = built;
    *rng = local_rng;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_flow_synthesize(
    m3_backend *backend, const m3_flow_weights *weights,
    const m3_condition_weights *condition_weights,
    const m3_tensor_view *frame_hiddens, m3_rng *rng,
    m3_progress_callback progress, void *progress_context,
    m3_flow_output *output, m3_error *error)
{
    m3_flow_config config;

    m3_flow_config_init(&config);
    return m3_flow_synthesize_core(
        backend, &config, weights, condition_weights, frame_hiddens, rng,
        progress, progress_context, output, error);
}

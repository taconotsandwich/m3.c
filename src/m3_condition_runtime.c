/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_rvq_condition_internal.h"

#include <stdint.h>
#include <string.h>

typedef struct {
    m3_backend *backend;
    const m3_condition_config *config;
    const m3_condition_weights *weights;
    const m3_tensor_view *frames;
    uint64_t frame_count;
    uint64_t output_length;
    m3_runtime_workspace workspace;
    m3_command_executor executor;
    m3_condition_output built;
} m3_condition_run;

static m3_status m3_condition_view(
    m3_condition_run *run, size_t workspace_index, uint8_t rank,
    const uint64_t *shape, m3_tensor_view *view, m3_error *error)
{
    m3_tensor_view_init(view);
    return m3_tensor_view_contiguous(
        view, run->workspace.storages[workspace_index], M3_DTYPE_F32,
        rank, shape, 0U, error);
}

static void m3_condition_copy(m3_command *command,
                              const m3_tensor_view *input,
                              m3_tensor_view *output)
{
    (void)memset(command, 0, sizeof(*command));
    command->kind = M3_OP_COPY;
    command->descriptor.copy.input = input;
    command->descriptor.copy.output = output;
}

static m3_status m3_condition_mix(
    m3_condition_run *run, m3_progress_callback progress,
    void *progress_context, m3_error *error)
{
    const uint8_t permutation[] = {0U, 2U, 3U, 1U};
    const uint64_t layer_shape[] = {
        1U, 1U, run->config->hidden_size, run->frame_count
    };
    const uint64_t mix_shape[] = {
        1U, run->config->hidden_size, run->frame_count
    };
    const uint64_t scalar_shape[] = {1U, 1U, 1U, 1U};
    m3_tensor_view cast_layer;
    m3_tensor_view frames_permuted;
    m3_tensor_view mix;
    m3_tensor_view softmax_weights;
    m3_tensor_view weighted_layer;
    m3_command softmax = {0};
    size_t layer;
    m3_status status;

    m3_tensor_view_init(&frames_permuted);
    status = m3_tensor_permute(
        run->frames, permutation, &frames_permuted, error);
    if (status == M3_STATUS_OK) {
        const uint64_t weights_shape[] = {run->config->layer_count};

        status = m3_condition_view(
            run, M3_CONDITION_WS_LAYER_WEIGHTS, 1U, weights_shape,
            &softmax_weights, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_condition_view(
            run, M3_CONDITION_WS_CAST_LAYER, 4U, layer_shape,
            &cast_layer, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_condition_view(
            run, M3_CONDITION_WS_WEIGHTED_LAYER, 4U, layer_shape,
            &weighted_layer, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_condition_view(
            run, M3_CONDITION_WS_MIX, 3U, mix_shape, &mix, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    softmax.kind = M3_OP_SOFTMAX;
    softmax.descriptor.softmax.input = run->weights->layer_weight_logits;
    softmax.descriptor.softmax.output = &softmax_weights;
    status = m3_command_executor_execute(
        &run->executor, &softmax, 1U, error);
    for (layer = 0U; layer < run->config->layer_count &&
                     status == M3_STATUS_OK; ++layer) {
        m3_tensor_view frame_layer;
        m3_tensor_view scalar;
        m3_tensor_view scalar_slice;
        m3_tensor_view weighted_mix_shape;
        m3_command commands[3] = {0};

        m3_tensor_view_init(&frame_layer);
        m3_tensor_view_init(&scalar_slice);
        m3_tensor_view_init(&scalar);
        m3_tensor_view_init(&weighted_mix_shape);
        status = m3_tensor_slice(
            &frames_permuted, 1U, (uint64_t)layer, 1U, &frame_layer,
            error);
        if (status == M3_STATUS_OK) {
            status = m3_tensor_slice(
                &softmax_weights, 0U, (uint64_t)layer, 1U,
                &scalar_slice, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_tensor_reshape(
                &scalar_slice, 4U, scalar_shape, &scalar, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_tensor_reshape(
                &weighted_layer, 3U, mix_shape, &weighted_mix_shape,
                error);
        }
        if (status != M3_STATUS_OK) {
            break;
        }
        commands[0].kind = M3_OP_CAST;
        commands[0].descriptor.cast = (m3_op_unary){
            &frame_layer, &cast_layer
        };
        commands[1].kind = M3_OP_MUL;
        commands[1].descriptor.mul = (m3_op_binary){
            &cast_layer, &scalar, &weighted_layer
        };
        if (layer == 0U) {
            m3_condition_copy(
                &commands[2], &weighted_mix_shape, &mix);
        } else {
            commands[2].kind = M3_OP_ADD;
            commands[2].descriptor.add = (m3_op_binary){
                &mix, &weighted_mix_shape, &mix
            };
        }
        status = m3_command_executor_execute(
            &run->executor, commands, 3U, error);
        if (status == M3_STATUS_OK) {
            status = m3_rvq_condition_cancel(
                progress, progress_context, (uint64_t)(layer + 1U),
                (uint64_t)run->config->layer_count + 2U,
                "condition encode", error);
        }
    }
    return status;
}

static m3_status m3_condition_project(
    m3_condition_run *run, m3_progress_callback progress,
    void *progress_context, m3_error *error)
{
    const uint64_t mix_shape[] = {
        1U, run->config->hidden_size, run->frame_count
    };
    const uint64_t convolution_shape[] = {
        1U, run->config->output_size, run->frame_count
    };
    const uint64_t scale_shape[] = {1U, 1U, 1U};
    m3_tensor_view convolution;
    m3_tensor_view mix;
    m3_tensor_view scale;
    m3_command commands[2] = {0};
    m3_status status;

    m3_tensor_view_init(&scale);
    status = m3_condition_view(
        run, M3_CONDITION_WS_MIX, 3U, mix_shape, &mix, error);

    if (status == M3_STATUS_OK) {
        status = m3_condition_view(
            run, M3_CONDITION_WS_CONVOLUTION, 3U, convolution_shape,
            &convolution, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(
            run->weights->layer_scale, 3U, scale_shape, &scale, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    commands[0].kind = M3_OP_MUL;
    commands[0].descriptor.mul = (m3_op_binary){&mix, &scale, &mix};
    commands[1].kind = M3_OP_CONV1D;
    commands[1].descriptor.conv1d = (m3_op_conv1d){
        &mix, run->weights->projection, run->weights->bias, &convolution,
        1U, 1U, 1U, 1U, 1U
    };
    status = m3_command_executor_execute(
        &run->executor, commands, 2U, error);
    if (status == M3_STATUS_OK) {
        status = m3_rvq_condition_cancel(
            progress, progress_context,
            (uint64_t)run->config->layer_count + 1U,
            (uint64_t)run->config->layer_count + 2U,
            "condition encode", error);
    }
    return status;
}

static m3_status m3_condition_resize(
    m3_condition_run *run, m3_progress_callback progress,
    void *progress_context, m3_error *error)
{
    const uint64_t convolution_shape[] = {
        1U, run->config->output_size, run->frame_count
    };
    const uint64_t resized_shape[] = {
        1U, run->config->output_size, run->output_length
    };
    const uint8_t permutation[] = {0U, 2U, 1U};
    m3_tensor_view convolution;
    m3_tensor_view resized;
    m3_tensor_view transposed;
    m3_command commands[2] = {0};
    m3_status status;

    m3_tensor_view_init(&transposed);
    status = m3_condition_view(
        run, M3_CONDITION_WS_CONVOLUTION, 3U, convolution_shape,
        &convolution, error);

    if (status == M3_STATUS_OK) {
        status = m3_condition_view(
            run, M3_CONDITION_WS_RESIZED, 3U, resized_shape, &resized,
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(
            &resized, permutation, &transposed, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    commands[0].kind = M3_OP_NEAREST_RESIZE1D;
    commands[0].descriptor.nearest_resize1d = (m3_op_unary){
        &convolution, &resized
    };
    m3_condition_copy(&commands[1], &transposed, &run->built.tensor);
    status = m3_command_executor_execute(
        &run->executor, commands, 2U, error);
    if (status == M3_STATUS_OK) {
        status = m3_rvq_condition_cancel(
            progress, progress_context,
            (uint64_t)run->config->layer_count + 2U,
            (uint64_t)run->config->layer_count + 2U,
            "condition encode", error);
    }
    return status;
}

m3_status m3_condition_encode_core(
    m3_backend *backend, const m3_condition_config *config,
    const m3_condition_weights *weights, const m3_tensor_view *frames,
    m3_progress_callback progress, void *progress_context,
    m3_condition_output *output, m3_error *error)
{
    m3_runtime_tensor_spec specs[M3_CONDITION_WORKSPACE_COUNT];
    m3_condition_run run;
    uint64_t output_shape[3];
    size_t output_bytes = 0U;
    m3_status status;

    (void)memset(&run, 0, sizeof(run));
    run.backend = backend;
    run.config = config;
    run.weights = weights;
    run.frames = frames;
    m3_runtime_workspace_init(&run.workspace);
    m3_command_executor_init(&run.executor, backend);
    m3_condition_output_init(&run.built);
    status = m3_condition_validate(
        backend, config, weights, frames, output, &run.output_length,
        &output_bytes, error);
    if (status == M3_STATUS_OK) {
        run.frame_count = frames->metadata.shape[1];
        m3_condition_workspace_specs(
            config, run.frame_count, run.output_length, specs);
        status = m3_rvq_condition_preflight(
            backend, output_bytes, specs, M3_CONDITION_WORKSPACE_COUNT,
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_rvq_condition_cancel(
            progress, progress_context, 0U,
            (uint64_t)config->layer_count + 2U, "condition encode",
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_storage_allocate(
            backend, output_bytes, 64U, &run.built.storage, error);
    }
    output_shape[0] = 1U;
    output_shape[1] = run.output_length;
    output_shape[2] = config != NULL ? config->output_size : 0U;
    if (status == M3_STATUS_OK) {
        status = m3_tensor_view_contiguous(
            &run.built.tensor, run.built.storage, M3_DTYPE_F32, 3U,
            output_shape, 0U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_runtime_workspace_build(
            &run.workspace, backend, specs, M3_CONDITION_WORKSPACE_COUNT,
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_condition_mix(
            &run, progress, progress_context, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_condition_project(
            &run, progress, progress_context, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_condition_resize(
            &run, progress, progress_context, error);
    }
    m3_runtime_workspace_dispose(&run.workspace);
    m3_command_executor_dispose(&run.executor);
    if (status != M3_STATUS_OK) {
        m3_condition_output_dispose(&run.built);
        return status;
    }
    m3_condition_output_dispose(output);
    *output = run.built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_condition_encode(
    m3_backend *backend, const m3_condition_weights *weights,
    const m3_tensor_view *frames, m3_progress_callback progress,
    void *progress_context, m3_condition_output *output, m3_error *error)
{
    static const m3_condition_config config = {
        4096U, 8U, 2048U, 441U, 128U
    };

    return m3_condition_encode_core(
        backend, &config, weights, frames, progress, progress_context,
        output, error);
}

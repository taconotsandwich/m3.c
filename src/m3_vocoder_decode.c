/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_vocoder_internal.h"

#include <string.h>

#define M3_VOCODER_BATCH_COMMAND_CAPACITY 17U
#define M3_VOCODER_BATCH_VIEW_CAPACITY 37U

typedef struct {
    m3_command commands[M3_VOCODER_BATCH_COMMAND_CAPACITY];
    m3_tensor_view views[M3_VOCODER_BATCH_VIEW_CAPACITY];
    size_t command_count;
    size_t view_count;
} m3_vocoder_batch;

typedef struct {
    m3_vocoder_runtime *runtime;
    m3_storage *buffers[M3_VOCODER_DECODE_BUFFER_COUNT];
    m3_command_executor executor;
    m3_vocoder_output built;
    uint64_t completed;
    m3_progress_callback progress;
    void *progress_context;
} m3_vocoder_decode_run;

static m3_status m3_vocoder_decode_checkpoint(
    m3_vocoder_decode_run *run, uint64_t completed, m3_error *error)
{
    run->completed = completed;
    if (run->progress != NULL &&
        !run->progress(run->progress_context, completed,
                       M3_VOCODER_DECODE_OPERATION_COUNT)) {
        return m3_error_set(error, M3_STATUS_CANCELLED,
                            "vocoder chunk decode was cancelled");
    }
    return M3_STATUS_OK;
}

static void m3_vocoder_batch_init(m3_vocoder_batch *batch)
{
    (void)memset(batch, 0, sizeof(*batch));
}

static m3_status m3_vocoder_batch_snapshot(
    m3_vocoder_batch *batch, const m3_tensor_view *source,
    m3_tensor_view **snapshot, m3_error *error)
{
    if (source == NULL || batch->view_count >=
                              M3_VOCODER_BATCH_VIEW_CAPACITY) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "vocoder command view capacity is inconsistent");
    }
    *snapshot = &batch->views[batch->view_count++];
    **snapshot = *source;
    return M3_STATUS_OK;
}

static m3_status m3_vocoder_batch_pair(
    m3_vocoder_batch *batch, const m3_tensor_view *input,
    const m3_tensor_view *output, m3_command **command,
    const m3_tensor_view **input_snapshot,
    m3_tensor_view **output_snapshot, m3_error *error)
{
    m3_tensor_view *input_copy = NULL;
    m3_status status;

    if (batch->command_count >= M3_VOCODER_BATCH_COMMAND_CAPACITY) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "vocoder command capacity is inconsistent");
    }
    status = m3_vocoder_batch_snapshot(
        batch, input, &input_copy, error);
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_snapshot(
            batch, output, output_snapshot, error);
    }
    if (status == M3_STATUS_OK) {
        *command = &batch->commands[batch->command_count++];
        (void)memset(*command, 0, sizeof(**command));
        *input_snapshot = input_copy;
    }
    return status;
}

static m3_status m3_vocoder_batch_conv1d(
    m3_vocoder_batch *batch, const m3_tensor_view *input,
    const m3_tensor_view *weight, const m3_tensor_view *bias,
    const m3_tensor_view *output, uint64_t dilation, uint64_t padding,
    m3_error *error)
{
    const m3_tensor_view *input_snapshot = NULL;
    m3_tensor_view *output_snapshot = NULL;
    m3_command *command = NULL;
    m3_status status = m3_vocoder_batch_pair(
        batch, input, output, &command, &input_snapshot, &output_snapshot,
        error);

    if (status == M3_STATUS_OK) {
        command->kind = M3_OP_CONV1D;
        command->descriptor.conv1d.input = input_snapshot;
        command->descriptor.conv1d.weight = weight;
        command->descriptor.conv1d.bias = bias;
        command->descriptor.conv1d.output = output_snapshot;
        command->descriptor.conv1d.groups = 1U;
        command->descriptor.conv1d.stride = 1U;
        command->descriptor.conv1d.dilation = dilation;
        command->descriptor.conv1d.pad_left = padding;
        command->descriptor.conv1d.pad_right = padding;
    }
    return status;
}

static m3_status m3_vocoder_batch_transpose(
    m3_vocoder_batch *batch, const m3_tensor_view *input,
    const m3_tensor_view *weight, const m3_tensor_view *bias,
    const m3_tensor_view *output, uint64_t stride, m3_error *error)
{
    const m3_tensor_view *input_snapshot = NULL;
    m3_tensor_view *output_snapshot = NULL;
    m3_command *command = NULL;
    m3_status status = m3_vocoder_batch_pair(
        batch, input, output, &command, &input_snapshot, &output_snapshot,
        error);

    if (status == M3_STATUS_OK) {
        command->kind = M3_OP_CONV_TRANSPOSE1D;
        command->descriptor.conv_transpose1d.input = input_snapshot;
        command->descriptor.conv_transpose1d.weight = weight;
        command->descriptor.conv_transpose1d.bias = bias;
        command->descriptor.conv_transpose1d.output = output_snapshot;
        command->descriptor.conv_transpose1d.groups = 1U;
        command->descriptor.conv_transpose1d.stride = stride;
        command->descriptor.conv_transpose1d.dilation = 1U;
        command->descriptor.conv_transpose1d.pad_left = stride / 2U;
        command->descriptor.conv_transpose1d.pad_right = stride / 2U;
        command->descriptor.conv_transpose1d.output_padding = 0U;
    }
    return status;
}

static m3_status m3_vocoder_batch_snake(
    m3_vocoder_batch *batch, const m3_tensor_view *input,
    const m3_tensor_view *alpha, const m3_tensor_view *output,
    m3_error *error)
{
    const m3_tensor_view *input_snapshot = NULL;
    m3_tensor_view *output_snapshot = NULL;
    m3_command *command = NULL;
    m3_status status = m3_vocoder_batch_pair(
        batch, input, output, &command, &input_snapshot, &output_snapshot,
        error);

    if (status == M3_STATUS_OK) {
        command->kind = M3_OP_SNAKE1D;
        command->descriptor.snake1d.input = input_snapshot;
        command->descriptor.snake1d.alpha = alpha;
        command->descriptor.snake1d.output = output_snapshot;
    }
    return status;
}

static m3_status m3_vocoder_batch_add(
    m3_vocoder_batch *batch, const m3_tensor_view *left,
    const m3_tensor_view *right, const m3_tensor_view *output,
    m3_error *error)
{
    m3_tensor_view *left_snapshot = NULL;
    m3_tensor_view *right_snapshot = NULL;
    m3_tensor_view *output_snapshot = NULL;
    m3_command *command;
    m3_status status;

    if (batch->command_count >= M3_VOCODER_BATCH_COMMAND_CAPACITY) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "vocoder command capacity is inconsistent");
    }
    status = m3_vocoder_batch_snapshot(
        batch, left, &left_snapshot, error);
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_snapshot(
            batch, right, &right_snapshot, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_snapshot(
            batch, output, &output_snapshot, error);
    }
    if (status == M3_STATUS_OK) {
        command = &batch->commands[batch->command_count++];
        (void)memset(command, 0, sizeof(*command));
        command->kind = M3_OP_ADD;
        command->descriptor.add.left = left_snapshot;
        command->descriptor.add.right = right_snapshot;
        command->descriptor.add.output = output_snapshot;
    }
    return status;
}

static m3_status m3_vocoder_batch_tanh(
    m3_vocoder_batch *batch, const m3_tensor_view *input,
    const m3_tensor_view *output, m3_error *error)
{
    const m3_tensor_view *input_snapshot = NULL;
    m3_tensor_view *output_snapshot = NULL;
    m3_command *command = NULL;
    m3_status status = m3_vocoder_batch_pair(
        batch, input, output, &command, &input_snapshot, &output_snapshot,
        error);

    if (status == M3_STATUS_OK) {
        command->kind = M3_OP_TANH;
        command->descriptor.tanh.input = input_snapshot;
        command->descriptor.tanh.output = output_snapshot;
    }
    return status;
}

static m3_status m3_vocoder_workspace_view(
    const m3_vocoder_decode_run *run, size_t slot, uint64_t channels,
    uint64_t length, m3_tensor_view *view, m3_error *error)
{
    const uint64_t shape[] = {2U, channels, length};

    m3_tensor_view_init(view);
    return m3_tensor_view_contiguous(
        view, run->buffers[slot], M3_DTYPE_F32, 3U, shape, 0U, error);
}

static m3_status m3_vocoder_batch_execute(
    m3_vocoder_decode_run *run, m3_vocoder_batch *batch,
    size_t expected_count, m3_error *error)
{
    if (batch->command_count != expected_count) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "vocoder batch operation count is inconsistent");
    }
    return m3_command_executor_execute(
        &run->executor, batch->commands, batch->command_count, error);
}

static m3_status m3_vocoder_decode_prelude(
    m3_vocoder_decode_run *run, const m3_tensor_view *folded,
    size_t *hidden_slot, uint64_t length, m3_error *error)
{
    const m3_vocoder_plan_config *config = &run->runtime->config;
    const m3_vocoder_weights *weights = &run->runtime->bound;
    m3_vocoder_batch batch;
    m3_tensor_view projected;
    m3_tensor_view hidden;
    m3_status status;

    m3_vocoder_batch_init(&batch);
    status = m3_vocoder_workspace_view(
        run, 0U, config->decoder_output_channels, length, &projected,
        error);
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_conv1d(
            &batch, folded, weights->decoder_input_weight,
            weights->decoder_input_bias, &projected, 1U, 0U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_workspace_view(
            run, 1U, config->initial_channels, length, &hidden, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_conv1d(
            &batch, &projected, weights->convolution_input_weight,
            weights->convolution_input_bias, &hidden, 1U, 3U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_execute(run, &batch, 2U, error);
    }
    if (status == M3_STATUS_OK) {
        *hidden_slot = 1U;
    }
    return status;
}

static m3_status m3_vocoder_decode_residual(
    m3_vocoder_decode_run *run, m3_vocoder_batch *batch,
    const m3_vocoder_residual_weights *weights, size_t *hidden_slot,
    uint64_t channels, uint64_t length, size_t residual, m3_error *error)
{
    static const uint64_t dilations[M3_VOCODER_RESIDUAL_COUNT] = {
        1U, 3U, 9U
    };
    size_t first_slot = (*hidden_slot + 1U) %
                        M3_VOCODER_DECODE_BUFFER_COUNT;
    size_t second_slot = (*hidden_slot + 2U) %
                         M3_VOCODER_DECODE_BUFFER_COUNT;
    m3_tensor_view hidden;
    m3_tensor_view first;
    m3_tensor_view second;
    uint64_t dilation = dilations[residual];
    m3_status status = m3_vocoder_workspace_view(
        run, *hidden_slot, channels, length, &hidden, error);

    if (status == M3_STATUS_OK) {
        status = m3_vocoder_workspace_view(
            run, first_slot, channels, length, &first, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_workspace_view(
            run, second_slot, channels, length, &second, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_snake(
            batch, &hidden, weights->snake1_alpha, &first, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_conv1d(
            batch, &first, weights->conv1_weight, weights->conv1_bias,
            &second, dilation, 3U * dilation, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_snake(
            batch, &second, weights->snake2_alpha, &first, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_conv1d(
            batch, &first, weights->conv2_weight, weights->conv2_bias,
            &second, 1U, 0U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_add(
            batch, &hidden, &second, &second, error);
    }
    if (status == M3_STATUS_OK) {
        *hidden_slot = second_slot;
    }
    return status;
}

static m3_status m3_vocoder_decode_block(
    m3_vocoder_decode_run *run, size_t block, size_t *hidden_slot,
    uint64_t *channels, uint64_t *length, m3_error *error)
{
    const m3_vocoder_block_weights *weights =
        &run->runtime->bound.blocks[block];
    uint64_t stride = run->runtime->config.strides[block];
    size_t first_slot = (*hidden_slot + 1U) %
                        M3_VOCODER_DECODE_BUFFER_COUNT;
    size_t second_slot = (*hidden_slot + 2U) %
                         M3_VOCODER_DECODE_BUFFER_COUNT;
    m3_vocoder_batch batch;
    m3_tensor_view hidden;
    m3_tensor_view first;
    m3_tensor_view second;
    size_t residual;
    m3_status status;

    m3_vocoder_batch_init(&batch);
    status = m3_vocoder_workspace_view(
        run, *hidden_slot, *channels, *length, &hidden, error);
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_workspace_view(
            run, first_slot, *channels, *length, &first, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_snake(
            &batch, &hidden, weights->snake_alpha, &first, error);
    }
    *channels /= 2U;
    *length *= stride;
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_workspace_view(
            run, second_slot, *channels, *length, &second, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_transpose(
            &batch, &first, weights->transpose_weight,
            weights->transpose_bias, &second, stride, error);
    }
    *hidden_slot = second_slot;
    for (residual = 0U; residual < M3_VOCODER_RESIDUAL_COUNT &&
                        status == M3_STATUS_OK; ++residual) {
        status = m3_vocoder_decode_residual(
            run, &batch, &weights->residuals[residual], hidden_slot,
            *channels, *length, residual, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_execute(run, &batch, 17U, error);
    }
    return status;
}

static m3_status m3_vocoder_decode_final(
    m3_vocoder_decode_run *run, size_t hidden_slot, uint64_t channels,
    uint64_t length, m3_error *error)
{
    const m3_vocoder_weights *weights = &run->runtime->bound;
    const uint64_t mono_shape[] = {2U, 1U, length};
    size_t final_slot = (hidden_slot + 1U) %
                        M3_VOCODER_DECODE_BUFFER_COUNT;
    m3_vocoder_batch batch;
    m3_tensor_view hidden;
    m3_tensor_view activated;
    m3_tensor_view mono;
    m3_status status;

    m3_vocoder_batch_init(&batch);
    m3_tensor_view_init(&mono);
    status = m3_vocoder_workspace_view(
        run, hidden_slot, channels, length, &hidden, error);
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_workspace_view(
            run, final_slot, channels, length, &activated, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(
            &run->built.waveform, 3U, mono_shape, &mono, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_snake(
            &batch, &hidden, weights->snake_output_alpha, &activated,
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_conv1d(
            &batch, &activated, weights->convolution_output_weight,
            weights->convolution_output_bias, &mono, 1U, 3U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_tanh(
            &batch, &mono, &mono, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_batch_execute(run, &batch, 3U, error);
    }
    return status;
}

static void m3_vocoder_decode_run_dispose(m3_vocoder_decode_run *run)
{
    size_t slot;

    for (slot = 0U; slot < M3_VOCODER_DECODE_BUFFER_COUNT; ++slot) {
        m3_storage_free(run->buffers[slot]);
    }
    m3_command_executor_dispose(&run->executor);
}

static m3_status m3_vocoder_decode_allocate(
    m3_vocoder_decode_run *run,
    const m3_vocoder_decode_measurement *measurement, m3_error *error)
{
    const uint64_t output_shape[] = {
        1U, 2U, measurement->output_length
    };
    size_t slot;
    m3_status status = M3_STATUS_OK;

    for (slot = 0U; slot < M3_VOCODER_DECODE_BUFFER_COUNT &&
                    status == M3_STATUS_OK; ++slot) {
        status = m3_storage_allocate(
            run->runtime->backend, measurement->buffer_bytes[slot], 64U,
            &run->buffers[slot], error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_storage_allocate(
            run->runtime->backend, measurement->output_bytes, 64U,
            &run->built.storage, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_view_contiguous(
            &run->built.waveform, run->built.storage, M3_DTYPE_F32, 3U,
            output_shape, 0U, error);
    }
    return status;
}

m3_status m3_vocoder_decode_chunk(
    m3_vocoder_runtime *runtime, const m3_tensor_view *latents,
    m3_progress_callback progress, void *progress_context,
    m3_vocoder_output *output, m3_error *error)
{
    m3_vocoder_decode_measurement measurement;
    m3_vocoder_decode_run run;
    m3_tensor_view folded;
    uint64_t folded_shape[3];
    uint64_t channels;
    uint64_t length;
    size_t hidden_slot = 0U;
    size_t block;
    m3_status status = m3_vocoder_decode_prepare(
        runtime, latents, output, &measurement, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    (void)memset(&run, 0, sizeof(run));
    run.runtime = runtime;
    run.progress = progress;
    run.progress_context = progress_context;
    m3_command_executor_init(&run.executor, runtime->backend);
    m3_vocoder_output_init(&run.built);
    status = m3_vocoder_decode_checkpoint(&run, 0U, error);
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_decode_allocate(&run, &measurement, error);
    }
    folded_shape[0] = 2U;
    folded_shape[1] = runtime->config.decoder_input_channels;
    folded_shape[2] = latents->metadata.shape[2];
    m3_tensor_view_init(&folded);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(
            latents, 3U, folded_shape, &folded, error);
    }
    length = latents->metadata.shape[2];
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_decode_prelude(
            &run, &folded, &hidden_slot, length, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_decode_checkpoint(&run, 2U, error);
    }
    channels = runtime->config.initial_channels;
    for (block = 0U; block < M3_VOCODER_BLOCK_COUNT &&
                         status == M3_STATUS_OK; ++block) {
        status = m3_vocoder_decode_block(
            &run, block, &hidden_slot, &channels, &length, error);
        if (status == M3_STATUS_OK) {
            status = m3_vocoder_decode_checkpoint(
                &run, 19U + 17U * (uint64_t)block, error);
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_decode_final(
            &run, hidden_slot, channels, length, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_decode_checkpoint(
            &run, M3_VOCODER_DECODE_OPERATION_COUNT, error);
    }
    m3_vocoder_decode_run_dispose(&run);
    if (status != M3_STATUS_OK) {
        m3_vocoder_output_dispose(&run.built);
        return status;
    }
    m3_vocoder_output_dispose(output);
    *output = run.built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

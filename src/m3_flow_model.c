/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_flow_internal.h"

#include <string.h>

static void m3_flow_model_linear(m3_command *command,
                                 const m3_tensor_view *input,
                                 const m3_tensor_view *weight,
                                 const m3_tensor_view *bias,
                                 m3_tensor_view *output)
{
    (void)memset(command, 0, sizeof(*command));
    command->kind = M3_OP_LINEAR;
    command->descriptor.linear = (m3_op_linear){
        input, weight, bias, output
    };
}

static m3_status m3_flow_model_row(const m3_tensor_view *source,
                                   uint64_t row, m3_tensor_view *view,
                                   m3_error *error)
{
    if (source == NULL || source->metadata.rank == 0U ||
        source->metadata.shape[0] <= row) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow model row is invalid");
    }
    m3_tensor_view_init(view);
    return m3_tensor_slice(source, 0U, row, 1U, view, error);
}

static m3_status m3_flow_concat_input(
    m3_flow_run *run, const m3_tensor_view *latent,
    const m3_tensor_view *condition, uint64_t length,
    m3_tensor_view *concatenated, m3_error *error)
{
    const uint64_t channels =
        (uint64_t)run->config->latent_channels * 2U +
        run->config->condition_dimension;
    const uint64_t shape[] = {2U, channels, length};
    const uint8_t to_channels[] = {0U, 2U, 1U};
    m3_tensor_view latent_channels;
    m3_tensor_view condition_channels;
    m3_tensor_view row_zero;
    m3_tensor_view row_one;
    m3_tensor_view latent_zero;
    m3_tensor_view latent_one;
    m3_tensor_view condition_zero;
    m3_command commands[3] = {0};
    m3_status status = m3_flow_view(
        run, M3_FLOW_WS_CONCATENATED, M3_DTYPE_F32, 3U, shape,
        concatenated, error);

    m3_tensor_view_init(&latent_channels);
    m3_tensor_view_init(&condition_channels);
    m3_tensor_view_init(&latent_zero);
    m3_tensor_view_init(&latent_one);
    m3_tensor_view_init(&condition_zero);
    if (status == M3_STATUS_OK) {
        status = m3_flow_zero_storage(concatenated->storage, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(
            latent, to_channels, &latent_channels, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(
            condition, to_channels, &condition_channels, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_model_row(
            concatenated, 0U, &row_zero, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_model_row(
            concatenated, 1U, &row_one, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_slice(
            &row_zero, 1U, 0U, run->config->latent_channels,
            &latent_zero, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_slice(
            &row_one, 1U, 0U, run->config->latent_channels,
            &latent_one, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_slice(
            &row_zero, 1U,
            (uint64_t)run->config->latent_channels * 2U,
            run->config->condition_dimension, &condition_zero, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    commands[0].kind = M3_OP_COPY;
    commands[0].descriptor.copy = (m3_op_unary){
        &latent_channels, &latent_zero
    };
    commands[1].kind = M3_OP_COPY;
    commands[1].descriptor.copy = (m3_op_unary){
        &latent_channels, &latent_one
    };
    commands[2].kind = M3_OP_COPY;
    commands[2].descriptor.copy = (m3_op_unary){
        &condition_channels, &condition_zero
    };
    return m3_command_executor_execute(
        &run->executor, commands, 3U, error);
}

static m3_status m3_flow_preprocess(
    m3_flow_run *run, const m3_tensor_view *concatenated,
    uint64_t length, m3_tensor_view *projected, m3_error *error)
{
    const uint64_t inner = (uint64_t)run->config->attention_heads *
                           run->config->head_dimension;
    const uint64_t channels =
        (uint64_t)run->config->latent_channels * 2U +
        run->config->condition_dimension;
    const uint64_t preprocessed_shape[] = {2U, channels, length};
    const uint64_t projected_shape[] = {2U, length, inner};
    const uint8_t to_sequence[] = {0U, 2U, 1U};
    m3_tensor_view preprocessed;
    m3_tensor_view preprocessed_sequence;
    m3_command command = {0};
    m3_status status = m3_flow_view(
        run, M3_FLOW_WS_PREPROCESSED, M3_DTYPE_F32, 3U,
        preprocessed_shape, &preprocessed, error);

    m3_tensor_view_init(&preprocessed_sequence);
    if (status == M3_STATUS_OK) {
        status = m3_flow_view(
            run, M3_FLOW_WS_PROJECTED, M3_DTYPE_F32, 3U,
            projected_shape, projected, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    command.kind = M3_OP_CONV1D;
    command.descriptor.conv1d = (m3_op_conv1d){
        concatenated, run->weights->preprocess_convolution, NULL,
        &preprocessed, 1U, 1U, 1U, 0U, 0U
    };
    status = m3_command_executor_execute(
        &run->executor, &command, 1U, error);
    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_ADD;
    command.descriptor.add = (m3_op_binary){
        &preprocessed, concatenated, &preprocessed
    };
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, &command, 1U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(
            &preprocessed, to_sequence, &preprocessed_sequence, error);
    }
    m3_flow_model_linear(&command, &preprocessed_sequence,
                         run->weights->input_projection, NULL,
                         projected);
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, &command, 1U, error);
    }
    return status;
}

static m3_status m3_flow_time_embedding(
    m3_flow_run *run, uint64_t length, float timestep,
    m3_tensor_view *embedding, m3_error *error)
{
    const uint64_t inner = (uint64_t)run->config->attention_heads *
                           run->config->head_dimension;
    const uint64_t fourier_shape[] = {
        2U, run->config->fourier_dimension
    };
    const uint64_t time_shape[] = {2U, inner};
    m3_tensor_view fourier;
    m3_tensor_view hidden;
    m3_tensor_view activated;
    m3_tensor_view ones;
    m3_command command = {0};
    m3_status status = m3_flow_prepare_tables(
        run, length, timestep, error);

    if (status == M3_STATUS_OK) {
        status = m3_flow_view(
            run, M3_FLOW_WS_FOURIER, M3_DTYPE_F32, 2U,
            fourier_shape, &fourier, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_view(
            run, M3_FLOW_WS_TIME_HIDDEN, M3_DTYPE_F32, 2U,
            time_shape, &hidden, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_view(
            run, M3_FLOW_WS_TIME_ACTIVATED, M3_DTYPE_F32, 2U,
            time_shape, &activated, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_view(
            run, M3_FLOW_WS_TIME_EMBEDDING, M3_DTYPE_F32, 2U,
            time_shape, embedding, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_view(
            run, M3_FLOW_WS_TIME_ONES, M3_DTYPE_F32, 2U,
            time_shape, &ones, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_fill_f32(&ones, 1.0F, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_flow_model_linear(&command, &fourier,
                         run->weights->time_linear_in,
                         run->weights->time_linear_in_bias, &hidden);
    status = m3_command_executor_execute(
        &run->executor, &command, 1U, error);
    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_GATED_SILU;
    command.descriptor.gated_silu = (m3_op_binary){
        &hidden, &ones, &activated
    };
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, &command, 1U, error);
    }
    m3_flow_model_linear(&command, &activated,
                         run->weights->time_linear_out,
                         run->weights->time_linear_out_bias, embedding);
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, &command, 1U, error);
    }
    return status;
}

static m3_status m3_flow_assemble_sequence(
    m3_flow_run *run, const m3_tensor_view *projected,
    const m3_tensor_view *time_embedding, uint64_t length,
    m3_error *error)
{
    const uint64_t inner = (uint64_t)run->config->attention_heads *
                           run->config->head_dimension;
    const uint64_t hidden_shape[] = {2U, length + 1U, inner};
    const uint64_t time_shape[] = {1U, inner};
    m3_tensor_view hidden;
    m3_tensor_view hidden_rows[2];
    m3_tensor_view projected_rows[2];
    m3_tensor_view time_rows[2];
    m3_tensor_view time_tokens[2];
    m3_tensor_view time_vectors[2];
    m3_tensor_view sequence_tokens[2];
    m3_command commands[4] = {0};
    size_t row;
    m3_status status = m3_flow_view(
        run, M3_FLOW_WS_HIDDEN, M3_DTYPE_F32, 3U, hidden_shape,
        &hidden, error);

    for (row = 0U; row < 2U && status == M3_STATUS_OK; ++row) {
        m3_tensor_view_init(&time_tokens[row]);
        m3_tensor_view_init(&time_vectors[row]);
        m3_tensor_view_init(&sequence_tokens[row]);
        status = m3_flow_model_row(
            &hidden, (uint64_t)row, &hidden_rows[row], error);
        if (status == M3_STATUS_OK) {
            status = m3_flow_model_row(
                projected, (uint64_t)row, &projected_rows[row], error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_flow_model_row(
                time_embedding, (uint64_t)row, &time_rows[row], error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_tensor_slice(
                &hidden_rows[row], 1U, 0U, 1U,
                &time_tokens[row], error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_tensor_reshape(
                &time_tokens[row], 2U, time_shape,
                &time_vectors[row], error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_tensor_slice(
                &hidden_rows[row], 1U, 1U, length,
                &sequence_tokens[row], error);
        }
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    for (row = 0U; row < 2U; ++row) {
        commands[row * 2U].kind = M3_OP_COPY;
        commands[row * 2U].descriptor.copy = (m3_op_unary){
            &time_rows[row], &time_vectors[row]
        };
        commands[row * 2U + 1U].kind = M3_OP_COPY;
        commands[row * 2U + 1U].descriptor.copy = (m3_op_unary){
            &projected_rows[row], &sequence_tokens[row]
        };
    }
    return m3_command_executor_execute(
        &run->executor, commands, 4U, error);
}

static m3_status m3_flow_project_output(
    m3_flow_run *run, uint64_t length, m3_tensor_view *velocity,
    m3_error *error)
{
    const uint64_t inner = (uint64_t)run->config->attention_heads *
                           run->config->head_dimension;
    const uint64_t hidden_shape[] = {2U, length + 1U, inner};
    const uint64_t output_shape[] = {
        2U, length, run->config->latent_channels
    };
    const uint64_t channels_shape[] = {
        2U, run->config->latent_channels, length
    };
    const uint8_t to_channels[] = {0U, 2U, 1U};
    const uint8_t to_sequence[] = {0U, 2U, 1U};
    m3_tensor_view hidden;
    m3_tensor_view hidden_sequence;
    m3_tensor_view output_sequence;
    m3_tensor_view output_channels;
    m3_tensor_view postprocessed;
    m3_tensor_view postprocessed_sequence;
    m3_command command = {0};
    m3_status status = m3_flow_view(
        run, M3_FLOW_WS_HIDDEN, M3_DTYPE_F32, 3U, hidden_shape,
        &hidden, error);

    m3_tensor_view_init(&hidden_sequence);
    m3_tensor_view_init(&output_channels);
    m3_tensor_view_init(&postprocessed_sequence);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_slice(
            &hidden, 1U, 1U, length, &hidden_sequence, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_view(
            run, M3_FLOW_WS_OUTPUT_SEQUENCE, M3_DTYPE_F32, 3U,
            output_shape, &output_sequence, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_view(
            run, M3_FLOW_WS_POSTPROCESSED, M3_DTYPE_F32, 3U,
            channels_shape, &postprocessed, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_view(
            run, M3_FLOW_WS_VELOCITY, M3_DTYPE_F32, 3U,
            output_shape, velocity, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_flow_model_linear(&command, &hidden_sequence,
                         run->weights->output_projection, NULL,
                         &output_sequence);
    status = m3_command_executor_execute(
        &run->executor, &command, 1U, error);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(
            &output_sequence, to_channels, &output_channels, error);
    }
    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_CONV1D;
    command.descriptor.conv1d = (m3_op_conv1d){
        &output_channels, run->weights->postprocess_convolution, NULL,
        &postprocessed, 1U, 1U, 1U, 0U, 0U
    };
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, &command, 1U, error);
    }
    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_ADD;
    command.descriptor.add = (m3_op_binary){
        &postprocessed, &output_channels, &postprocessed
    };
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, &command, 1U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_permute(
            &postprocessed, to_sequence, &postprocessed_sequence, error);
    }
    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_COPY;
    command.descriptor.copy = (m3_op_unary){
        &postprocessed_sequence, velocity
    };
    if (status == M3_STATUS_OK) {
        status = m3_command_executor_execute(
            &run->executor, &command, 1U, error);
    }
    return status;
}

m3_status m3_flow_forward(m3_flow_run *run,
                          const m3_tensor_view *latent,
                          const m3_tensor_view *condition,
                          uint64_t length, float timestep,
                          m3_tensor_view *velocity, m3_error *error)
{
    m3_tensor_view concatenated;
    m3_tensor_view projected;
    m3_tensor_view time_embedding;
    size_t layer;
    m3_status status;

    if (run == NULL || run->config == NULL || run->weights == NULL ||
        latent == NULL || condition == NULL || velocity == NULL ||
        length == 0U || length > run->maximum_length) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow forward request is invalid");
    }
    status = m3_flow_concat_input(
        run, latent, condition, length, &concatenated, error);
    if (status == M3_STATUS_OK) {
        status = m3_flow_preprocess(
            run, &concatenated, length, &projected, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_time_embedding(
            run, length, timestep, &time_embedding, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_assemble_sequence(
            run, &projected, &time_embedding, length, error);
    }
    for (layer = 0U; status == M3_STATUS_OK &&
                    layer < run->config->layer_count; ++layer) {
        status = m3_flow_transformer_layer(
            run, &run->weights->layers[layer], length, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_project_output(
            run, length, velocity, error);
    }
    return status;
}

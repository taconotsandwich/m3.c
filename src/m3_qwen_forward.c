/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_qwen_internal.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    M3_QWEN_ACT_HIDDEN = 0,
    M3_QWEN_ACT_NORMALIZED,
    M3_QWEN_ACT_QUERY,
    M3_QWEN_ACT_KEY,
    M3_QWEN_ACT_VALUE,
    M3_QWEN_ACT_ROTATED_QUERY,
    M3_QWEN_ACT_ROTATED_KEY,
    M3_QWEN_ACT_ATTENTION,
    M3_QWEN_ACT_ATTENTION_REORDERED,
    M3_QWEN_ACT_PROJECTION,
    M3_QWEN_ACT_GATE,
    M3_QWEN_ACT_UP,
    M3_QWEN_ACT_DOWN,
    M3_QWEN_ACT_COUNT
};

enum {
    M3_QWEN_RESULT_HIDDEN = 0,
    M3_QWEN_RESULT_EOS_BF16,
    M3_QWEN_RESULT_SEMANTIC_BF16,
    M3_QWEN_RESULT_EOS_F32,
    M3_QWEN_RESULT_SEMANTIC_F32,
    M3_QWEN_RESULT_COUNT
};

static void m3_qwen_spec(m3_runtime_tensor_spec *spec, m3_dtype dtype,
                         uint8_t rank, const uint64_t *shape)
{
    uint8_t axis;

    (void)memset(spec, 0, sizeof(*spec));
    spec->dtype = dtype;
    spec->rank = rank;
    for (axis = 0U; axis < rank; ++axis) {
        spec->shape[axis] = shape[axis];
    }
    spec->alignment = 64U;
}

static m3_status m3_qwen_activation_build(
    m3_runtime_workspace *activation, m3_backend *backend,
    const m3_qwen_dimensions *dimensions, uint64_t sequence,
    m3_error *error)
{
    m3_runtime_tensor_spec specs[M3_QWEN_ACT_COUNT];
    uint64_t hidden[] = {2U, sequence, dimensions->hidden_size};
    uint64_t query[] = {2U, sequence,
                        dimensions->query_heads *
                            dimensions->head_dimension};
    uint64_t key[] = {2U, sequence,
                      dimensions->key_value_heads *
                          dimensions->head_dimension};
    uint64_t rotated_query[] = {2U, dimensions->query_heads, sequence,
                                dimensions->head_dimension};
    uint64_t rotated_key[] = {2U, dimensions->key_value_heads, sequence,
                              dimensions->head_dimension};
    uint64_t reordered[] = {2U, sequence, dimensions->query_heads,
                            dimensions->head_dimension};
    uint64_t intermediate[] = {2U, sequence,
                               dimensions->intermediate_size};

    m3_qwen_spec(&specs[M3_QWEN_ACT_HIDDEN], M3_DTYPE_BF16, 3U,
                 hidden);
    m3_qwen_spec(&specs[M3_QWEN_ACT_NORMALIZED], M3_DTYPE_BF16, 3U,
                 hidden);
    m3_qwen_spec(&specs[M3_QWEN_ACT_QUERY], M3_DTYPE_BF16, 3U, query);
    m3_qwen_spec(&specs[M3_QWEN_ACT_KEY], M3_DTYPE_BF16, 3U, key);
    m3_qwen_spec(&specs[M3_QWEN_ACT_VALUE], M3_DTYPE_BF16, 3U, key);
    m3_qwen_spec(&specs[M3_QWEN_ACT_ROTATED_QUERY], M3_DTYPE_BF16, 4U,
                 rotated_query);
    m3_qwen_spec(&specs[M3_QWEN_ACT_ROTATED_KEY], M3_DTYPE_BF16, 4U,
                 rotated_key);
    m3_qwen_spec(&specs[M3_QWEN_ACT_ATTENTION], M3_DTYPE_BF16, 4U,
                 rotated_query);
    m3_qwen_spec(&specs[M3_QWEN_ACT_ATTENTION_REORDERED],
                 M3_DTYPE_BF16, 4U, reordered);
    m3_qwen_spec(&specs[M3_QWEN_ACT_PROJECTION], M3_DTYPE_BF16, 3U,
                 hidden);
    m3_qwen_spec(&specs[M3_QWEN_ACT_GATE], M3_DTYPE_BF16, 3U,
                 intermediate);
    m3_qwen_spec(&specs[M3_QWEN_ACT_UP], M3_DTYPE_BF16, 3U,
                 intermediate);
    m3_qwen_spec(&specs[M3_QWEN_ACT_DOWN], M3_DTYPE_BF16, 3U, hidden);
    return m3_runtime_workspace_build(activation, backend, specs,
                                      M3_QWEN_ACT_COUNT, error);
}

static m3_status m3_qwen_result_build(
    m3_runtime_workspace *result, m3_backend *backend,
    const m3_qwen_dimensions *dimensions, m3_error *error)
{
    m3_runtime_tensor_spec specs[M3_QWEN_RESULT_COUNT];
    uint64_t hidden[] = {2U, dimensions->hidden_size};
    uint64_t eos[] = {2U, 1U};
    uint64_t semantic[] = {2U, dimensions->semantic_token_count};

    m3_qwen_spec(&specs[M3_QWEN_RESULT_HIDDEN], M3_DTYPE_BF16, 2U,
                 hidden);
    m3_qwen_spec(&specs[M3_QWEN_RESULT_EOS_BF16], M3_DTYPE_BF16, 2U,
                 eos);
    m3_qwen_spec(&specs[M3_QWEN_RESULT_SEMANTIC_BF16], M3_DTYPE_BF16, 2U,
                 semantic);
    m3_qwen_spec(&specs[M3_QWEN_RESULT_EOS_F32], M3_DTYPE_F32, 2U, eos);
    m3_qwen_spec(&specs[M3_QWEN_RESULT_SEMANTIC_F32], M3_DTYPE_F32, 2U,
                 semantic);
    return m3_runtime_workspace_build(result, backend, specs,
                                      M3_QWEN_RESULT_COUNT, error);
}

static m3_status m3_qwen_command(m3_command_executor *executor,
                                 m3_command *command, m3_error *error)
{
    return m3_command_executor_execute(executor, command, 1U, error);
}

static m3_status m3_qwen_embedding(m3_command_executor *executor,
                                   const m3_tensor_view *ids,
                                   const m3_tensor_view *table,
                                   m3_tensor_view *output,
                                   m3_error *error)
{
    m3_command command;

    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_EMBEDDING;
    command.descriptor.embedding.ids = ids;
    command.descriptor.embedding.table = table;
    command.descriptor.embedding.output = output;
    return m3_qwen_command(executor, &command, error);
}

static m3_status m3_qwen_forward_copy(m3_command_executor *executor,
                                      const m3_tensor_view *input,
                                      m3_tensor_view *output,
                                      m3_error *error)
{
    m3_command command;

    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_COPY;
    command.descriptor.copy.input = input;
    command.descriptor.copy.output = output;
    return m3_qwen_command(executor, &command, error);
}

static m3_status m3_qwen_forward_rms(m3_command_executor *executor,
                                     const m3_tensor_view *input,
                                     const m3_tensor_view *scale,
                                     m3_tensor_view *output, float epsilon,
                                     m3_error *error)
{
    m3_command command;

    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_RMS_NORM;
    command.descriptor.rms_norm.input = input;
    command.descriptor.rms_norm.scale = scale;
    command.descriptor.rms_norm.output = output;
    command.descriptor.rms_norm.epsilon = epsilon;
    return m3_qwen_command(executor, &command, error);
}

static m3_status m3_qwen_forward_linear(m3_command_executor *executor,
                                        const m3_tensor_view *input,
                                        const m3_tensor_view *weight,
                                        m3_tensor_view *output,
                                        m3_error *error)
{
    m3_command command;

    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_LINEAR;
    command.descriptor.linear.input = input;
    command.descriptor.linear.weight = weight;
    command.descriptor.linear.output = output;
    return m3_qwen_command(executor, &command, error);
}

static m3_status m3_qwen_forward_cast(m3_command_executor *executor,
                                      const m3_tensor_view *input,
                                      m3_tensor_view *output,
                                      m3_error *error)
{
    m3_command command;

    (void)memset(&command, 0, sizeof(command));
    command.kind = M3_OP_CAST;
    command.descriptor.cast.input = input;
    command.descriptor.cast.output = output;
    return m3_qwen_command(executor, &command, error);
}

static bool m3_qwen_progress(m3_progress_callback progress, void *context,
                             uint64_t completed, uint64_t total)
{
    return progress == NULL || progress(context, completed, total);
}

static m3_status m3_qwen_cancelled(m3_error *error)
{
    return m3_error_set(error, M3_STATUS_CANCELLED,
                        "Qwen forward execution was cancelled");
}

static m3_status m3_qwen_feedback_rows_equal(
    const m3_tensor_view *feedback, m3_error *error)
{
    const void *raw = NULL;
    const uint8_t *data;
    size_t channel;
    m3_status status = m3_tensor_const_data(feedback, &raw, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (raw == NULL) {
        return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                            "Qwen feedback rows are not host-readable");
    }
    data = raw;
    for (channel = 0U;
         channel < (size_t)feedback->metadata.shape[2]; ++channel) {
        size_t first = channel * feedback->byte_strides[2];
        size_t second = feedback->byte_strides[0] + first;

        if (memcmp(data + first, data + second, sizeof(uint16_t)) != 0) {
            return m3_error_set(
                error, M3_STATUS_INVALID_ARGUMENT,
                "Qwen feedback rows must be bit-identical");
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_qwen_forward_validate(
    const m3_qwen_forward_state *state, m3_qwen_forward_kind kind,
    const m3_tensor_view *input, uint64_t *sequence, uint64_t *position,
    uint64_t *new_count,
    m3_error *error)
{
    if (state == NULL || state->dimensions == NULL ||
        state->weights == NULL || state->backend == NULL ||
        state->executor == NULL || state->cache_views == NULL ||
        state->cosines == NULL || state->sines == NULL || input == NULL ||
        sequence == NULL || position == NULL || new_count == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen forward state and inputs are required");
    }
    if (kind == M3_QWEN_FORWARD_PREFILL) {
        const int32_t *values;
        const void *data = NULL;
        size_t index;
        m3_status status;

        if (input->metadata.dtype != M3_DTYPE_I32 ||
            input->metadata.rank != 2U ||
            input->metadata.shape[0] != 2U ||
            input->metadata.shape[1] == 0U ||
            !m3_tensor_is_contiguous(input) || input->storage == NULL ||
            m3_storage_backend(input->storage) != state->backend) {
            return m3_error_set(
                error, M3_STATUS_INVALID_ARGUMENT,
                "Qwen IDs must be contiguous I32 [2,T] on the runtime backend");
        }
        if (state->token_count != 0U) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "Qwen prefill requires an empty cache state");
        }
        status = m3_tensor_const_data(input, &data, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        if (data == NULL) {
            return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                                "Qwen IDs are not host-readable");
        }
        values = data;
        for (index = 0U; index < input->metadata.element_count; ++index) {
            if (values[index] < 0 ||
                (uint64_t)values[index] >=
                    state->dimensions->vocab_size) {
                return m3_error_set(
                    error, M3_STATUS_OUT_OF_RANGE,
                    "Qwen token ID is outside the vocabulary");
            }
        }
        *sequence = input->metadata.shape[1];
        *position = 0U;
    } else if (kind == M3_QWEN_FORWARD_ADVANCE) {
        m3_status status;

        if (input->metadata.dtype != M3_DTYPE_BF16 ||
            input->metadata.rank != 3U ||
            input->metadata.shape[0] != 2U ||
            input->metadata.shape[1] != 1U ||
            input->metadata.shape[2] != state->dimensions->hidden_size ||
            input->storage == NULL ||
            m3_storage_backend(input->storage) != state->backend) {
            return m3_error_set(
                error, M3_STATUS_INVALID_ARGUMENT,
                "Qwen feedback must be BF16 [2,1,H] on the runtime "
                "backend");
        }
        status = m3_qwen_feedback_rows_equal(input, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        if (state->token_count == 0U) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "Qwen advance requires a prefilled state");
        }
        *sequence = 1U;
        *position = state->token_count;
    } else {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen forward kind is invalid");
    }
    if (*sequence > UINT64_MAX - *position) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Qwen published token count overflows");
    }
    *new_count = *position + *sequence;
    if (*new_count > state->cache_capacity) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "Qwen forward exceeds cache capacity");
    }
    return M3_STATUS_OK;
}

static void m3_qwen_layer_workspace_set(
    m3_qwen_layer_workspace *layer, m3_runtime_workspace *activation)
{
    layer->hidden = &activation->views[M3_QWEN_ACT_HIDDEN];
    layer->normalized = &activation->views[M3_QWEN_ACT_NORMALIZED];
    layer->query = &activation->views[M3_QWEN_ACT_QUERY];
    layer->key = &activation->views[M3_QWEN_ACT_KEY];
    layer->value = &activation->views[M3_QWEN_ACT_VALUE];
    layer->rotated_query =
        &activation->views[M3_QWEN_ACT_ROTATED_QUERY];
    layer->rotated_key = &activation->views[M3_QWEN_ACT_ROTATED_KEY];
    layer->attention = &activation->views[M3_QWEN_ACT_ATTENTION];
    layer->attention_reordered =
        &activation->views[M3_QWEN_ACT_ATTENTION_REORDERED];
    layer->projection = &activation->views[M3_QWEN_ACT_PROJECTION];
    layer->gate = &activation->views[M3_QWEN_ACT_GATE];
    layer->up = &activation->views[M3_QWEN_ACT_UP];
    layer->down = &activation->views[M3_QWEN_ACT_DOWN];
}

static m3_status m3_qwen_final_hidden(
    m3_qwen_forward_state *state, m3_runtime_workspace *activation,
    m3_runtime_workspace *result, m3_tensor_view *last_hidden,
    m3_error *error)
{
    m3_tensor_view last_sequence;
    m3_tensor_view result_sequence;
    uint64_t sequence =
        activation->views[M3_QWEN_ACT_HIDDEN].metadata.shape[1];
    uint64_t shape[] = {2U, 1U, state->dimensions->hidden_size};
    m3_status status;

    m3_tensor_view_init(&last_sequence);
    m3_tensor_view_init(&result_sequence);
    m3_tensor_view_init(last_hidden);
    status = m3_qwen_forward_rms(
        state->executor, &activation->views[M3_QWEN_ACT_HIDDEN],
        state->weights->final_norm,
        &activation->views[M3_QWEN_ACT_NORMALIZED],
        state->dimensions->rms_epsilon, error);

    if (status == M3_STATUS_OK) {
        status = m3_tensor_slice(
            &activation->views[M3_QWEN_ACT_NORMALIZED], 1U,
            sequence - 1U, 1U, &last_sequence, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_reshape(
            &result->views[M3_QWEN_RESULT_HIDDEN], 3U, shape,
            &result_sequence, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_forward_copy(
            state->executor, &last_sequence, &result_sequence, error);
    }
    if (status == M3_STATUS_OK) {
        *last_hidden = result->views[M3_QWEN_RESULT_HIDDEN];
    }
    return status;
}

static m3_status m3_qwen_head(
    m3_qwen_forward_state *state, const m3_tensor_view *last_hidden,
    m3_runtime_workspace *result, m3_error *error)
{
    m3_tensor_view eos_weight;
    m3_tensor_view semantic_weight;
    m3_status status;

    m3_tensor_view_init(&eos_weight);
    m3_tensor_view_init(&semantic_weight);
    status = m3_tensor_slice(
        state->weights->head, 0U, state->dimensions->eos_token_id, 1U,
        &eos_weight, error);

    if (status == M3_STATUS_OK) {
        status = m3_tensor_slice(
            state->weights->head, 0U,
            state->dimensions->semantic_token_start,
            state->dimensions->semantic_token_count, &semantic_weight,
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_forward_linear(
            state->executor, last_hidden, &eos_weight,
            &result->views[M3_QWEN_RESULT_EOS_BF16], error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_forward_linear(
            state->executor, last_hidden, &semantic_weight,
            &result->views[M3_QWEN_RESULT_SEMANTIC_BF16], error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_forward_cast(
            state->executor, &result->views[M3_QWEN_RESULT_EOS_BF16],
            &result->views[M3_QWEN_RESULT_EOS_F32], error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_forward_cast(
            state->executor,
            &result->views[M3_QWEN_RESULT_SEMANTIC_BF16],
            &result->views[M3_QWEN_RESULT_SEMANTIC_F32], error);
    }
    return status;
}

void m3_qwen_forward_state_init(m3_qwen_forward_state *state)
{
    if (state != NULL) {
        (void)memset(state, 0, sizeof(*state));
        m3_runtime_workspace_init(&state->published);
    }
}

void m3_qwen_forward_state_dispose(m3_qwen_forward_state *state)
{
    if (state == NULL) {
        return;
    }
    m3_runtime_workspace_dispose(&state->published);
    m3_qwen_forward_state_init(state);
}

m3_status m3_qwen_forward_execute(m3_qwen_forward_state *state,
                                  m3_qwen_forward_kind kind,
                                  const m3_tensor_view *input,
                                  m3_progress_callback progress,
                                  void *progress_context,
                                  m3_qwen_forward_result *result,
                                  m3_error *error)
{
    m3_runtime_workspace activation;
    m3_runtime_workspace built_result;
    m3_qwen_layer_workspace layer_workspace;
    m3_tensor_view last_hidden;
    uint64_t sequence = 0U;
    uint64_t position = 0U;
    uint64_t new_count = 0U;
    uint64_t total;
    uint64_t layer;
    m3_status status;

    if (result == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen forward result is null");
    }
    status = m3_qwen_forward_validate(state, kind, input, &sequence,
                                      &position, &new_count, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    total = state->dimensions->layer_count + 3U;
    if (!m3_qwen_progress(progress, progress_context, 0U, total)) {
        return m3_qwen_cancelled(error);
    }
    m3_runtime_workspace_init(&activation);
    m3_runtime_workspace_init(&built_result);
    status = m3_qwen_activation_build(
        &activation, state->backend, state->dimensions,
        sequence, error);
    if (status == M3_STATUS_OK) {
        status = m3_qwen_result_build(&built_result, state->backend,
                                      state->dimensions, error);
    }
    if (status == M3_STATUS_OK && kind == M3_QWEN_FORWARD_PREFILL) {
        status = m3_qwen_embedding(
            state->executor, input, state->weights->embedding,
            &activation.views[M3_QWEN_ACT_HIDDEN], error);
    } else if (status == M3_STATUS_OK) {
        status = m3_qwen_forward_copy(
            state->executor, input,
            &activation.views[M3_QWEN_ACT_HIDDEN], error);
    }
    if (status == M3_STATUS_OK &&
        !m3_qwen_progress(progress, progress_context, 1U, total)) {
        status = m3_qwen_cancelled(error);
    }
    m3_qwen_layer_workspace_set(&layer_workspace, &activation);
    for (layer = 0U; layer < state->dimensions->layer_count &&
                     status == M3_STATUS_OK;
         ++layer) {
        status = m3_qwen_layer_execute(
            state->executor, state->dimensions,
            &state->weights->layers[layer], &layer_workspace,
            &state->cache_views[layer * 2U],
            &state->cache_views[layer * 2U + 1U], state->cosines,
            state->sines, position, error);
        if (status == M3_STATUS_OK &&
            !m3_qwen_progress(progress, progress_context, layer + 2U,
                              total)) {
            status = m3_qwen_cancelled(error);
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_final_hidden(state, &activation, &built_result,
                                      &last_hidden, error);
    }
    if (status == M3_STATUS_OK &&
        !m3_qwen_progress(progress, progress_context,
                          state->dimensions->layer_count + 2U, total)) {
        status = m3_qwen_cancelled(error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_qwen_head(state, &last_hidden, &built_result, error);
    }
    if (status == M3_STATUS_OK &&
        !m3_qwen_progress(progress, progress_context, total, total)) {
        status = m3_qwen_cancelled(error);
    }
    m3_runtime_workspace_dispose(&activation);
    if (status != M3_STATUS_OK) {
        m3_runtime_workspace_dispose(&built_result);
        return status;
    }
    m3_runtime_workspace_dispose(&state->published);
    state->published = built_result;
    state->token_count = new_count;
    result->hidden = &state->published.views[M3_QWEN_RESULT_HIDDEN];
    result->eos_logits =
        &state->published.views[M3_QWEN_RESULT_EOS_F32];
    result->semantic_logits =
        &state->published.views[M3_QWEN_RESULT_SEMANTIC_F32];
    m3_error_reset(error);
    return M3_STATUS_OK;
}

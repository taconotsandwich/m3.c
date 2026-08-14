/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"

#include <math.h>
#include <stdint.h>

static m3_status m3_preflight_float_to_i32(const m3_op_unary *op,
                                            m3_error *error)
{
    size_t index;

    if (op->output->metadata.dtype != M3_DTYPE_I32 ||
        op->input->metadata.dtype == M3_DTYPE_I32) {
        return M3_STATUS_OK;
    }
    for (index = 0U; index < op->input->metadata.element_count; ++index) {
        float value = m3_op_load_float(
            op->input, m3_op_element_offset(op->input, index));

        if (!isfinite(value) || value < -2147483648.0F ||
            value >= 2147483648.0F) {
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                "float-to-I32 cast value is invalid");
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_preflight_embedding(const m3_op_embedding *op,
                                        m3_error *error)
{
    size_t vocabulary = (size_t)op->table->metadata.shape[0];
    size_t token;

    for (token = 0U; token < op->ids->metadata.element_count; ++token) {
        int32_t id = m3_op_load_i32(
            op->ids, m3_op_element_offset(op->ids, token));

        if (id < 0 || (size_t)id >= vocabulary) {
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                "embedding ID is outside the vocabulary");
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_preflight_nan_values(const m3_tensor_view *view,
                                         const char *message,
                                         m3_error *error)
{
    size_t index;

    for (index = 0U; index < view->metadata.element_count; ++index) {
        float value = m3_op_load_float(
            view, m3_op_element_offset(view, index));

        if (isnan(value)) {
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE, "%s",
                                message);
        }
    }
    return M3_STATUS_OK;
}

static bool m3_attention_causal_allowed(size_t query, size_t key,
                                        int64_t offset)
{
    uint64_t query_value = (uint64_t)query;
    uint64_t key_value = (uint64_t)key;

    if (offset >= 0) {
        uint64_t positive = (uint64_t)offset;

        return query_value > UINT64_MAX - positive ||
               key_value <= query_value + positive;
    }
    {
        uint64_t magnitude = (uint64_t)(-(offset + 1)) + 1U;

        return query_value >= magnitude &&
               key_value <= query_value - magnitude;
    }
}

size_t m3_op_attention_offset(const m3_tensor_view *view, size_t batch,
                              size_t head, size_t position, size_t depth)
{
    return view->byte_offset + batch * view->byte_strides[0] +
           head * view->byte_strides[1] +
           position * view->byte_strides[2] +
           depth * view->byte_strides[3];
}

static float m3_attention_mask(const m3_tensor_view *mask, size_t batch,
                               size_t head, size_t query, size_t key)
{
    size_t offset = mask->byte_offset;

    if (mask->metadata.shape[0] != 1U) {
        offset += batch * mask->byte_strides[0];
    }
    if (mask->metadata.shape[1] != 1U) {
        offset += head * mask->byte_strides[1];
    }
    if (mask->metadata.shape[2] != 1U) {
        offset += query * mask->byte_strides[2];
    }
    offset += key * mask->byte_strides[3];
    return m3_op_load_float(mask, offset);
}

float m3_op_attention_score(const m3_op_attention *op, size_t batch,
                            size_t query_head, size_t kv_head,
                            size_t query, size_t key, size_t depth)
{
    float sum = 0.0F;
    size_t channel;

    if (op->causal &&
        !m3_attention_causal_allowed(query, key, op->causal_offset)) {
        return -INFINITY;
    }
    for (channel = 0U; channel < depth; ++channel) {
        float query_value = m3_op_load_float(
            op->query,
            m3_op_attention_offset(op->query, batch, query_head, query,
                                   channel));
        float key_value = m3_op_load_float(
            op->key,
            m3_op_attention_offset(op->key, batch, kv_head, key,
                                   channel));
        float product = query_value * key_value;

        sum = sum + product;
    }
    sum = sum * op->scale;
    if (op->mask != NULL) {
        sum = sum + m3_attention_mask(op->mask, batch, query_head, query,
                                      key);
    }
    return sum;
}

static m3_status m3_preflight_attention_finite(
    const m3_tensor_view *view, bool allow_negative_infinity,
    const char *name, m3_error *error)
{
    size_t index;

    for (index = 0U; index < view->metadata.element_count; ++index) {
        float value = m3_op_load_float(
            view, m3_op_element_offset(view, index));

        if (!isfinite(value) &&
            !(allow_negative_infinity && value == -INFINITY)) {
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                "%s contains an invalid non-finite value",
                                name);
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_preflight_attention_scores(
    const m3_op_attention *op, size_t batch_count, size_t query_heads,
    size_t kv_heads, size_t query_count, size_t key_count, size_t depth,
    m3_error *error)
{
    size_t heads_per_kv = query_heads / kv_heads;
    size_t batch;

    for (batch = 0U; batch < batch_count; ++batch) {
        size_t query_head;

        for (query_head = 0U; query_head < query_heads; ++query_head) {
            size_t kv_head = query_head / heads_per_kv;
            size_t query;

            for (query = 0U; query < query_count; ++query) {
                size_t key;

                for (key = 0U; key < key_count; ++key) {
                    float score = m3_op_attention_score(
                        op, batch, query_head, kv_head, query, key, depth);

                    if (isnan(score)) {
                        return m3_error_set(
                            error, M3_STATUS_OUT_OF_RANGE,
                            "attention score is NaN after accumulation");
                    }
                }
            }
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_preflight_attention(const m3_op_attention *op,
                                         m3_error *error)
{
    size_t batch_count = (size_t)op->query->metadata.shape[0];
    size_t query_heads = (size_t)op->query->metadata.shape[1];
    size_t query_count = (size_t)op->query->metadata.shape[2];
    size_t depth = (size_t)op->query->metadata.shape[3];
    size_t kv_heads = (size_t)op->key->metadata.shape[1];
    size_t key_count = (size_t)op->key->metadata.shape[2];
    m3_status status = m3_preflight_attention_finite(
        op->query, false, "attention query", error);

    if (status == M3_STATUS_OK) {
        status = m3_preflight_attention_finite(
            op->key, false, "attention key", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_preflight_attention_finite(
            op->value, false, "attention value", error);
    }
    if (status == M3_STATUS_OK && op->mask != NULL) {
        status = m3_preflight_attention_finite(
            op->mask, true, "attention mask", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_preflight_attention_scores(
            op, batch_count, query_heads, kv_heads, query_count, key_count,
            depth, error);
    }
    return status;
}

static m3_status m3_preflight_categorical(const m3_op_categorical *op,
                                           m3_error *error)
{
    size_t vocabulary = (size_t)op->probabilities->metadata.shape[
        op->probabilities->metadata.rank - 1U];
    size_t rows = op->uniforms->metadata.element_count;
    size_t row;

    for (row = 0U; row < rows; ++row) {
        float uniform = m3_op_load_float(
            op->uniforms, m3_op_element_offset(op->uniforms, row));
        float sum = 0.0F;
        size_t index;

        if (!isfinite(uniform) || uniform < 0.0F || uniform >= 1.0F) {
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                "categorical uniform is outside [0,1)");
        }
        for (index = 0U; index < vocabulary; ++index) {
            float probability = m3_op_load_float(
                op->probabilities,
                m3_op_element_offset(
                    op->probabilities, row * vocabulary + index));

            if (!isfinite(probability) || probability < 0.0F) {
                return m3_error_set(
                    error, M3_STATUS_OUT_OF_RANGE,
                    "categorical probability is invalid");
            }
            sum = sum + probability;
        }
        if (!isfinite(sum) || sum <= 0.0F) {
            return m3_error_set(
                error, M3_STATUS_OUT_OF_RANGE,
                "categorical row sum is not positive finite");
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_command_data_preflight(const m3_command *command,
                                    m3_error *error)
{
    switch (command->kind) {
    case M3_OP_CAST:
        return m3_preflight_float_to_i32(&command->descriptor.cast,
                                         error);
    case M3_OP_EMBEDDING:
        return m3_preflight_embedding(&command->descriptor.embedding,
                                      error);
    case M3_OP_SOFTMAX:
        return m3_preflight_nan_values(
            command->descriptor.softmax.input,
            "softmax input contains NaN", error);
    case M3_OP_ATTENTION:
        return m3_preflight_attention(&command->descriptor.attention,
                                      error);
    case M3_OP_TOP_K:
        return m3_preflight_nan_values(
            command->descriptor.top_k.logits,
            "top-k logits contain NaN", error);
    case M3_OP_CATEGORICAL:
        return m3_preflight_categorical(
            &command->descriptor.categorical, error);
    default:
        return M3_STATUS_OK;
    }
}

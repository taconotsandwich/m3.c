/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"

#include <math.h>
#include <stdint.h>

static m3_status m3_host_attention_finite(
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

static bool m3_host_attention_causal_allowed(size_t query, size_t key,
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

static size_t m3_host_attention_offset(const m3_tensor_view *view,
                                       size_t batch, size_t head,
                                       size_t position, size_t depth)
{
    return view->byte_offset + batch * view->byte_strides[0] +
           head * view->byte_strides[1] +
           position * view->byte_strides[2] +
           depth * view->byte_strides[3];
}

static float m3_host_attention_mask(const m3_tensor_view *mask,
                                    size_t batch, size_t head,
                                    size_t query, size_t key)
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

static float m3_host_attention_score(const m3_op_attention *op,
                                     size_t batch, size_t query_head,
                                     size_t kv_head, size_t query,
                                     size_t key, size_t depth)
{
    float sum = 0.0F;
    size_t channel;

    if (op->causal &&
        !m3_host_attention_causal_allowed(query, key,
                                          op->causal_offset)) {
        return -INFINITY;
    }
    for (channel = 0U; channel < depth; ++channel) {
        float query_value = m3_op_load_float(
            op->query,
            m3_host_attention_offset(op->query, batch, query_head, query,
                                     channel));
        float key_value = m3_op_load_float(
            op->key,
            m3_host_attention_offset(op->key, batch, kv_head, key,
                                     channel));
        float product = query_value * key_value;

        sum = sum + product;
    }
    sum = sum * op->scale;
    if (op->mask != NULL) {
        sum = sum + m3_host_attention_mask(op->mask, batch, query_head,
                                           query, key);
    }
    return sum;
}

static m3_status m3_host_attention_preflight_scores(
    const m3_op_attention *op, size_t batch_count, size_t query_heads,
    size_t kv_heads, size_t query_count, size_t key_count, size_t depth,
    m3_error *error)
{
    size_t batch;
    size_t heads_per_kv = query_heads / kv_heads;

    for (batch = 0U; batch < batch_count; ++batch) {
        size_t query_head;

        for (query_head = 0U; query_head < query_heads; ++query_head) {
            size_t kv_head = query_head / heads_per_kv;
            size_t query;

            for (query = 0U; query < query_count; ++query) {
                size_t key;

                for (key = 0U; key < key_count; ++key) {
                    float score = m3_host_attention_score(
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

static void m3_host_attention_softmax(float *scores, size_t count)
{
    float maximum = -INFINITY;
    float sum = 0.0F;
    size_t positive_infinities = 0U;
    size_t index;

    for (index = 0U; index < count; ++index) {
        if (scores[index] > maximum) {
            maximum = scores[index];
        }
        if (scores[index] == INFINITY) {
            ++positive_infinities;
        }
    }
    if (maximum == -INFINITY) {
        for (index = 0U; index < count; ++index) {
            scores[index] = 0.0F;
        }
        return;
    }
    if (maximum == INFINITY) {
        for (index = 0U; index < count; ++index) {
            scores[index] = scores[index] == INFINITY
                                ? 1.0F / (float)positive_infinities
                                : 0.0F;
        }
        return;
    }
    for (index = 0U; index < count; ++index) {
        scores[index] = expf(scores[index] - maximum);
        sum = sum + scores[index];
    }
    for (index = 0U; index < count; ++index) {
        scores[index] /= sum;
    }
}

static m3_status m3_host_attention(const m3_op_attention *op,
                                   m3_scratch_arena *scratch,
                                   m3_error *error)
{
    size_t batch_count = (size_t)op->query->metadata.shape[0];
    size_t query_heads = (size_t)op->query->metadata.shape[1];
    size_t query_count = (size_t)op->query->metadata.shape[2];
    size_t depth = (size_t)op->query->metadata.shape[3];
    size_t kv_heads = (size_t)op->key->metadata.shape[1];
    size_t key_count = (size_t)op->key->metadata.shape[2];
    size_t heads_per_kv = query_heads / kv_heads;
    float *scores = NULL;
    size_t batch;
    m3_status status;

    status = m3_host_attention_finite(op->query, false,
                                      "attention query", error);
    if (status == M3_STATUS_OK) {
        status = m3_host_attention_finite(op->key, false,
                                          "attention key", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_host_attention_finite(op->value, false,
                                          "attention value", error);
    }
    if (status == M3_STATUS_OK && op->mask != NULL) {
        status = m3_host_attention_finite(op->mask, true,
                                          "attention mask", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_host_attention_preflight_scores(
            op, batch_count, query_heads, kv_heads, query_count, key_count,
            depth, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    status = m3_scratch_arena_allocate(
        scratch, key_count * sizeof(*scores), _Alignof(float),
        (void **)&scores, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    for (batch = 0U; batch < batch_count; ++batch) {
        size_t query_head;

        for (query_head = 0U; query_head < query_heads; ++query_head) {
            size_t kv_head = query_head / heads_per_kv;
            size_t query;

            for (query = 0U; query < query_count; ++query) {
                size_t key;
                size_t channel;

                for (key = 0U; key < key_count; ++key) {
                    scores[key] = m3_host_attention_score(
                        op, batch, query_head, kv_head, query, key, depth);
                }
                m3_host_attention_softmax(scores, key_count);
                for (key = 0U; key < key_count; ++key) {
                    scores[key] = m3_op_round_float(
                        op->query->metadata.dtype, scores[key]);
                }
                for (channel = 0U; channel < depth; ++channel) {
                    float sum = 0.0F;

                    for (key = 0U; key < key_count; ++key) {
                        float value = m3_op_load_float(
                            op->value,
                            m3_host_attention_offset(
                                op->value, batch, kv_head, key, channel));
                        float product = scores[key] * value;

                        sum = sum + product;
                    }
                    m3_op_store_float(
                        op->output,
                        m3_host_attention_offset(
                            op->output, batch, query_head, query, channel),
                        sum);
                }
            }
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_host_execute_attention(const m3_command *command,
                                    m3_scratch_arena *scratch,
                                    bool *handled, m3_error *error)
{
    if (command->kind != M3_OP_ATTENTION) {
        *handled = false;
        return M3_STATUS_OK;
    }
    *handled = true;
    return m3_host_attention(&command->descriptor.attention, scratch,
                             error);
}

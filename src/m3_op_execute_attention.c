/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"

#include <math.h>
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
    m3_status status = m3_scratch_arena_allocate(
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
                    scores[key] = m3_op_attention_score(
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
                            m3_op_attention_offset(
                                op->value, batch, kv_head, key, channel));
                        float product = scores[key] * value;

                        sum = sum + product;
                    }
                    m3_op_store_float(
                        op->output,
                        m3_op_attention_offset(
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

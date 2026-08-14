/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"

#include <math.h>
#include <stddef.h>

static bool m3_host_top_precedes(float left_value, int32_t left_index,
                                 float right_value, int32_t right_index)
{
    return left_value > right_value ||
           (left_value == right_value && left_index < right_index);
}

static m3_status m3_host_top_k(const m3_op_top_k *op,
                               m3_scratch_arena *scratch,
                               m3_error *error)
{
    size_t vocabulary = (size_t)op->logits->metadata.shape[
        op->logits->metadata.rank - 1U];
    size_t rows = op->logits->metadata.element_count / vocabulary;
    size_t k = (size_t)op->k;
    m3_top_pair *pairs = NULL;
    size_t index;
    size_t row;
    m3_status status;

    for (index = 0U; index < op->logits->metadata.element_count; ++index) {
        float value = m3_op_load_float(
            op->logits, m3_op_element_offset(op->logits, index));

        if (isnan(value)) {
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                "top-k logits contain NaN");
        }
    }
    status = m3_scratch_arena_allocate(
        scratch, k * sizeof(*pairs), _Alignof(m3_top_pair),
        (void **)&pairs, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    for (row = 0U; row < rows; ++row) {
        size_t pair_count = 0U;
        size_t vocabulary_index;

        for (vocabulary_index = 0U; vocabulary_index < vocabulary;
             ++vocabulary_index) {
            float value = m3_op_load_float(
                op->logits,
                m3_op_element_offset(
                    op->logits, row * vocabulary + vocabulary_index));
            int32_t integer_index = (int32_t)vocabulary_index;
            size_t position = 0U;

            while (position < pair_count &&
                   !m3_host_top_precedes(
                       value, integer_index, pairs[position].value,
                       pairs[position].index)) {
                ++position;
            }
            if (position < k) {
                size_t move = pair_count < k ? pair_count : k - 1U;

                while (move > position) {
                    pairs[move] = pairs[move - 1U];
                    --move;
                }
                pairs[position].value = value;
                pairs[position].index = integer_index;
                if (pair_count < k) {
                    ++pair_count;
                }
            }
        }
        for (index = 0U; index < k; ++index) {
            size_t output_flat = row * k + index;

            m3_op_store_float(
                op->values,
                m3_op_element_offset(op->values, output_flat),
                pairs[index].value);
            m3_op_store_i32(
                op->indices,
                m3_op_element_offset(op->indices, output_flat),
                pairs[index].index);
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_host_categorical_preflight(
    const m3_op_categorical *op, size_t rows, size_t vocabulary,
    m3_error *error)
{
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
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                "categorical row sum is not positive finite");
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_host_categorical(const m3_op_categorical *op,
                                     m3_error *error)
{
    size_t vocabulary = (size_t)op->probabilities->metadata.shape[
        op->probabilities->metadata.rank - 1U];
    size_t rows = op->uniforms->metadata.element_count;
    size_t row;
    m3_status status = m3_host_categorical_preflight(
        op, rows, vocabulary, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    for (row = 0U; row < rows; ++row) {
        float uniform = m3_op_load_float(
            op->uniforms, m3_op_element_offset(op->uniforms, row));
        float sum = 0.0F;
        float cumulative = 0.0F;
        float target;
        size_t index;
        size_t selected = vocabulary - 1U;

        for (index = 0U; index < vocabulary; ++index) {
            sum = sum + m3_op_load_float(
                            op->probabilities,
                            m3_op_element_offset(
                                op->probabilities,
                                row * vocabulary + index));
        }
        target = uniform * sum;
        for (index = 0U; index < vocabulary; ++index) {
            cumulative = cumulative + m3_op_load_float(
                                          op->probabilities,
                                          m3_op_element_offset(
                                              op->probabilities,
                                              row * vocabulary + index));
            if (target < cumulative) {
                selected = index;
                break;
            }
        }
        m3_op_store_i32(op->output,
                        m3_op_element_offset(op->output, row),
                        (int32_t)selected);
    }
    return M3_STATUS_OK;
}

m3_status m3_host_execute_sampling(const m3_command *command,
                                   m3_scratch_arena *scratch,
                                   bool *handled, m3_error *error)
{
    *handled = true;
    switch (command->kind) {
    case M3_OP_TOP_K:
        return m3_host_top_k(&command->descriptor.top_k, scratch, error);
    case M3_OP_CATEGORICAL:
        return m3_host_categorical(&command->descriptor.categorical,
                                   error);
    default:
        *handled = false;
        return M3_STATUS_OK;
    }
}

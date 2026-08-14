/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"

#include "m3_backend.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static m3_status m3_host_copy(const m3_op_unary *op)
{
    const uint8_t *input = m3_storage_const_data(op->input->storage);
    uint8_t *output = m3_storage_data(op->output->storage);
    size_t element_size = m3_dtype_size(op->input->metadata.dtype);
    size_t index;

    for (index = 0U; index < op->input->metadata.element_count; ++index) {
        size_t input_offset = m3_op_element_offset(op->input, index);
        size_t output_offset = m3_op_element_offset(op->output, index);

        (void)memmove(output + output_offset, input + input_offset,
                      element_size);
    }
    return M3_STATUS_OK;
}

static bool m3_host_float_i32_valid(float value)
{
    return isfinite(value) && value >= -2147483648.0F &&
           value < 2147483648.0F;
}

static m3_status m3_host_cast(const m3_op_unary *op, m3_error *error)
{
    size_t index;

    if (op->input->metadata.dtype == op->output->metadata.dtype) {
        return m3_host_copy(op);
    }
    if (op->output->metadata.dtype == M3_DTYPE_I32 &&
        op->input->metadata.dtype != M3_DTYPE_I32) {
        for (index = 0U; index < op->input->metadata.element_count; ++index) {
            float value = m3_op_load_float(
                op->input, m3_op_element_offset(op->input, index));

            if (!m3_host_float_i32_valid(value)) {
                return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                    "float-to-I32 cast value is invalid");
            }
        }
    }
    for (index = 0U; index < op->input->metadata.element_count; ++index) {
        size_t input_offset = m3_op_element_offset(op->input, index);
        size_t output_offset = m3_op_element_offset(op->output, index);

        if (op->input->metadata.dtype == M3_DTYPE_I32) {
            int32_t integer = m3_op_load_i32(op->input, input_offset);

            if (op->output->metadata.dtype == M3_DTYPE_I32) {
                m3_op_store_i32(op->output, output_offset, integer);
            } else {
                m3_op_store_float(op->output, output_offset,
                                  (float)integer);
            }
        } else {
            float value = m3_op_load_float(op->input, input_offset);

            if (op->output->metadata.dtype == M3_DTYPE_I32) {
                m3_op_store_i32(op->output, output_offset, (int32_t)value);
            } else {
                m3_op_store_float(op->output, output_offset, value);
            }
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_host_binary(const m3_op_binary *op, bool multiply)
{
    size_t index;

    for (index = 0U; index < op->output->metadata.element_count; ++index) {
        float left = m3_op_load_float(
            op->left, m3_op_broadcast_offset(op->left, op->output, index));
        float right = m3_op_load_float(
            op->right,
            m3_op_broadcast_offset(op->right, op->output, index));
        float result = multiply ? left * right : left + right;

        m3_op_store_float(op->output,
                          m3_op_element_offset(op->output, index), result);
    }
    return M3_STATUS_OK;
}

static m3_status m3_host_embedding(const m3_op_embedding *op,
                                   m3_error *error)
{
    size_t token;
    size_t channels = (size_t)op->table->metadata.shape[1];
    size_t vocabulary = (size_t)op->table->metadata.shape[0];

    for (token = 0U; token < op->ids->metadata.element_count; ++token) {
        int32_t id = m3_op_load_i32(
            op->ids, m3_op_element_offset(op->ids, token));

        if (id < 0 || (size_t)id >= vocabulary) {
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                "embedding ID is outside the vocabulary");
        }
    }
    for (token = 0U; token < op->ids->metadata.element_count; ++token) {
        int32_t id = m3_op_load_i32(
            op->ids, m3_op_element_offset(op->ids, token));
        size_t channel;

        for (channel = 0U; channel < channels; ++channel) {
            size_t table_offset = op->table->byte_offset +
                                  (size_t)id * op->table->byte_strides[0] +
                                  channel * op->table->byte_strides[1];
            size_t output_index = token * channels + channel;
            float value = m3_op_load_float(op->table, table_offset);

            m3_op_store_float(
                op->output,
                m3_op_element_offset(op->output, output_index), value);
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_host_gated_silu(const m3_op_binary *op)
{
    size_t index;

    for (index = 0U; index < op->output->metadata.element_count; ++index) {
        size_t gate_offset = m3_op_element_offset(op->left, index);
        size_t up_offset = m3_op_element_offset(op->right, index);
        float gate = m3_op_load_float(op->left, gate_offset);
        float up = m3_op_load_float(op->right, up_offset);
        float activation = m3_op_round_float(
            op->left->metadata.dtype,
            gate / (1.0F + expf(-gate)));
        float result = activation * up;

        m3_op_store_float(op->output,
                          m3_op_element_offset(op->output, index), result);
    }
    return M3_STATUS_OK;
}

static m3_status m3_host_softmax(const m3_op_unary *op, m3_error *error)
{
    size_t total = op->input->metadata.element_count;
    size_t width = (size_t)op->input->metadata.shape[
        op->input->metadata.rank - 1U];
    size_t row;
    size_t index;

    for (index = 0U; index < total; ++index) {
        float value = m3_op_load_float(
            op->input, m3_op_element_offset(op->input, index));

        if (isnan(value)) {
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                "softmax input contains NaN");
        }
    }
    for (row = 0U; row < total / width; ++row) {
        float maximum = -INFINITY;
        float sum = 0.0F;
        size_t positive_infinities = 0U;
        size_t column;

        for (column = 0U; column < width; ++column) {
            float value = m3_op_load_float(
                op->input,
                m3_op_element_offset(op->input, row * width + column));

            if (value > maximum) {
                maximum = value;
            }
            if (value == INFINITY) {
                ++positive_infinities;
            }
        }
        if (maximum != -INFINITY && maximum != INFINITY) {
            for (column = 0U; column < width; ++column) {
                float value = m3_op_load_float(
                    op->input,
                    m3_op_element_offset(op->input,
                                         row * width + column));

                sum += expf(value - maximum);
            }
        }
        for (column = 0U; column < width; ++column) {
            size_t flat = row * width + column;
            float value = m3_op_load_float(
                op->input, m3_op_element_offset(op->input, flat));
            float probability;

            if (maximum == -INFINITY) {
                probability = 0.0F;
            } else if (maximum == INFINITY) {
                probability = value == INFINITY
                                  ? 1.0F / (float)positive_infinities
                                  : 0.0F;
            } else {
                probability = expf(value - maximum) / sum;
            }
            m3_op_store_float(op->output,
                              m3_op_element_offset(op->output, flat),
                              probability);
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_host_execute_basic(const m3_command *command,
                                m3_scratch_arena *scratch,
                                bool *handled, m3_error *error)
{
    (void)scratch;
    *handled = true;
    switch (command->kind) {
    case M3_OP_COPY:
        return m3_host_copy(&command->descriptor.copy);
    case M3_OP_CAST:
        return m3_host_cast(&command->descriptor.cast, error);
    case M3_OP_ADD:
        return m3_host_binary(&command->descriptor.add, false);
    case M3_OP_MUL:
        return m3_host_binary(&command->descriptor.mul, true);
    case M3_OP_EMBEDDING:
        return m3_host_embedding(&command->descriptor.embedding, error);
    case M3_OP_GATED_SILU:
        return m3_host_gated_silu(&command->descriptor.gated_silu);
    case M3_OP_SOFTMAX:
        return m3_host_softmax(&command->descriptor.softmax, error);
    default:
        *handled = false;
        return M3_STATUS_OK;
    }
}

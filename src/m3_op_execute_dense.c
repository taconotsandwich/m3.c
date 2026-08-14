/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"

#include <math.h>
#include <stddef.h>

static float m3_host_dot_product(const m3_tensor_view *left,
                                 size_t left_base, size_t left_stride,
                                 const m3_tensor_view *right,
                                 size_t right_base, size_t right_stride,
                                 size_t count)
{
    float sum = 0.0F;
    size_t index;

    for (index = 0U; index < count; ++index) {
        float left_value = m3_op_load_float(
            left, left_base + index * left_stride);
        float right_value = m3_op_load_float(
            right, right_base + index * right_stride);
        float product = left_value * right_value;

        sum = sum + product;
    }
    return sum;
}

static m3_status m3_host_matmul(const m3_op_matmul *op)
{
    size_t rows = (size_t)op->left->metadata.shape[0];
    size_t inner = (size_t)op->left->metadata.shape[1];
    size_t columns = (size_t)op->right->metadata.shape[1];
    size_t row;

    for (row = 0U; row < rows; ++row) {
        size_t column;

        for (column = 0U; column < columns; ++column) {
            size_t left_base = op->left->byte_offset +
                               row * op->left->byte_strides[0];
            size_t right_base = op->right->byte_offset +
                                column * op->right->byte_strides[1];
            float sum = m3_host_dot_product(
                op->left, left_base, op->left->byte_strides[1], op->right,
                right_base, op->right->byte_strides[0], inner);
            size_t output_flat = row * columns + column;

            m3_op_store_float(
                op->output, m3_op_element_offset(op->output, output_flat),
                sum);
        }
    }
    return M3_STATUS_OK;
}

static size_t m3_host_linear_rows(const m3_tensor_view *input)
{
    size_t rows = 1U;
    uint8_t axis;

    for (axis = 0U; axis + 1U < input->metadata.rank; ++axis) {
        rows *= (size_t)input->metadata.shape[axis];
    }
    return rows;
}

static m3_status m3_host_linear(const m3_op_linear *op)
{
    uint8_t input_axis = op->input->metadata.rank - 1U;
    size_t input_features = (size_t)op->weight->metadata.shape[1];
    size_t output_features = (size_t)op->weight->metadata.shape[0];
    size_t rows = m3_host_linear_rows(op->input);
    size_t row;

    for (row = 0U; row < rows; ++row) {
        size_t output_feature;

        for (output_feature = 0U; output_feature < output_features;
             ++output_feature) {
            size_t input_base = m3_op_element_offset(
                op->input, row * input_features);
            size_t weight_base = op->weight->byte_offset +
                                 output_feature *
                                     op->weight->byte_strides[0];
            float sum = 0.0F;
            size_t input_feature;

            if (op->bias != NULL) {
                size_t bias_offset = op->bias->byte_offset +
                                     output_feature *
                                         op->bias->byte_strides[0];

                sum = m3_op_load_float(op->bias, bias_offset);
            }
            for (input_feature = 0U; input_feature < input_features;
                 ++input_feature) {
                float input_value = m3_op_load_float(
                    op->input,
                    input_base + input_feature *
                                     op->input->byte_strides[input_axis]);
                float weight_value = m3_op_load_float(
                    op->weight,
                    weight_base + input_feature *
                                      op->weight->byte_strides[1]);
                float product = input_value * weight_value;

                sum = sum + product;
            }
            m3_op_store_float(
                op->output,
                m3_op_element_offset(
                    op->output, row * output_features + output_feature),
                sum);
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_host_rms_norm(const m3_op_rms_norm *op)
{
    size_t channels = (size_t)op->input->metadata.shape[
        op->input->metadata.rank - 1U];
    size_t rows = op->input->metadata.element_count / channels;
    size_t row;

    for (row = 0U; row < rows; ++row) {
        float square_sum = 0.0F;
        float inverse_rms;
        size_t channel;

        for (channel = 0U; channel < channels; ++channel) {
            size_t flat = row * channels + channel;
            float value = m3_op_load_float(
                op->input, m3_op_element_offset(op->input, flat));
            float square = value * value;

            square_sum = square_sum + square;
        }
        inverse_rms = 1.0F /
                      sqrtf(square_sum / (float)channels + op->epsilon);
        for (channel = 0U; channel < channels; ++channel) {
            size_t flat = row * channels + channel;
            float value = m3_op_load_float(
                op->input, m3_op_element_offset(op->input, flat));
            float scale = m3_op_load_float(
                op->scale, op->scale->byte_offset +
                               channel * op->scale->byte_strides[0]);
            float normalized = m3_op_round_float(
                op->input->metadata.dtype, value * inverse_rms);
            float result = normalized * scale;

            m3_op_store_float(op->output,
                              m3_op_element_offset(op->output, flat),
                              result);
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_host_layer_norm(const m3_op_layer_norm *op)
{
    size_t channels = (size_t)op->input->metadata.shape[
        op->input->metadata.rank - 1U];
    size_t rows = op->input->metadata.element_count / channels;
    size_t row;

    for (row = 0U; row < rows; ++row) {
        float sum = 0.0F;
        float variance_sum = 0.0F;
        float mean;
        float inverse_standard_deviation;
        size_t channel;

        for (channel = 0U; channel < channels; ++channel) {
            size_t flat = row * channels + channel;

            sum = sum + m3_op_load_float(
                            op->input,
                            m3_op_element_offset(op->input, flat));
        }
        mean = sum / (float)channels;
        for (channel = 0U; channel < channels; ++channel) {
            size_t flat = row * channels + channel;
            float value = m3_op_load_float(
                op->input, m3_op_element_offset(op->input, flat));
            float centered = value - mean;
            float square = centered * centered;

            variance_sum = variance_sum + square;
        }
        inverse_standard_deviation =
            1.0F /
            sqrtf(variance_sum / (float)channels + op->epsilon);
        for (channel = 0U; channel < channels; ++channel) {
            size_t flat = row * channels + channel;
            size_t vector_offset = op->scale->byte_offset +
                                   channel * op->scale->byte_strides[0];
            float value = m3_op_load_float(
                op->input, m3_op_element_offset(op->input, flat));
            float scale = m3_op_load_float(op->scale, vector_offset);
            float result = ((value - mean) * inverse_standard_deviation) *
                           scale;

            if (op->bias != NULL) {
                size_t bias_offset = op->bias->byte_offset +
                                     channel * op->bias->byte_strides[0];

                result = result + m3_op_load_float(op->bias, bias_offset);
            }
            m3_op_store_float(op->output,
                              m3_op_element_offset(op->output, flat),
                              result);
        }
    }
    return M3_STATUS_OK;
}

static void m3_host_rope_pair(const m3_op_rope *op, size_t input_first,
                              size_t input_second, size_t output_first,
                              size_t output_second, float cosine,
                              float sine)
{
    float first = m3_op_load_float(op->input, input_first);
    float second = m3_op_load_float(op->input, input_second);
    float rotated_first = first * cosine - second * sine;
    float rotated_second = second * cosine + first * sine;

    m3_op_store_float(op->output, output_first, rotated_first);
    m3_op_store_float(op->output, output_second, rotated_second);
}

static m3_status m3_host_rope(const m3_op_rope *op)
{
    size_t batch_count = (size_t)op->input->metadata.shape[0];
    size_t head_count = (size_t)op->input->metadata.shape[1];
    size_t sequence = (size_t)op->input->metadata.shape[2];
    size_t depth = (size_t)op->input->metadata.shape[3];
    size_t rotary = (size_t)op->rotary_dimension;
    size_t batch;

    for (batch = 0U; batch < batch_count; ++batch) {
        size_t head;

        for (head = 0U; head < head_count; ++head) {
            size_t position;

            for (position = 0U; position < sequence; ++position) {
                size_t vector = ((batch * head_count + head) * sequence +
                                 position) *
                                depth;
                size_t table_row = (size_t)op->position_offset + position;
                size_t pair;

                for (pair = 0U; pair < rotary / 2U; ++pair) {
                    size_t first = op->mode == M3_ROPE_HALF_SPLIT
                                       ? pair
                                       : pair * 2U;
                    size_t second = op->mode == M3_ROPE_HALF_SPLIT
                                        ? pair + rotary / 2U
                                        : pair * 2U + 1U;
                    size_t cosine_offset = op->cosines->byte_offset +
                                           table_row *
                                               op->cosines->byte_strides[0] +
                                           pair *
                                               op->cosines->byte_strides[1];
                    size_t sine_offset = op->sines->byte_offset +
                                         table_row *
                                             op->sines->byte_strides[0] +
                                         pair * op->sines->byte_strides[1];

                    m3_host_rope_pair(
                        op, m3_op_element_offset(op->input, vector + first),
                        m3_op_element_offset(op->input, vector + second),
                        m3_op_element_offset(op->output, vector + first),
                        m3_op_element_offset(op->output, vector + second),
                        m3_op_load_float(op->cosines, cosine_offset),
                        m3_op_load_float(op->sines, sine_offset));
                }
                for (pair = rotary; pair < depth; ++pair) {
                    float value = m3_op_load_float(
                        op->input,
                        m3_op_element_offset(op->input, vector + pair));

                    m3_op_store_float(
                        op->output,
                        m3_op_element_offset(op->output, vector + pair),
                        value);
                }
            }
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_host_execute_dense(const m3_command *command,
                                m3_scratch_arena *scratch,
                                bool *handled, m3_error *error)
{
    (void)scratch;
    (void)error;
    *handled = true;
    switch (command->kind) {
    case M3_OP_MATMUL:
        return m3_host_matmul(&command->descriptor.matmul);
    case M3_OP_LINEAR:
        return m3_host_linear(&command->descriptor.linear);
    case M3_OP_RMS_NORM:
        return m3_host_rms_norm(&command->descriptor.rms_norm);
    case M3_OP_LAYER_NORM:
        return m3_host_layer_norm(&command->descriptor.layer_norm);
    case M3_OP_ROPE:
        return m3_host_rope(&command->descriptor.rope);
    default:
        *handled = false;
        return M3_STATUS_OK;
    }
}

/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"

#include <stddef.h>
#include <stdint.h>

static float m3_host_convolution_bias(const m3_tensor_view *bias,
                                      size_t channel)
{
    if (bias == NULL) {
        return 0.0F;
    }
    return m3_op_load_float(
        bias, bias->byte_offset + channel * bias->byte_strides[0]);
}

static size_t m3_host_conv_output_offset(m3_tensor_view *output,
                                         size_t batch, size_t channel,
                                         size_t position)
{
    size_t channels = (size_t)output->metadata.shape[1];
    size_t length = (size_t)output->metadata.shape[2];
    size_t flat = (batch * channels + channel) * length + position;

    return m3_op_element_offset(output, flat);
}

static float m3_host_conv1d_accumulate(const m3_op_conv1d *op,
                                       size_t batch,
                                       size_t output_channel,
                                       size_t output_position)
{
    size_t input_channels = (size_t)op->input->metadata.shape[1];
    size_t input_length = (size_t)op->input->metadata.shape[2];
    size_t output_channels = (size_t)op->output->metadata.shape[1];
    size_t kernel = (size_t)op->weight->metadata.shape[2];
    size_t inputs_per_group = input_channels / (size_t)op->groups;
    size_t outputs_per_group = output_channels / (size_t)op->groups;
    size_t group = output_channel / outputs_per_group;
    size_t input_start = group * inputs_per_group;
    uint64_t output_base = (uint64_t)output_position * op->stride;
    float sum = m3_host_convolution_bias(op->bias, output_channel);
    size_t input_in_group;

    for (input_in_group = 0U; input_in_group < inputs_per_group;
         ++input_in_group) {
        size_t input_channel = input_start + input_in_group;
        size_t kernel_index;

        for (kernel_index = 0U; kernel_index < kernel; ++kernel_index) {
            uint64_t padded_position =
                output_base + (uint64_t)kernel_index * op->dilation;
            uint64_t input_position;
            size_t input_offset;
            size_t weight_offset;
            float input_value;
            float weight_value;
            float product;

            if (padded_position < op->pad_left) {
                continue;
            }
            input_position = padded_position - op->pad_left;
            if (input_position >= (uint64_t)input_length) {
                continue;
            }
            input_offset =
                op->input->byte_offset +
                batch * op->input->byte_strides[0] +
                input_channel * op->input->byte_strides[1] +
                (size_t)input_position * op->input->byte_strides[2];
            weight_offset =
                op->weight->byte_offset +
                output_channel * op->weight->byte_strides[0] +
                input_in_group * op->weight->byte_strides[1] +
                kernel_index * op->weight->byte_strides[2];
            input_value = m3_op_load_float(op->input, input_offset);
            weight_value = m3_op_load_float(op->weight, weight_offset);
            product = input_value * weight_value;
            sum = sum + product;
        }
    }
    return sum;
}

static m3_status m3_host_conv1d(const m3_op_conv1d *op)
{
    size_t batches = (size_t)op->input->metadata.shape[0];
    size_t output_channels = (size_t)op->output->metadata.shape[1];
    size_t output_length = (size_t)op->output->metadata.shape[2];
    size_t batch;

    for (batch = 0U; batch < batches; ++batch) {
        size_t output_channel;

        for (output_channel = 0U; output_channel < output_channels;
             ++output_channel) {
            size_t output_position;

            for (output_position = 0U; output_position < output_length;
                 ++output_position) {
                float sum = m3_host_conv1d_accumulate(
                    op, batch, output_channel, output_position);

                m3_op_store_float(
                    op->output,
                    m3_host_conv_output_offset(
                        op->output, batch, output_channel, output_position),
                    sum);
            }
        }
    }
    return M3_STATUS_OK;
}

static float m3_host_conv_transpose1d_accumulate(
    const m3_op_conv_transpose1d *op, size_t batch,
    size_t output_channel, size_t output_position)
{
    size_t input_channels = (size_t)op->input->metadata.shape[1];
    size_t input_length = (size_t)op->input->metadata.shape[2];
    size_t output_channels = (size_t)op->output->metadata.shape[1];
    size_t kernel = (size_t)op->weight->metadata.shape[2];
    size_t inputs_per_group = input_channels / (size_t)op->groups;
    size_t outputs_per_group = output_channels / (size_t)op->groups;
    size_t group = output_channel / outputs_per_group;
    size_t input_start = group * inputs_per_group;
    size_t output_in_group = output_channel - group * outputs_per_group;
    uint64_t padded_output =
        (uint64_t)output_position + op->pad_left;
    float sum = m3_host_convolution_bias(op->bias, output_channel);
    size_t input_in_group;

    for (input_in_group = 0U; input_in_group < inputs_per_group;
         ++input_in_group) {
        size_t input_channel = input_start + input_in_group;
        size_t kernel_index;

        for (kernel_index = 0U; kernel_index < kernel; ++kernel_index) {
            uint64_t kernel_position =
                (uint64_t)kernel_index * op->dilation;
            uint64_t distance;
            uint64_t input_position;
            size_t input_offset;
            size_t weight_offset;
            float input_value;
            float weight_value;
            float product;

            if (kernel_position > padded_output) {
                continue;
            }
            distance = padded_output - kernel_position;
            if (distance % op->stride != 0U) {
                continue;
            }
            input_position = distance / op->stride;
            if (input_position >= (uint64_t)input_length) {
                continue;
            }
            input_offset =
                op->input->byte_offset +
                batch * op->input->byte_strides[0] +
                input_channel * op->input->byte_strides[1] +
                (size_t)input_position * op->input->byte_strides[2];
            weight_offset =
                op->weight->byte_offset +
                input_channel * op->weight->byte_strides[0] +
                output_in_group * op->weight->byte_strides[1] +
                kernel_index * op->weight->byte_strides[2];
            input_value = m3_op_load_float(op->input, input_offset);
            weight_value = m3_op_load_float(op->weight, weight_offset);
            product = input_value * weight_value;
            sum = sum + product;
        }
    }
    return sum;
}

static m3_status m3_host_conv_transpose1d(
    const m3_op_conv_transpose1d *op)
{
    size_t batches = (size_t)op->input->metadata.shape[0];
    size_t output_channels = (size_t)op->output->metadata.shape[1];
    size_t output_length = (size_t)op->output->metadata.shape[2];
    size_t batch;

    for (batch = 0U; batch < batches; ++batch) {
        size_t output_channel;

        for (output_channel = 0U; output_channel < output_channels;
             ++output_channel) {
            size_t output_position;

            for (output_position = 0U; output_position < output_length;
                 ++output_position) {
                float sum = m3_host_conv_transpose1d_accumulate(
                    op, batch, output_channel, output_position);

                m3_op_store_float(
                    op->output,
                    m3_host_conv_output_offset(
                        op->output, batch, output_channel, output_position),
                    sum);
            }
        }
    }
    return M3_STATUS_OK;
}

static void m3_host_resize_advance(uint64_t *source, uint64_t *remainder,
                                   uint64_t quotient,
                                   uint64_t remainder_step,
                                   uint64_t output_length)
{
    *source += quotient;
    if (remainder_step != 0U) {
        uint64_t threshold = output_length - remainder_step;

        if (*remainder >= threshold) {
            *remainder -= threshold;
            ++*source;
        } else {
            *remainder += remainder_step;
        }
    }
}

static m3_status m3_host_nearest_resize1d(const m3_op_unary *op)
{
    size_t batches = (size_t)op->input->metadata.shape[0];
    size_t channels = (size_t)op->input->metadata.shape[1];
    uint64_t input_length = op->input->metadata.shape[2];
    uint64_t output_length = op->output->metadata.shape[2];
    uint64_t quotient = input_length / output_length;
    uint64_t remainder_step = input_length % output_length;
    size_t batch;

    for (batch = 0U; batch < batches; ++batch) {
        size_t channel;

        for (channel = 0U; channel < channels; ++channel) {
            uint64_t source = 0U;
            uint64_t remainder = 0U;
            uint64_t output_position;

            for (output_position = 0U; output_position < output_length;
                 ++output_position) {
                size_t input_offset =
                    op->input->byte_offset +
                    batch * op->input->byte_strides[0] +
                    channel * op->input->byte_strides[1] +
                    (size_t)source * op->input->byte_strides[2];
                size_t output_offset = m3_host_conv_output_offset(
                    op->output, batch, channel,
                    (size_t)output_position);
                float value = m3_op_load_float(op->input, input_offset);

                m3_op_store_float(op->output, output_offset, value);
                if (output_position + 1U < output_length) {
                    m3_host_resize_advance(
                        &source, &remainder, quotient, remainder_step,
                        output_length);
                }
            }
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_host_execute_convolution(const m3_command *command,
                                      m3_scratch_arena *scratch,
                                      bool *handled, m3_error *error)
{
    (void)scratch;
    (void)error;
    *handled = true;
    switch (command->kind) {
    case M3_OP_CONV1D:
        return m3_host_conv1d(&command->descriptor.conv1d);
    case M3_OP_CONV_TRANSPOSE1D:
        return m3_host_conv_transpose1d(
            &command->descriptor.conv_transpose1d);
    case M3_OP_NEAREST_RESIZE1D:
        return m3_host_nearest_resize1d(
            &command->descriptor.nearest_resize1d);
    default:
        *handled = false;
        return M3_STATUS_OK;
    }
}

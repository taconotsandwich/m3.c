/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"

#include <stdint.h>

static bool m3_op_u64_add(uint64_t left, uint64_t right,
                          uint64_t *result)
{
    if (left > UINT64_MAX - right) {
        return false;
    }
    *result = left + right;
    return true;
}

static bool m3_op_u64_multiply(uint64_t left, uint64_t right,
                               uint64_t *result)
{
    if (left != 0U && right > UINT64_MAX / left) {
        return false;
    }
    *result = left * right;
    return true;
}

static m3_status m3_op_convolution_view(const m3_backend *backend,
                                        const m3_tensor_view *view,
                                        bool output, const char *name,
                                        m3_error *error)
{
    m3_status status = m3_op_check_view(backend, view, output, name,
                                        error);

    if (status == M3_STATUS_OK && !m3_op_dtype_float(view->metadata.dtype)) {
        status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                              "%s must have a floating-point dtype", name);
    }
    return status;
}

static m3_status m3_op_convolution_views(
    const m3_backend *backend, const m3_tensor_view *input,
    const m3_tensor_view *weight, const m3_tensor_view *bias,
    m3_tensor_view *output, const char *name, m3_error *error)
{
    m3_status status = m3_op_convolution_view(backend, input, false,
                                               name, error);

    if (status == M3_STATUS_OK) {
        status = m3_op_convolution_view(backend, weight, false,
                                        "convolution weight", error);
    }
    if (status == M3_STATUS_OK && bias != NULL) {
        status = m3_op_convolution_view(backend, bias, false,
                                        "convolution bias", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_op_convolution_view(backend, output, true,
                                        "convolution output", error);
    }
    return status;
}

static m3_status m3_op_convolution_aliases(
    const m3_tensor_view *input, const m3_tensor_view *weight,
    const m3_tensor_view *bias, m3_tensor_view *output, m3_error *error)
{
    m3_status status = m3_op_check_alias(output, input, false,
                                          "convolution input", error);

    if (status == M3_STATUS_OK) {
        status = m3_op_check_alias(output, weight, false,
                                   "convolution weight", error);
    }
    if (status == M3_STATUS_OK && bias != NULL) {
        status = m3_op_check_alias(output, bias, false,
                                   "convolution bias", error);
    }
    return status;
}

static m3_status m3_op_effective_kernel(uint64_t kernel,
                                         uint64_t dilation,
                                         uint64_t *effective,
                                         m3_error *error)
{
    uint64_t span;

    if (!m3_op_u64_multiply(kernel - 1U, dilation, &span) ||
        !m3_op_u64_add(span, 1U, effective)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "convolution effective kernel overflows");
    }
    return M3_STATUS_OK;
}

static m3_status m3_op_conv1d_output_length(const m3_op_conv1d *op,
                                             uint64_t input_length,
                                             uint64_t kernel,
                                             uint64_t *output_length,
                                             m3_error *error)
{
    uint64_t padded = 0U;
    uint64_t effective = 0U;
    m3_status status = m3_op_effective_kernel(kernel, op->dilation,
                                               &effective, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (!m3_op_u64_add(input_length, op->pad_left, &padded) ||
        !m3_op_u64_add(padded, op->pad_right, &padded)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "convolution padded input length overflows");
    }
    if (effective > padded) {
        return m3_error_set(
            error, M3_STATUS_INVALID_ARGUMENT,
            "convolution effective kernel exceeds padded input");
    }
    *output_length = (padded - effective) / op->stride + 1U;
    return M3_STATUS_OK;
}

static m3_status m3_op_validate_conv1d(const m3_backend *backend,
                                        const m3_op_conv1d *op,
                                        m3_error *error)
{
    uint64_t input_channels;
    uint64_t output_channels;
    uint64_t input_length;
    uint64_t kernel;
    uint64_t expected_length = 0U;
    m3_status status = m3_op_convolution_views(
        backend, op->input, op->weight, op->bias, op->output,
        "convolution input", error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (op->input->metadata.rank != 3U ||
        op->weight->metadata.rank != 3U ||
        op->output->metadata.rank != 3U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Conv1d tensors must have rank three");
    }
    input_channels = op->input->metadata.shape[1];
    output_channels = op->output->metadata.shape[1];
    input_length = op->input->metadata.shape[2];
    kernel = op->weight->metadata.shape[2];
    if (op->groups == 0U || op->stride == 0U || op->dilation == 0U ||
        input_channels == 0U || output_channels == 0U ||
        input_length == 0U || kernel == 0U ||
        input_channels % op->groups != 0U ||
        output_channels % op->groups != 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Conv1d dimensions or parameters are invalid");
    }
    if (op->weight->metadata.shape[0] != output_channels ||
        op->weight->metadata.shape[1] != input_channels / op->groups ||
        op->output->metadata.shape[0] != op->input->metadata.shape[0] ||
        (op->bias != NULL &&
         (op->bias->metadata.rank != 1U ||
          op->bias->metadata.shape[0] != output_channels))) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Conv1d channel shapes are incompatible");
    }
    status = m3_op_conv1d_output_length(op, input_length, kernel,
                                         &expected_length, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (op->output->metadata.shape[2] != expected_length) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Conv1d output length is incorrect");
    }
    return m3_op_convolution_aliases(op->input, op->weight, op->bias,
                                     op->output, error);
}

static m3_status m3_op_conv_transpose_output_length(
    const m3_op_conv_transpose1d *op, uint64_t input_length,
    uint64_t kernel, uint64_t *output_length, m3_error *error)
{
    uint64_t input_span;
    uint64_t kernel_span;
    uint64_t positive;
    uint64_t padding;

    if (!m3_op_u64_multiply(input_length - 1U, op->stride,
                            &input_span) ||
        !m3_op_u64_multiply(kernel - 1U, op->dilation, &kernel_span) ||
        !m3_op_u64_add(input_span, kernel_span, &positive) ||
        !m3_op_u64_add(positive, op->output_padding, &positive) ||
        !m3_op_u64_add(positive, 1U, &positive) ||
        !m3_op_u64_add(op->pad_left, op->pad_right, &padding)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "ConvTranspose1d output length overflows");
    }
    if (positive <= padding) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "ConvTranspose1d output length is not positive");
    }
    *output_length = positive - padding;
    return M3_STATUS_OK;
}

static m3_status m3_op_validate_conv_transpose1d(
    const m3_backend *backend, const m3_op_conv_transpose1d *op,
    m3_error *error)
{
    uint64_t input_channels;
    uint64_t output_channels;
    uint64_t input_length;
    uint64_t kernel;
    uint64_t expected_length = 0U;
    m3_status status = m3_op_convolution_views(
        backend, op->input, op->weight, op->bias, op->output,
        "transposed convolution input", error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (op->input->metadata.rank != 3U ||
        op->weight->metadata.rank != 3U ||
        op->output->metadata.rank != 3U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "ConvTranspose1d tensors must have rank three");
    }
    input_channels = op->input->metadata.shape[1];
    output_channels = op->output->metadata.shape[1];
    input_length = op->input->metadata.shape[2];
    kernel = op->weight->metadata.shape[2];
    if (op->groups == 0U || op->stride == 0U || op->dilation == 0U ||
        op->output_padding >= op->stride || input_channels == 0U ||
        output_channels == 0U || input_length == 0U || kernel == 0U ||
        input_channels % op->groups != 0U ||
        output_channels % op->groups != 0U) {
        return m3_error_set(
            error, M3_STATUS_INVALID_ARGUMENT,
            "ConvTranspose1d dimensions or parameters are invalid");
    }
    if (op->weight->metadata.shape[0] != input_channels ||
        op->weight->metadata.shape[1] != output_channels / op->groups ||
        op->output->metadata.shape[0] != op->input->metadata.shape[0] ||
        (op->bias != NULL &&
         (op->bias->metadata.rank != 1U ||
          op->bias->metadata.shape[0] != output_channels))) {
        return m3_error_set(
            error, M3_STATUS_INVALID_ARGUMENT,
            "ConvTranspose1d channel shapes are incompatible");
    }
    status = m3_op_conv_transpose_output_length(
        op, input_length, kernel, &expected_length, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (op->output->metadata.shape[2] != expected_length) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "ConvTranspose1d output length is incorrect");
    }
    return m3_op_convolution_aliases(op->input, op->weight, op->bias,
                                     op->output, error);
}

static m3_status m3_op_validate_nearest_resize1d(
    const m3_backend *backend, const m3_op_unary *op, m3_error *error)
{
    bool exact_allowed;
    m3_status status = m3_op_convolution_view(
        backend, op->input, false, "nearest resize input", error);

    if (status == M3_STATUS_OK) {
        status = m3_op_convolution_view(
            backend, op->output, true, "nearest resize output", error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (op->input->metadata.rank != 3U ||
        op->output->metadata.rank != 3U ||
        op->input->metadata.shape[0] != op->output->metadata.shape[0] ||
        op->input->metadata.shape[1] != op->output->metadata.shape[1] ||
        op->input->metadata.shape[2] == 0U ||
        op->output->metadata.shape[2] == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "nearest resize shapes must be [B,C,L]");
    }
    exact_allowed =
        op->input->metadata.dtype == op->output->metadata.dtype &&
        m3_op_shape_equal(op->input, op->output);
    return m3_op_check_alias(op->output, op->input, exact_allowed,
                             "nearest resize input", error);
}

m3_status m3_op_validate_convolution(const m3_backend *backend,
                                     const m3_command *command,
                                     bool *handled, m3_error *error)
{
    *handled = true;
    switch (command->kind) {
    case M3_OP_CONV1D:
        return m3_op_validate_conv1d(backend, &command->descriptor.conv1d,
                                     error);
    case M3_OP_CONV_TRANSPOSE1D:
        return m3_op_validate_conv_transpose1d(
            backend, &command->descriptor.conv_transpose1d, error);
    case M3_OP_NEAREST_RESIZE1D:
        return m3_op_validate_nearest_resize1d(
            backend, &command->descriptor.nearest_resize1d, error);
    default:
        *handled = false;
        return M3_STATUS_OK;
    }
}

/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_metal_internal.h"

#include "m3_backend_metal_convolution_internal.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint64_t groups;
    uint64_t stride;
    uint64_t dilation;
    uint64_t pad_left;
    uint32_t has_bias;
    uint32_t reserved;
} m3_metal_convolution_parameters;

static const char *const m3_metal_convolution_kernels[] = {
    "m3_conv1d",
    "m3_conv_transpose1d",
    "m3_nearest_resize1d"
};

static const size_t m3_metal_convolution_kernel_count =
    sizeof(m3_metal_convolution_kernels) /
    sizeof(m3_metal_convolution_kernels[0]);

_Static_assert(sizeof(m3_metal_convolution_parameters) == 40U,
               "Metal convolution parameter ABI changed");
_Static_assert(_Alignof(m3_metal_convolution_parameters) == 8U,
               "Metal convolution parameter alignment changed");

NSString *m3_metal_source_convolution(void)
{
    return m3_metal_convolution_kernel_source();
}

m3_status m3_metal_prepare_convolution(m3_metal_context *context,
                                        m3_error *error)
{
    size_t index;

    for (index = 0U; index < m3_metal_convolution_kernel_count; ++index) {
        m3_status status = m3_metal_register_pipeline(
            context, m3_metal_convolution_kernels[index], error);

        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    return M3_STATUS_OK;
}

static m3_storage *m3_metal_convolution_output_storage(
    const m3_command *command)
{
    switch (command->kind) {
    case M3_OP_CONV1D:
        return command->descriptor.conv1d.output->storage;
    case M3_OP_CONV_TRANSPOSE1D:
        return command->descriptor.conv_transpose1d.output->storage;
    case M3_OP_NEAREST_RESIZE1D:
        return command->descriptor.nearest_resize1d.output->storage;
    default:
        return NULL;
    }
}

bool m3_metal_convolution_writes_storage(const m3_command *command,
                                          const m3_storage *storage)
{
    return storage != NULL &&
           m3_metal_convolution_output_storage(command) == storage;
}

m3_status m3_metal_preflight_convolution(const m3_command *commands,
                                          size_t command_index,
                                          m3_error *error)
{
    (void)commands;
    (void)command_index;
    (void)error;
    return M3_STATUS_OK;
}

m3_status m3_metal_convolution_work(uint64_t count, size_t *work,
                                    m3_error *error)
{
    *work = 0U;
    if (count > (uint64_t)SIZE_MAX) {
        return m3_error_set(
            error, M3_STATUS_OVERFLOW,
            "Metal convolution work count overflows size_t");
    }
    if (count > UINT32_MAX) {
        return m3_error_set(
            error, M3_STATUS_OUT_OF_RANGE,
            "Metal dispatch grid exceeds 32-bit indexing");
    }
    *work = (size_t)count;
    return M3_STATUS_OK;
}

static id<MTLBuffer> m3_metal_convolution_buffer(
    const m3_tensor_view *view)
{
    return (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(view->storage);
}

static void m3_metal_convolution_bind_view(
    id<MTLComputeCommandEncoder> encoder, const m3_tensor_view *view,
    NSUInteger buffer_index, NSUInteger parameters_index)
{
    m3_metal_view_parameters parameters;

    m3_metal_view_parameters_init(&parameters, view);
    [encoder setBuffer:m3_metal_convolution_buffer(view)
                offset:0U
               atIndex:buffer_index];
    [encoder setBytes:&parameters
               length:sizeof(parameters)
              atIndex:parameters_index];
}

static void m3_metal_convolution_bind_optional(
    id<MTLComputeCommandEncoder> encoder, const m3_tensor_view *view,
    const m3_tensor_view *placeholder, NSUInteger buffer_index,
    NSUInteger parameters_index)
{
    m3_metal_view_parameters parameters;

    (void)memset(&parameters, 0, sizeof(parameters));
    if (view != NULL) {
        m3_metal_view_parameters_init(&parameters, view);
    }
    [encoder setBuffer:m3_metal_convolution_buffer(
                           view != NULL ? view : placeholder)
                offset:0U
               atIndex:buffer_index];
    [encoder setBytes:&parameters
               length:sizeof(parameters)
              atIndex:parameters_index];
}

static m3_status m3_metal_convolution_pipeline(
    const m3_metal_context *context, const char *name,
    id<MTLComputePipelineState> *pipeline, m3_error *error)
{
    *pipeline = m3_metal_find_pipeline(context, name);
    if (*pipeline == nil) {
        return m3_error_set(
            error, M3_STATUS_INTERNAL,
            "Metal convolution pipeline is unavailable");
    }
    return M3_STATUS_OK;
}

static m3_status m3_metal_convolution_require(
    const m3_tensor_view *input, const m3_tensor_view *weight,
    const m3_tensor_view *bias, const m3_tensor_view *output,
    m3_error *error)
{
    if (m3_metal_convolution_buffer(input) == nil ||
        m3_metal_convolution_buffer(weight) == nil ||
        m3_metal_convolution_buffer(output) == nil ||
        (bias != NULL && m3_metal_convolution_buffer(bias) == nil)) {
        return m3_error_set(
            error, M3_STATUS_INTERNAL,
            "Metal convolution buffers are unavailable");
    }
    return M3_STATUS_OK;
}

static m3_status m3_metal_encode_convolution_op(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_tensor_view *input,
    const m3_tensor_view *weight, const m3_tensor_view *bias,
    m3_tensor_view *output, const char *kernel, uint64_t groups,
    uint64_t stride, uint64_t dilation, uint64_t pad_left,
    m3_error *error)
{
    id<MTLComputePipelineState> pipeline;
    m3_metal_convolution_parameters parameters;
    size_t work;
    m3_status status = m3_metal_convolution_work(
        (uint64_t)output->metadata.element_count, &work, error);

    if (status != M3_STATUS_OK || work == 0U) {
        return status;
    }
    status = m3_metal_convolution_pipeline(
        context, kernel, &pipeline, error);
    if (status == M3_STATUS_OK) {
        status = m3_metal_convolution_require(
            input, weight, bias, output, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    (void)memset(&parameters, 0, sizeof(parameters));
    parameters.groups = groups;
    parameters.stride = stride;
    parameters.dilation = dilation;
    parameters.pad_left = pad_left;
    parameters.has_bias = bias != NULL ? 1U : 0U;
    m3_metal_convolution_bind_view(encoder, input, 0U, 4U);
    m3_metal_convolution_bind_view(encoder, weight, 1U, 5U);
    m3_metal_convolution_bind_optional(
        encoder, bias, input, 2U, 6U);
    m3_metal_convolution_bind_view(encoder, output, 3U, 7U);
    [encoder setBytes:&parameters
               length:sizeof(parameters)
              atIndex:8U];
    return m3_metal_dispatch_1d(encoder, pipeline, work, error);
}

static m3_status m3_metal_encode_conv1d(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_conv1d *op,
    m3_error *error)
{
    return m3_metal_encode_convolution_op(
        context, encoder, op->input, op->weight, op->bias, op->output,
        "m3_conv1d", op->groups, op->stride, op->dilation,
        op->pad_left, error);
}

static m3_status m3_metal_encode_conv_transpose1d(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder,
    const m3_op_conv_transpose1d *op, m3_error *error)
{
    return m3_metal_encode_convolution_op(
        context, encoder, op->input, op->weight, op->bias, op->output,
        "m3_conv_transpose1d", op->groups, op->stride, op->dilation,
        op->pad_left, error);
}

static m3_status m3_metal_encode_nearest_resize1d(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_unary *op,
    m3_error *error)
{
    id<MTLComputePipelineState> pipeline;
    uint64_t output_length = op->output->metadata.shape[2];
    uint64_t rows =
        (uint64_t)op->output->metadata.element_count / output_length;
    size_t work;
    m3_status status = m3_metal_convolution_work(rows, &work, error);

    if (status != M3_STATUS_OK || work == 0U) {
        return status;
    }
    status = m3_metal_convolution_pipeline(
        context, "m3_nearest_resize1d", &pipeline, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (m3_metal_convolution_buffer(op->input) == nil ||
        m3_metal_convolution_buffer(op->output) == nil) {
        return m3_error_set(
            error, M3_STATUS_INTERNAL,
            "Metal resize buffers are unavailable");
    }
    m3_metal_convolution_bind_view(encoder, op->input, 0U, 2U);
    m3_metal_convolution_bind_view(encoder, op->output, 1U, 3U);
    return m3_metal_dispatch_1d(encoder, pipeline, work, error);
}

m3_status m3_metal_encode_convolution(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_command *command,
    bool *handled, m3_error *error)
{
    *handled = true;
    switch (command->kind) {
    case M3_OP_CONV1D:
        return m3_metal_encode_conv1d(
            context, encoder, &command->descriptor.conv1d, error);
    case M3_OP_CONV_TRANSPOSE1D:
        return m3_metal_encode_conv_transpose1d(
            context, encoder, &command->descriptor.conv_transpose1d,
            error);
    case M3_OP_NEAREST_RESIZE1D:
        return m3_metal_encode_nearest_resize1d(
            context, encoder, &command->descriptor.nearest_resize1d,
            error);
    default:
        *handled = false;
        return M3_STATUS_OK;
    }
}

/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_metal_internal.h"

#include "m3_op_internal.h"

#include <math.h>
#include <string.h>

_Static_assert(sizeof(m3_metal_parameters) == 168U,
               "Metal parameter ABI changed");
_Static_assert(M3_DTYPE_F32 == 0 && M3_DTYPE_F16 == 1 &&
                   M3_DTYPE_BF16 == 2 && M3_DTYPE_I32 == 3,
               "Metal dtype ABI changed");

static const m3_op_unary *m3_metal_basic_unary(const m3_command *command)
{
    return command->kind == M3_OP_COPY ? &command->descriptor.copy
                                       : &command->descriptor.cast;
}

static bool m3_metal_float_i32_valid(float value)
{
    return isfinite(value) && value >= -2147483648.0F &&
           value < 2147483648.0F;
}

static m3_storage *m3_metal_basic_output_storage(
    const m3_command *command)
{
    if (command->kind == M3_OP_COPY) {
        return command->descriptor.copy.output->storage;
    }
    if (command->kind == M3_OP_CAST) {
        return command->descriptor.cast.output->storage;
    }
    return NULL;
}

m3_status m3_metal_preflight_basic(const m3_command *commands,
                                    size_t command_index,
                                    m3_error *error)
{
    const m3_command *command = &commands[command_index];
    const m3_op_unary *op;
    size_t prior_index;
    size_t index;

    if (command->kind != M3_OP_CAST) {
        return M3_STATUS_OK;
    }
    op = &command->descriptor.cast;
    if (op->input->metadata.dtype == M3_DTYPE_I32 ||
        op->output->metadata.dtype != M3_DTYPE_I32) {
        return M3_STATUS_OK;
    }
    for (prior_index = 0U; prior_index < command_index; ++prior_index) {
        if (m3_metal_basic_output_storage(&commands[prior_index]) ==
            op->input->storage) {
            return m3_error_set(
                error, M3_STATUS_UNSUPPORTED,
                "Metal float-to-I32 cast depends on an earlier output");
        }
    }
    for (index = 0U; index < op->input->metadata.element_count; ++index) {
        size_t offset = m3_op_element_offset(op->input, index);
        float value = m3_op_load_float(op->input, offset);

        if (!m3_metal_float_i32_valid(value)) {
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                "float-to-I32 cast value is invalid");
        }
    }
    return M3_STATUS_OK;
}

static void m3_metal_parameters_init(m3_metal_parameters *parameters,
                                     const m3_op_unary *op)
{
    uint8_t axis;

    (void)memset(parameters, 0, sizeof(*parameters));
    parameters->element_count =
        (uint64_t)op->input->metadata.element_count;
    parameters->input_byte_offset = (uint64_t)op->input->byte_offset;
    parameters->output_byte_offset = (uint64_t)op->output->byte_offset;
    parameters->rank = (uint32_t)op->input->metadata.rank;
    parameters->input_dtype = (uint32_t)op->input->metadata.dtype;
    parameters->output_dtype = (uint32_t)op->output->metadata.dtype;
    parameters->element_size =
        (uint32_t)m3_dtype_size(op->input->metadata.dtype);
    for (axis = 0U; axis < op->input->metadata.rank; ++axis) {
        parameters->input_shape[axis] = op->input->metadata.shape[axis];
        parameters->input_strides[axis] =
            (uint64_t)op->input->byte_strides[axis];
    }
}

static m3_status m3_metal_encode_unary(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_unary *op,
    bool raw_copy, m3_error *error)
{
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)(
        raw_copy ? context->copy_pipeline : context->cast_pipeline);
    id<MTLBuffer> input = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->input->storage);
    id<MTLBuffer> output = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->output->storage);
    m3_metal_parameters parameters;
    NSUInteger threads;

    if (op->input->metadata.element_count == 0U) {
        return M3_STATUS_OK;
    }
    if (op->input->metadata.element_count > UINT32_MAX) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "Metal dispatch grid exceeds 32-bit indexing");
    }
    if (pipeline == nil || input == nil || output == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal command resources are unavailable");
    }
    threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256U) {
        threads = 256U;
    }
    if (threads == 0U) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal pipeline has no dispatch capacity");
    }
    m3_metal_parameters_init(&parameters, op);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0U atIndex:0U];
    [encoder setBuffer:output offset:0U atIndex:1U];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2U];
    [encoder dispatchThreads:MTLSizeMake(
                                 (NSUInteger)op->input->metadata.element_count,
                                 1U, 1U)
          threadsPerThreadgroup:MTLSizeMake(threads, 1U, 1U)];
    return M3_STATUS_OK;
}

m3_status m3_metal_encode_basic(const m3_metal_context *context,
                                 id<MTLComputeCommandEncoder> encoder,
                                 const m3_command *command, bool *handled,
                                 m3_error *error)
{
    const m3_op_unary *op;
    bool raw_copy;

    if (command->kind != M3_OP_COPY && command->kind != M3_OP_CAST) {
        *handled = false;
        return M3_STATUS_OK;
    }
    *handled = true;
    op = m3_metal_basic_unary(command);
    raw_copy = command->kind == M3_OP_COPY ||
               op->input->metadata.dtype == op->output->metadata.dtype;
    return m3_metal_encode_unary(context, encoder, op, raw_copy, error);
}

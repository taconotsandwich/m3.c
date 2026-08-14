/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_metal_internal.h"

#include "m3_backend_metal_dense_internal.h"
#include "m3_op_internal.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

static const char *const m3_metal_dense_kernels[] = {
    "m3_embedding",
    "m3_matmul",
    "m3_matmul_zero",
    "m3_linear",
    "m3_rms_norm",
    "m3_layer_norm",
    "m3_rope",
    "m3_gated_silu"
};

static const size_t m3_metal_dense_kernel_count =
    sizeof(m3_metal_dense_kernels) /
    sizeof(m3_metal_dense_kernels[0]);

NSString *m3_metal_source_dense(void)
{
    return m3_metal_dense_kernel_source();
}

m3_status m3_metal_prepare_dense(m3_metal_context *context,
                                  m3_error *error)
{
    size_t index;

    for (index = 0U; index < m3_metal_dense_kernel_count; ++index) {
        m3_status status = m3_metal_register_pipeline(
            context, m3_metal_dense_kernels[index], error);

        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    return M3_STATUS_OK;
}

static m3_storage *m3_metal_dense_output_storage(
    const m3_command *command)
{
    switch (command->kind) {
    case M3_OP_EMBEDDING:
        return command->descriptor.embedding.output->storage;
    case M3_OP_MATMUL:
        return command->descriptor.matmul.output->storage;
    case M3_OP_LINEAR:
        return command->descriptor.linear.output->storage;
    case M3_OP_RMS_NORM:
        return command->descriptor.rms_norm.output->storage;
    case M3_OP_LAYER_NORM:
        return command->descriptor.layer_norm.output->storage;
    case M3_OP_ROPE:
        return command->descriptor.rope.output->storage;
    case M3_OP_GATED_SILU:
        return command->descriptor.gated_silu.output->storage;
    default:
        return NULL;
    }
}

bool m3_metal_dense_writes_storage(const m3_command *command,
                                    const m3_storage *storage)
{
    return storage != NULL &&
           m3_metal_dense_output_storage(command) == storage;
}

m3_status m3_metal_preflight_dense(const m3_command *commands,
                                    size_t command_index,
                                    m3_error *error)
{
    const m3_command *command = &commands[command_index];

    if (command->kind != M3_OP_EMBEDDING) {
        return M3_STATUS_OK;
    }
    if (m3_metal_has_prior_writer(
            commands, command_index,
            command->descriptor.embedding.ids->storage)) {
        return m3_error_set(
            error, M3_STATUS_UNSUPPORTED,
            "Metal embedding IDs depend on an earlier output");
    }
    return m3_command_data_preflight(command, error);
}

m3_status m3_metal_dense_work(uint64_t count, size_t *work,
                              m3_error *error)
{
    *work = 0U;
    if (count > (uint64_t)SIZE_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Metal dense work count overflows size_t");
    }
    if (count > UINT32_MAX) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "Metal dispatch grid exceeds 32-bit indexing");
    }
    *work = (size_t)count;
    return M3_STATUS_OK;
}

m3_status m3_metal_dense_product(uint64_t left, uint64_t right,
                                 size_t *work, m3_error *error)
{
    if (left != 0U && right > UINT64_MAX / left) {
        *work = 0U;
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Metal dense dimension product overflows");
    }
    return m3_metal_dense_work(left * right, work, error);
}

static id<MTLBuffer> m3_metal_dense_buffer(const m3_tensor_view *view)
{
    return (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(view->storage);
}

static void m3_metal_dense_bind_view(
    id<MTLComputeCommandEncoder> encoder, const m3_tensor_view *view,
    NSUInteger buffer_index, NSUInteger parameters_index)
{
    m3_metal_view_parameters parameters;

    m3_metal_view_parameters_init(&parameters, view);
    [encoder setBuffer:m3_metal_dense_buffer(view)
                offset:0U
               atIndex:buffer_index];
    [encoder setBytes:&parameters
               length:sizeof(parameters)
              atIndex:parameters_index];
}

static m3_status m3_metal_dense_pipeline(
    const m3_metal_context *context, const char *name, size_t work,
    id<MTLComputePipelineState> *pipeline, m3_error *error)
{
    *pipeline = nil;
    if (work == 0U) {
        return M3_STATUS_OK;
    }
    *pipeline = m3_metal_find_pipeline(context, name);
    if (*pipeline == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal dense pipeline is unavailable");
    }
    return M3_STATUS_OK;
}

static m3_status m3_metal_dense_require(
    id<MTLBuffer> first, id<MTLBuffer> second, id<MTLBuffer> output,
    m3_error *error)
{
    if (first == nil || second == nil || output == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal dense buffers are unavailable");
    }
    return M3_STATUS_OK;
}

static m3_status m3_metal_encode_embedding(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_embedding *op,
    m3_error *error)
{
    size_t work;
    id<MTLComputePipelineState> pipeline;
    m3_status status = m3_metal_dense_work(
        (uint64_t)op->output->metadata.element_count, &work, error);

    if (status != M3_STATUS_OK || work == 0U) {
        return status;
    }
    status = m3_metal_dense_pipeline(
        context, "m3_embedding", work, &pipeline, error);
    if (status == M3_STATUS_OK) {
        status = m3_metal_dense_require(
            m3_metal_dense_buffer(op->ids),
            m3_metal_dense_buffer(op->table),
            m3_metal_dense_buffer(op->output), error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_metal_dense_bind_view(encoder, op->ids, 0U, 3U);
    m3_metal_dense_bind_view(encoder, op->table, 1U, 4U);
    m3_metal_dense_bind_view(encoder, op->output, 2U, 5U);
    return m3_metal_dispatch_1d(encoder, pipeline, work, error);
}

static m3_status m3_metal_encode_matmul_zero(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_matmul *op,
    size_t work, m3_error *error)
{
    id<MTLComputePipelineState> pipeline;
    m3_status status = m3_metal_dense_pipeline(
        context, "m3_matmul_zero", work, &pipeline, error);

    if (status != M3_STATUS_OK || work == 0U) {
        return status;
    }
    if (m3_metal_dense_buffer(op->output) == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal dense output buffer is unavailable");
    }
    m3_metal_dense_bind_view(encoder, op->output, 0U, 1U);
    return m3_metal_dispatch_1d(encoder, pipeline, work, error);
}

static m3_status m3_metal_encode_matmul(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_matmul *op,
    m3_error *error)
{
    uint64_t rows = op->left->metadata.shape[0];
    uint64_t columns = op->right->metadata.shape[1];
    size_t work;
    id<MTLComputePipelineState> pipeline;
    m3_status status = m3_metal_dense_product(
        rows, columns, &work, error);

    if (status != M3_STATUS_OK || work == 0U) {
        return status;
    }
    if (op->left->metadata.shape[1] == 0U) {
        return m3_metal_encode_matmul_zero(
            context, encoder, op, work, error);
    }
    status = m3_metal_dense_pipeline(
        context, "m3_matmul", work, &pipeline, error);
    if (status == M3_STATUS_OK) {
        status = m3_metal_dense_require(
            m3_metal_dense_buffer(op->left),
            m3_metal_dense_buffer(op->right),
            m3_metal_dense_buffer(op->output), error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_metal_dense_bind_view(encoder, op->left, 0U, 3U);
    m3_metal_dense_bind_view(encoder, op->right, 1U, 4U);
    m3_metal_dense_bind_view(encoder, op->output, 2U, 5U);
    return m3_metal_dispatch_1d(encoder, pipeline, work, error);
}

static void m3_metal_dense_bind_optional(
    id<MTLComputeCommandEncoder> encoder, const m3_tensor_view *view,
    const m3_tensor_view *placeholder, NSUInteger buffer_index,
    NSUInteger parameters_index)
{
    m3_metal_view_parameters parameters;

    (void)memset(&parameters, 0, sizeof(parameters));
    if (view != NULL) {
        m3_metal_view_parameters_init(&parameters, view);
    }
    [encoder setBuffer:m3_metal_dense_buffer(
                           view != NULL ? view : placeholder)
                offset:0U
               atIndex:buffer_index];
    [encoder setBytes:&parameters
               length:sizeof(parameters)
              atIndex:parameters_index];
}

static m3_status m3_metal_encode_linear(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_linear *op,
    m3_error *error)
{
    size_t work;
    id<MTLComputePipelineState> pipeline;
    uint32_t has_bias = op->bias != NULL ? 1U : 0U;
    m3_status status = m3_metal_dense_work(
        (uint64_t)op->output->metadata.element_count, &work, error);

    if (status != M3_STATUS_OK || work == 0U) {
        return status;
    }
    status = m3_metal_dense_pipeline(
        context, "m3_linear", work, &pipeline, error);
    if (status == M3_STATUS_OK) {
        status = m3_metal_dense_require(
            m3_metal_dense_buffer(op->input),
            m3_metal_dense_buffer(op->weight),
            m3_metal_dense_buffer(op->output), error);
    }
    if (status == M3_STATUS_OK && op->bias != NULL &&
        m3_metal_dense_buffer(op->bias) == nil) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "Metal linear bias is unavailable");
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_metal_dense_bind_view(encoder, op->input, 0U, 4U);
    m3_metal_dense_bind_view(encoder, op->weight, 1U, 5U);
    m3_metal_dense_bind_optional(
        encoder, op->bias, op->output, 2U, 6U);
    m3_metal_dense_bind_view(encoder, op->output, 3U, 7U);
    [encoder setBytes:&has_bias length:sizeof(has_bias) atIndex:8U];
    return m3_metal_dispatch_1d(encoder, pipeline, work, error);
}

static m3_status m3_metal_encode_rms_norm(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_rms_norm *op,
    m3_error *error)
{
    uint64_t channels = op->input->metadata.shape[
        op->input->metadata.rank - 1U];
    uint64_t rows = (uint64_t)op->input->metadata.element_count / channels;
    size_t work;
    id<MTLComputePipelineState> pipeline;
    m3_status status = m3_metal_dense_work(rows, &work, error);

    if (status != M3_STATUS_OK || work == 0U) {
        return status;
    }
    status = m3_metal_dense_pipeline(
        context, "m3_rms_norm", work, &pipeline, error);
    if (status == M3_STATUS_OK) {
        status = m3_metal_dense_require(
            m3_metal_dense_buffer(op->input),
            m3_metal_dense_buffer(op->scale),
            m3_metal_dense_buffer(op->output), error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_metal_dense_bind_view(encoder, op->input, 0U, 3U);
    m3_metal_dense_bind_view(encoder, op->scale, 1U, 4U);
    m3_metal_dense_bind_view(encoder, op->output, 2U, 5U);
    [encoder setBytes:&op->epsilon
               length:sizeof(op->epsilon)
              atIndex:6U];
    return m3_metal_dispatch_1d(encoder, pipeline, work, error);
}

static m3_status m3_metal_encode_layer_norm(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_layer_norm *op,
    m3_error *error)
{
    uint64_t channels = op->input->metadata.shape[
        op->input->metadata.rank - 1U];
    uint64_t rows = (uint64_t)op->input->metadata.element_count / channels;
    size_t work;
    id<MTLComputePipelineState> pipeline;
    uint32_t has_bias = op->bias != NULL ? 1U : 0U;
    m3_status status = m3_metal_dense_work(rows, &work, error);

    if (status != M3_STATUS_OK || work == 0U) {
        return status;
    }
    status = m3_metal_dense_pipeline(
        context, "m3_layer_norm", work, &pipeline, error);
    if (status == M3_STATUS_OK) {
        status = m3_metal_dense_require(
            m3_metal_dense_buffer(op->input),
            m3_metal_dense_buffer(op->scale),
            m3_metal_dense_buffer(op->output), error);
    }
    if (status == M3_STATUS_OK && op->bias != NULL &&
        m3_metal_dense_buffer(op->bias) == nil) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "Metal LayerNorm bias is unavailable");
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_metal_dense_bind_view(encoder, op->input, 0U, 4U);
    m3_metal_dense_bind_view(encoder, op->scale, 1U, 5U);
    m3_metal_dense_bind_optional(
        encoder, op->bias, op->output, 2U, 6U);
    m3_metal_dense_bind_view(encoder, op->output, 3U, 7U);
    [encoder setBytes:&op->epsilon
               length:sizeof(op->epsilon)
              atIndex:8U];
    [encoder setBytes:&has_bias length:sizeof(has_bias) atIndex:9U];
    return m3_metal_dispatch_1d(encoder, pipeline, work, error);
}

static m3_status m3_metal_encode_rope(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_rope *op,
    m3_error *error)
{
    uint64_t depth = op->input->metadata.shape[3];
    uint64_t vectors =
        (uint64_t)op->input->metadata.element_count / depth;
    size_t work;
    id<MTLComputePipelineState> pipeline;
    uint32_t mode = (uint32_t)op->mode;
    m3_status status = m3_metal_dense_work(vectors, &work, error);

    if (status != M3_STATUS_OK || work == 0U) {
        return status;
    }
    status = m3_metal_dense_pipeline(
        context, "m3_rope", work, &pipeline, error);
    if (status == M3_STATUS_OK) {
        status = m3_metal_dense_require(
            m3_metal_dense_buffer(op->input),
            m3_metal_dense_buffer(op->cosines),
            m3_metal_dense_buffer(op->output), error);
    }
    if (status == M3_STATUS_OK &&
        m3_metal_dense_buffer(op->sines) == nil) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "Metal RoPE sine table is unavailable");
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_metal_dense_bind_view(encoder, op->input, 0U, 4U);
    m3_metal_dense_bind_view(encoder, op->cosines, 1U, 5U);
    m3_metal_dense_bind_view(encoder, op->sines, 2U, 6U);
    m3_metal_dense_bind_view(encoder, op->output, 3U, 7U);
    [encoder setBytes:&op->position_offset
               length:sizeof(op->position_offset)
              atIndex:8U];
    [encoder setBytes:&op->rotary_dimension
               length:sizeof(op->rotary_dimension)
              atIndex:9U];
    [encoder setBytes:&mode length:sizeof(mode) atIndex:10U];
    return m3_metal_dispatch_1d(encoder, pipeline, work, error);
}

static m3_status m3_metal_encode_gated_silu(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_binary *op,
    m3_error *error)
{
    size_t work;
    id<MTLComputePipelineState> pipeline;
    m3_status status = m3_metal_dense_work(
        (uint64_t)op->output->metadata.element_count, &work, error);

    if (status != M3_STATUS_OK || work == 0U) {
        return status;
    }
    status = m3_metal_dense_pipeline(
        context, "m3_gated_silu", work, &pipeline, error);
    if (status == M3_STATUS_OK) {
        status = m3_metal_dense_require(
            m3_metal_dense_buffer(op->left),
            m3_metal_dense_buffer(op->right),
            m3_metal_dense_buffer(op->output), error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_metal_dense_bind_view(encoder, op->left, 0U, 3U);
    m3_metal_dense_bind_view(encoder, op->right, 1U, 4U);
    m3_metal_dense_bind_view(encoder, op->output, 2U, 5U);
    return m3_metal_dispatch_1d(encoder, pipeline, work, error);
}

m3_status m3_metal_encode_dense(const m3_metal_context *context,
                                 id<MTLComputeCommandEncoder> encoder,
                                 const m3_command *command, bool *handled,
                                 m3_error *error)
{
    *handled = true;
    switch (command->kind) {
    case M3_OP_EMBEDDING:
        return m3_metal_encode_embedding(
            context, encoder, &command->descriptor.embedding, error);
    case M3_OP_MATMUL:
        return m3_metal_encode_matmul(
            context, encoder, &command->descriptor.matmul, error);
    case M3_OP_LINEAR:
        return m3_metal_encode_linear(
            context, encoder, &command->descriptor.linear, error);
    case M3_OP_RMS_NORM:
        return m3_metal_encode_rms_norm(
            context, encoder, &command->descriptor.rms_norm, error);
    case M3_OP_LAYER_NORM:
        return m3_metal_encode_layer_norm(
            context, encoder, &command->descriptor.layer_norm, error);
    case M3_OP_ROPE:
        return m3_metal_encode_rope(
            context, encoder, &command->descriptor.rope, error);
    case M3_OP_GATED_SILU:
        return m3_metal_encode_gated_silu(
            context, encoder, &command->descriptor.gated_silu, error);
    default:
        *handled = false;
        return M3_STATUS_OK;
    }
}

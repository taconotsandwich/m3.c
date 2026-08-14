/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_metal_internal.h"

#include "m3_op_internal.h"

#include <string.h>

static const char m3_metal_copy_kernel[] = "m3_copy";
static const char m3_metal_cast_kernel[] = "m3_cast";
static const char m3_metal_add_kernel[] = "m3_add";
static const char m3_metal_mul_kernel[] = "m3_mul";
static const char m3_metal_softmax_kernel[] = "m3_softmax";

NSString *m3_metal_source_basic(void)
{
    NSString *unary =
        @"struct M3BasicParams { ulong count; ulong in_offset; "
            "ulong out_offset; ulong shape[8]; ulong strides[8]; "
            "uint rank; uint in_type; uint out_type; "
            "uint element_size; };\n"
            "ulong m3_basic_input_offset("
            "constant M3BasicParams& p, ulong flat) {\n"
            "  return m3_logical_offset(flat, p.in_offset, p.shape, "
            "p.strides, p.rank);\n"
            "}\n"
            "kernel void m3_copy(device const uchar* input [[buffer(0)]], "
            "device uchar* output [[buffer(1)]], "
            "constant M3BasicParams& p [[buffer(2)]], "
            "uint gid [[thread_position_in_grid]]) {\n"
            "  if (ulong(gid) >= p.count) return; "
            "ulong src = m3_basic_input_offset(p, ulong(gid));\n"
            "  if (src == ~ulong(0)) return;\n"
            "  ulong dst = p.out_offset + ulong(gid) * "
            "ulong(p.element_size);\n"
            "  for (uint b = 0; b < p.element_size; ++b) "
            "output[dst + b] = input[src + b];\n"
            "}\n"
            "kernel void m3_cast(device const uchar* input [[buffer(0)]], "
            "device uchar* output [[buffer(1)]], "
            "constant M3BasicParams& p [[buffer(2)]], "
            "uint gid [[thread_position_in_grid]]) {\n"
            "  if (ulong(gid) >= p.count) return; "
            "ulong src = m3_basic_input_offset(p, ulong(gid));\n"
            "  if (src == ~ulong(0)) return;\n"
            "  ulong dst = p.out_offset + ulong(gid) * "
            "ulong(m3_dtype_size(p.out_type));\n"
            "  if (p.in_type == 3u) {\n"
            "    int integer = m3_load_i32(input + src);\n"
            "    if (p.out_type == 3u) { "
            "m3_store_i32(output + dst, integer); return; }\n"
            "    uint bits = as_type<uint>(float(integer));\n"
            "    m3_store_float_bits(output + dst, p.out_type, bits);\n"
            "    return;\n"
            "  }\n"
            "  uint bits = m3_load_float_bits(input + src, "
            "p.in_type);\n"
            "  if (p.out_type == 3u) m3_store_i32(output + dst, "
            "int(as_type<float>(bits)));\n"
            "  else m3_store_float_bits(output + dst, p.out_type, bits);\n"
            "}\n";

    return [unary stringByAppendingString:
        @"void m3_binary_apply(device const uchar* left, "
            "device const uchar* right, device uchar* output, "
            "constant M3ViewParameters& left_view, "
            "constant M3ViewParameters& right_view, "
            "constant M3ViewParameters& output_view, uint gid, "
            "bool multiply) {\n"
            "  ulong flat = ulong(gid);\n"
            "  ulong left_at = m3_view_broadcast_offset("
            "left_view, output_view, flat);\n"
            "  ulong right_at = m3_view_broadcast_offset("
            "right_view, output_view, flat);\n"
            "  ulong dst = m3_view_contiguous_offset(output_view, flat);\n"
            "  if (left_at == ~ulong(0) || right_at == ~ulong(0)) return;\n"
            "  if (dst == ~ulong(0)) return;\n"
            "  float a = m3_load_float(left + left_at, left_view.dtype);\n"
            "  float b = m3_load_float(right + right_at, right_view.dtype);\n"
            "  float value = multiply ? a * b : a + b;\n"
            "  m3_store_float(output + dst, output_view.dtype, value);\n"
            "}\n"
            "kernel void m3_add(device const uchar* left [[buffer(0)]], "
            "device const uchar* right [[buffer(1)]], "
            "device uchar* output [[buffer(2)]], "
            "constant M3ViewParameters& left_view [[buffer(3)]], "
            "constant M3ViewParameters& right_view [[buffer(4)]], "
            "constant M3ViewParameters& output_view [[buffer(5)]], "
            "uint gid [[thread_position_in_grid]]) {\n"
            "  m3_binary_apply(left, right, output, left_view, right_view, "
            "output_view, gid, false);\n"
            "}\n"
            "kernel void m3_mul(device const uchar* left [[buffer(0)]], "
            "device const uchar* right [[buffer(1)]], "
            "device uchar* output [[buffer(2)]], "
            "constant M3ViewParameters& left_view [[buffer(3)]], "
            "constant M3ViewParameters& right_view [[buffer(4)]], "
            "constant M3ViewParameters& output_view [[buffer(5)]], "
            "uint gid [[thread_position_in_grid]]) {\n"
            "  m3_binary_apply(left, right, output, left_view, right_view, "
            "output_view, gid, true);\n"
            "}\n"
            "kernel void m3_softmax(device const uchar* input [[buffer(0)]], "
            "device uchar* output [[buffer(1)]], "
            "constant M3ViewParameters& input_view [[buffer(2)]], "
            "constant M3ViewParameters& output_view [[buffer(3)]], "
            "uint gid [[thread_position_in_grid]]) {\n"
            "  ulong width = input_view.shape[input_view.rank - 1u];\n"
            "  ulong rows = input_view.element_count / width; "
            "ulong row = ulong(gid);\n"
            "  if (row >= rows) return; ulong begin = row * width;\n"
            "  float maximum = -INFINITY; ulong positive_infinities = 0;\n"
            "  for (ulong column = 0; column < width; ++column) {\n"
            "    ulong src = m3_view_logical_offset("
            "input_view, begin + column);\n"
            "    float value = m3_load_float("
            "input + src, input_view.dtype);\n"
            "    if (value > maximum) maximum = value;\n"
            "    if (value == INFINITY) ++positive_infinities;\n"
            "  }\n"
            "  float sum = 0.0f;\n"
            "  if (maximum != -INFINITY && maximum != INFINITY) {\n"
            "    for (ulong column = 0; column < width; ++column) {\n"
            "      ulong src = m3_view_logical_offset("
            "input_view, begin + column);\n"
            "      float value = m3_load_float("
            "input + src, input_view.dtype);\n"
            "      sum = sum + exp(value - maximum);\n"
            "    }\n"
            "  }\n"
            "  for (ulong column = 0; column < width; ++column) {\n"
            "    ulong flat = begin + column;\n"
            "    ulong src = m3_view_logical_offset(input_view, flat);\n"
            "    ulong dst = m3_view_contiguous_offset(output_view, flat);\n"
            "    float value = m3_load_float("
            "input + src, input_view.dtype);\n"
            "    float probability;\n"
            "    if (maximum == -INFINITY) probability = 0.0f;\n"
            "    else if (maximum == INFINITY) probability = "
            "value == INFINITY ? 1.0f / float(positive_infinities) : 0.0f;\n"
            "    else probability = exp(value - maximum) / sum;\n"
            "    m3_store_float("
            "output + dst, output_view.dtype, probability);\n"
            "  }\n"
            "}\n"];
}

m3_status m3_metal_prepare_basic(m3_metal_context *context,
                                  m3_error *error)
{
    m3_status status = m3_metal_register_pipeline(
        context, m3_metal_copy_kernel, error);

    if (status == M3_STATUS_OK) {
        status = m3_metal_register_pipeline(
            context, m3_metal_cast_kernel, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_metal_register_pipeline(
            context, m3_metal_add_kernel, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_metal_register_pipeline(
            context, m3_metal_mul_kernel, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_metal_register_pipeline(
            context, m3_metal_softmax_kernel, error);
    }
    return status;
}

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

static m3_storage *m3_metal_basic_output_storage(
    const m3_command *command)
{
    if (command->kind == M3_OP_COPY) {
        return command->descriptor.copy.output->storage;
    }
    if (command->kind == M3_OP_CAST) {
        return command->descriptor.cast.output->storage;
    }
    if (command->kind == M3_OP_ADD) {
        return command->descriptor.add.output->storage;
    }
    if (command->kind == M3_OP_MUL) {
        return command->descriptor.mul.output->storage;
    }
    if (command->kind == M3_OP_SOFTMAX) {
        return command->descriptor.softmax.output->storage;
    }
    return NULL;
}

bool m3_metal_basic_writes_storage(const m3_command *command,
                                    const m3_storage *storage)
{
    return storage != NULL &&
           m3_metal_basic_output_storage(command) == storage;
}

m3_status m3_metal_preflight_basic(const m3_command *commands,
                                    size_t command_index,
                                    m3_error *error)
{
    const m3_command *command = &commands[command_index];
    const m3_op_unary *op;

    if (command->kind == M3_OP_CAST) {
        op = &command->descriptor.cast;
        if (op->input->metadata.dtype == M3_DTYPE_I32 ||
            op->output->metadata.dtype != M3_DTYPE_I32) {
            return M3_STATUS_OK;
        }
    } else if (command->kind == M3_OP_SOFTMAX) {
        op = &command->descriptor.softmax;
    } else {
        return M3_STATUS_OK;
    }
    if (m3_metal_has_prior_writer(
            commands, command_index, op->input->storage)) {
        const char *message = command->kind == M3_OP_CAST
                                  ? "Metal float-to-I32 cast depends on an "
                                    "earlier output"
                                  : "Metal softmax depends on an earlier output";

        return m3_error_set(error, M3_STATUS_UNSUPPORTED, "%s", message);
    }
    return m3_command_data_preflight(command, error);
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
    id<MTLComputePipelineState> pipeline = m3_metal_find_pipeline(
        context, raw_copy ? m3_metal_copy_kernel : m3_metal_cast_kernel);
    id<MTLBuffer> input = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->input->storage);
    id<MTLBuffer> output = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->output->storage);
    m3_metal_parameters parameters;

    if (op->input->metadata.element_count == 0U) {
        return M3_STATUS_OK;
    }
    if (input == nil || output == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal command resources are unavailable");
    }
    m3_metal_parameters_init(&parameters, op);
    [encoder setBuffer:input offset:0U atIndex:0U];
    [encoder setBuffer:output offset:0U atIndex:1U];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2U];
    return m3_metal_dispatch_1d(
        encoder, pipeline, op->input->metadata.element_count, error);
}

static m3_status m3_metal_encode_binary(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_binary *op,
    bool multiply, m3_error *error)
{
    id<MTLComputePipelineState> pipeline = m3_metal_find_pipeline(
        context, multiply ? m3_metal_mul_kernel : m3_metal_add_kernel);
    id<MTLBuffer> left = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->left->storage);
    id<MTLBuffer> right = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->right->storage);
    id<MTLBuffer> output = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->output->storage);
    m3_metal_view_parameters left_parameters;
    m3_metal_view_parameters right_parameters;
    m3_metal_view_parameters output_parameters;

    if (op->output->metadata.element_count == 0U) {
        return M3_STATUS_OK;
    }
    if (left == nil || right == nil || output == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal command resources are unavailable");
    }
    m3_metal_view_parameters_init(&left_parameters, op->left);
    m3_metal_view_parameters_init(&right_parameters, op->right);
    m3_metal_view_parameters_init(&output_parameters, op->output);
    [encoder setBuffer:left offset:0U atIndex:0U];
    [encoder setBuffer:right offset:0U atIndex:1U];
    [encoder setBuffer:output offset:0U atIndex:2U];
    [encoder setBytes:&left_parameters
               length:sizeof(left_parameters)
              atIndex:3U];
    [encoder setBytes:&right_parameters
               length:sizeof(right_parameters)
              atIndex:4U];
    [encoder setBytes:&output_parameters
               length:sizeof(output_parameters)
              atIndex:5U];
    return m3_metal_dispatch_1d(
        encoder, pipeline, op->output->metadata.element_count, error);
}

static m3_status m3_metal_encode_softmax(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_unary *op,
    m3_error *error)
{
    id<MTLComputePipelineState> pipeline = m3_metal_find_pipeline(
        context, m3_metal_softmax_kernel);
    id<MTLBuffer> input = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->input->storage);
    id<MTLBuffer> output = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->output->storage);
    size_t width = (size_t)op->input->metadata.shape[
        op->input->metadata.rank - 1U];
    size_t row_count = op->input->metadata.element_count / width;
    m3_metal_view_parameters input_parameters;
    m3_metal_view_parameters output_parameters;

    if (row_count == 0U) {
        return M3_STATUS_OK;
    }
    if (input == nil || output == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal command resources are unavailable");
    }
    m3_metal_view_parameters_init(&input_parameters, op->input);
    m3_metal_view_parameters_init(&output_parameters, op->output);
    [encoder setBuffer:input offset:0U atIndex:0U];
    [encoder setBuffer:output offset:0U atIndex:1U];
    [encoder setBytes:&input_parameters
               length:sizeof(input_parameters)
              atIndex:2U];
    [encoder setBytes:&output_parameters
               length:sizeof(output_parameters)
              atIndex:3U];
    return m3_metal_dispatch_1d(
        encoder, pipeline, row_count, error);
}

m3_status m3_metal_encode_basic(const m3_metal_context *context,
                                 id<MTLComputeCommandEncoder> encoder,
                                 const m3_command *command, bool *handled,
                                 m3_error *error)
{
    if (command->kind == M3_OP_COPY || command->kind == M3_OP_CAST) {
        const m3_op_unary *op = m3_metal_basic_unary(command);
        bool raw_copy = command->kind == M3_OP_COPY ||
                        op->input->metadata.dtype ==
                            op->output->metadata.dtype;

        *handled = true;
        return m3_metal_encode_unary(context, encoder, op, raw_copy, error);
    }
    if (command->kind == M3_OP_ADD || command->kind == M3_OP_MUL) {
        const m3_op_binary *op = command->kind == M3_OP_ADD
                                     ? &command->descriptor.add
                                     : &command->descriptor.mul;

        *handled = true;
        return m3_metal_encode_binary(context, encoder, op,
                                      command->kind == M3_OP_MUL, error);
    }
    if (command->kind == M3_OP_SOFTMAX) {
        *handled = true;
        return m3_metal_encode_softmax(
            context, encoder, &command->descriptor.softmax, error);
    }
    {
        *handled = false;
        return M3_STATUS_OK;
    }
}

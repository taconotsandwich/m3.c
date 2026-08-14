/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_metal_internal.h"

#include "m3_op_internal.h"

#include <string.h>

static const char m3_metal_top_k_kernel[] = "m3_top_k";
static const char m3_metal_categorical_kernel[] = "m3_categorical";

typedef struct {
    m3_metal_view_parameters logits;
    m3_metal_view_parameters values;
    m3_metal_view_parameters indices;
    uint64_t row_count;
    uint64_t vocabulary;
    uint64_t k;
} m3_metal_top_k_parameters;

typedef struct {
    m3_metal_view_parameters probabilities;
    m3_metal_view_parameters uniforms;
    m3_metal_view_parameters output;
    uint64_t row_count;
    uint64_t vocabulary;
} m3_metal_categorical_parameters;

_Static_assert(sizeof(m3_metal_top_k_parameters) == 480U,
               "Metal top-k parameter size changed");
_Static_assert(_Alignof(m3_metal_top_k_parameters) == 8U,
               "Metal top-k parameter alignment changed");
_Static_assert(sizeof(m3_metal_categorical_parameters) == 472U,
               "Metal categorical parameter size changed");
_Static_assert(_Alignof(m3_metal_categorical_parameters) == 8U,
               "Metal categorical parameter alignment changed");
_Static_assert(M3_DTYPE_F32 == 0 && M3_DTYPE_F16 == 1 &&
                   M3_DTYPE_BF16 == 2 && M3_DTYPE_I32 == 3,
               "Metal dtype ABI changed");

NSString *m3_metal_source_sampling(void)
{
    NSString *declarations =
        @"struct M3TopKParameters {\n"
            "  M3ViewParameters logits; M3ViewParameters values;\n"
            "  M3ViewParameters indices; ulong row_count;\n"
            "  ulong vocabulary; ulong k;\n"
            "};\n"
            "static_assert(sizeof(M3TopKParameters) == 480,\n"
            "  \"Metal top-k parameter size changed\");\n"
            "struct M3CategoricalParameters {\n"
            "  M3ViewParameters probabilities; M3ViewParameters uniforms;\n"
            "  M3ViewParameters output; ulong row_count; ulong vocabulary;\n"
            "};\n"
            "static_assert(sizeof(M3CategoricalParameters) == 472,\n"
            "  \"Metal categorical parameter size changed\");\n"
            "bool m3_sampling_equal_bits(uint left, uint right) {\n"
            "  return left == right ||\n"
            "    (((left | right) & 0x7fffffffu) == 0u);\n"
            "}\n"
            "bool m3_sampling_precedes(uint left, int left_index,\n"
            "    uint right, int right_index) {\n"
            "  if (m3_sampling_equal_bits(left, right))\n"
            "    return left_index < right_index;\n"
            "  bool left_negative = (left & 0x80000000u) != 0u;\n"
            "  bool right_negative = (right & 0x80000000u) != 0u;\n"
            "  if (left_negative != right_negative) return right_negative;\n"
            "  return left_negative ? left < right : left > right;\n"
            "}\n"
            "ulong m3_sampling_shift_right_jam(ulong value, uint shift) {\n"
            "  if (shift == 0u) return value;\n"
            "  if (shift >= 64u) return value == 0ul ? 0ul : 1ul;\n"
            "  ulong mask = (ulong(1) << shift) - 1ul;\n"
            "  return (value >> shift) | ulong((value & mask) != 0ul);\n"
            "}\n";
    NSString *addition =
        @"uint m3_sampling_add_positive(uint left, uint right) {\n"
            "  left &= 0x7fffffffu; right &= 0x7fffffffu;\n"
            "  if (left == 0u) return right; if (right == 0u) return left;\n"
            "  uint left_exp = (left >> 23) & 0xffu;\n"
            "  uint right_exp = (right >> 23) & 0xffu;\n"
            "  uint left_sig = (left & 0x7fffffu) |\n"
            "    (left_exp == 0u ? 0u : 0x800000u);\n"
            "  uint right_sig = (right & 0x7fffffu) |\n"
            "    (right_exp == 0u ? 0u : 0x800000u);\n"
            "  left_exp = left_exp == 0u ? 1u : left_exp;\n"
            "  right_exp = right_exp == 0u ? 1u : right_exp;\n"
            "  if (left_exp < right_exp) {\n"
            "    uint swap_exp = left_exp; left_exp = right_exp;\n"
            "    right_exp = swap_exp; uint swap_sig = left_sig;\n"
            "    left_sig = right_sig; right_sig = swap_sig;\n"
            "  }\n"
            "  ulong extended = (ulong(left_sig) << 3) +\n"
            "    m3_sampling_shift_right_jam(ulong(right_sig) << 3,\n"
            "      left_exp - right_exp);\n"
            "  if ((extended & (ulong(1) << 27)) != 0ul) {\n"
            "    extended = m3_sampling_shift_right_jam(extended, 1u);\n"
            "    ++left_exp;\n"
            "  }\n"
            "  while (left_exp > 1u &&\n"
            "      (extended & (ulong(1) << 26)) == 0ul) {\n"
            "    extended <<= 1; --left_exp;\n"
            "  }\n"
            "  uint result = uint(extended >> 3);\n"
            "  uint remainder = uint(extended & 7ul);\n"
            "  if (remainder > 4u || (remainder == 4u &&\n"
            "      (result & 1u) != 0u)) ++result;\n"
            "  if (result >= 0x1000000u) { result >>= 1; ++left_exp; }\n"
            "  if (left_exp >= 255u) return 0x7f800000u;\n"
            "  if (left_exp == 1u && result < 0x800000u) return result;\n"
            "  return (left_exp << 23) | (result & 0x7fffffu);\n"
            "}\n";
    NSString *multiplication =
        @"ulong m3_sampling_round_shift(ulong value, uint shift) {\n"
            "  if (shift == 0u) return value;\n"
            "  if (shift >= 64u) return 0ul;\n"
            "  ulong quotient = value >> shift;\n"
            "  ulong mask = (ulong(1) << shift) - 1ul;\n"
            "  ulong remainder = value & mask;\n"
            "  ulong halfway = ulong(1) << (shift - 1u);\n"
            "  bool increment = remainder > halfway ||\n"
            "    (remainder == halfway && (quotient & 1ul) != 0ul);\n"
            "  return quotient + ulong(increment);\n"
            "}\n"
            "uint m3_sampling_multiply_positive(uint left, uint right) {\n"
            "  left &= 0x7fffffffu; right &= 0x7fffffffu;\n"
            "  if (left == 0u || right == 0u) return 0u;\n"
            "  uint left_stored_exp = (left >> 23) & 0xffu;\n"
            "  uint right_stored_exp = (right >> 23) & 0xffu;\n"
            "  int left_exp = left_stored_exp == 0u ? -126 :\n"
            "    int(left_stored_exp) - 127;\n"
            "  int right_exp = right_stored_exp == 0u ? -126 :\n"
            "    int(right_stored_exp) - 127;\n"
            "  uint left_sig = (left & 0x7fffffu) |\n"
            "    (left_stored_exp == 0u ? 0u : 0x800000u);\n"
            "  uint right_sig = (right & 0x7fffffu) |\n"
            "    (right_stored_exp == 0u ? 0u : 0x800000u);\n"
            "  while ((left_sig & 0x800000u) == 0u) {\n"
            "    left_sig <<= 1; --left_exp;\n"
            "  }\n"
            "  while ((right_sig & 0x800000u) == 0u) {\n"
            "    right_sig <<= 1; --right_exp;\n"
            "  }\n"
            "  ulong product = ulong(left_sig) * ulong(right_sig);\n"
            "  uint leading = (product & (ulong(1) << 47)) != 0ul ?\n"
            "    47u : 46u;\n"
            "  int exponent = left_exp + right_exp + int(leading - 46u);\n"
            "  uint shift = leading - 23u; bool subnormal = exponent < -126;\n"
            "  if (subnormal) {\n"
            "    shift += uint(-126 - exponent); exponent = -126;\n"
            "  }\n"
            "  ulong rounded = m3_sampling_round_shift(product, shift);\n"
            "  if (subnormal) return rounded >= 0x800000ul ?\n"
            "    0x00800000u : uint(rounded);\n"
            "  if (rounded >= 0x1000000ul) { rounded >>= 1; ++exponent; }\n"
            "  if (exponent > 127) return 0x7f800000u;\n"
            "  return (uint(exponent + 127) << 23) |\n"
            "    (uint(rounded) & 0x7fffffu);\n"
            "}\n";
    NSString *source = [[declarations stringByAppendingString:addition]
        stringByAppendingString:multiplication];

    return [source stringByAppendingString:
        @"kernel void m3_top_k(\n"
            "    device const uchar* logits [[buffer(0)]],\n"
            "    device uchar* values [[buffer(1)]],\n"
            "    device uchar* indices [[buffer(2)]],\n"
            "    constant M3TopKParameters& p [[buffer(3)]],\n"
            "    uint gid [[thread_position_in_grid]]) {\n"
            "  ulong row = ulong(gid);\n"
            "  if (row >= p.row_count) return;\n"
            "  ulong begin = row * p.vocabulary;\n"
            "  bool have_previous = false; uint previous_value = 0u;\n"
            "  int previous_index = 0;\n"
            "  for (ulong rank = 0; rank < p.k; ++rank) {\n"
            "    bool have_best = false; uint best_value = 0u;\n"
            "    int best_index = 0;\n"
            "    for (ulong column = 0; column < p.vocabulary; ++column) {\n"
            "      ulong src = m3_view_logical_offset(\n"
            "        p.logits, begin + column);\n"
            "      uint value = m3_load_float_bits(\n"
            "        logits + src, p.logits.dtype);\n"
            "      int index = int(column);\n"
            "      bool follows = !have_previous || m3_sampling_precedes(\n"
            "        previous_value, previous_index, value, index);\n"
            "      if (follows && (!have_best || m3_sampling_precedes(\n"
            "          value, index, best_value, best_index))) {\n"
            "        have_best = true; best_value = value;\n"
            "        best_index = index;\n"
            "      }\n"
            "    }\n"
            "    ulong flat = row * p.k + rank;\n"
            "    ulong value_at = m3_view_contiguous_offset(p.values, flat);\n"
            "    ulong index_at = m3_view_contiguous_offset(p.indices, flat);\n"
            "    m3_store_float_bits(values + value_at, p.values.dtype,\n"
            "      best_value);\n"
            "    m3_store_i32(indices + index_at, best_index);\n"
            "    have_previous = true; previous_value = best_value;\n"
            "    previous_index = best_index;\n"
            "  }\n"
            "}\n"
            "kernel void m3_categorical(\n"
            "    device const uchar* probabilities [[buffer(0)]],\n"
            "    device const uchar* uniforms [[buffer(1)]],\n"
            "    device uchar* output [[buffer(2)]],\n"
            "    constant M3CategoricalParameters& p [[buffer(3)]],\n"
            "    uint gid [[thread_position_in_grid]]) {\n"
            "  ulong row = ulong(gid);\n"
            "  if (row >= p.row_count) return;\n"
            "  ulong begin = row * p.vocabulary; uint sum = 0u;\n"
            "  int last_positive = 0;\n"
            "  for (ulong column = 0; column < p.vocabulary; ++column) {\n"
            "    ulong src = m3_view_logical_offset(\n"
            "      p.probabilities, begin + column);\n"
            "    uint probability = m3_load_float_bits(\n"
            "      probabilities + src, p.probabilities.dtype);\n"
            "    sum = m3_sampling_add_positive(sum, probability);\n"
            "    if ((probability & 0x7fffffffu) != 0u)\n"
            "      last_positive = int(column);\n"
            "  }\n"
            "  ulong uniform_at = m3_view_logical_offset(p.uniforms, row);\n"
            "  uint uniform = m3_load_float_bits(\n"
            "    uniforms + uniform_at, p.uniforms.dtype);\n"
            "  uint target = m3_sampling_multiply_positive(uniform, sum);\n"
            "  uint cumulative = 0u;\n"
            "  int selected = last_positive;\n"
            "  for (ulong column = 0; column < p.vocabulary; ++column) {\n"
            "    ulong src = m3_view_logical_offset(\n"
            "      p.probabilities, begin + column);\n"
            "    uint probability = m3_load_float_bits(\n"
            "      probabilities + src, p.probabilities.dtype);\n"
            "    cumulative = m3_sampling_add_positive(\n"
            "      cumulative, probability);\n"
            "    if (target < cumulative) { selected = int(column); break; }\n"
            "  }\n"
            "  ulong dst = m3_view_contiguous_offset(p.output, row);\n"
            "  m3_store_i32(output + dst, selected);\n"
            "}\n"];
}

m3_status m3_metal_prepare_sampling(m3_metal_context *context,
                                     m3_error *error)
{
    m3_status status = m3_metal_register_pipeline(
        context, m3_metal_top_k_kernel, error);

    if (status == M3_STATUS_OK) {
        status = m3_metal_register_pipeline(
            context, m3_metal_categorical_kernel, error);
    }
    return status;
}

bool m3_metal_sampling_writes_storage(const m3_command *command,
                                       const m3_storage *storage)
{
    if (storage == NULL) {
        return false;
    }
    if (command->kind == M3_OP_TOP_K) {
        return command->descriptor.top_k.values->storage == storage ||
               command->descriptor.top_k.indices->storage == storage;
    }
    return command->kind == M3_OP_CATEGORICAL &&
           command->descriptor.categorical.output->storage == storage;
}

m3_status m3_metal_preflight_sampling(const m3_command *commands,
                                       size_t command_index,
                                       m3_error *error)
{
    const m3_command *command = &commands[command_index];

    if (command->kind == M3_OP_TOP_K) {
        const m3_op_top_k *op = &command->descriptor.top_k;

        if (m3_metal_has_prior_writer(
                commands, command_index, op->logits->storage)) {
            return m3_error_set(
                error, M3_STATUS_UNSUPPORTED,
                "Metal top-k depends on an earlier output");
        }
    } else if (command->kind == M3_OP_CATEGORICAL) {
        const m3_op_categorical *op = &command->descriptor.categorical;

        if (m3_metal_has_prior_writer(
                commands, command_index, op->probabilities->storage) ||
            m3_metal_has_prior_writer(
                commands, command_index, op->uniforms->storage)) {
            return m3_error_set(
                error, M3_STATUS_UNSUPPORTED,
                "Metal categorical sampling depends on an earlier output");
        }
    } else {
        return M3_STATUS_OK;
    }
    return m3_command_data_preflight(command, error);
}

static void m3_metal_top_k_parameters_init(
    m3_metal_top_k_parameters *parameters, const m3_op_top_k *op,
    size_t row_count, size_t vocabulary)
{
    (void)memset(parameters, 0, sizeof(*parameters));
    m3_metal_view_parameters_init(&parameters->logits, op->logits);
    m3_metal_view_parameters_init(&parameters->values, op->values);
    m3_metal_view_parameters_init(&parameters->indices, op->indices);
    parameters->row_count = (uint64_t)row_count;
    parameters->vocabulary = (uint64_t)vocabulary;
    parameters->k = op->k;
}

static m3_status m3_metal_encode_top_k(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_top_k *op,
    m3_error *error)
{
    id<MTLComputePipelineState> pipeline = m3_metal_find_pipeline(
        context, m3_metal_top_k_kernel);
    id<MTLBuffer> logits = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->logits->storage);
    id<MTLBuffer> values = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->values->storage);
    id<MTLBuffer> indices = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->indices->storage);
    size_t vocabulary = (size_t)op->logits->metadata.shape[
        op->logits->metadata.rank - 1U];
    size_t row_count = op->logits->metadata.element_count / vocabulary;
    m3_metal_top_k_parameters parameters;

    if (row_count == 0U) {
        return M3_STATUS_OK;
    }
    if (logits == nil || values == nil || indices == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal top-k resources are unavailable");
    }
    m3_metal_top_k_parameters_init(
        &parameters, op, row_count, vocabulary);
    [encoder setBuffer:logits offset:0U atIndex:0U];
    [encoder setBuffer:values offset:0U atIndex:1U];
    [encoder setBuffer:indices offset:0U atIndex:2U];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3U];
    return m3_metal_dispatch_1d(encoder, pipeline, row_count, error);
}

static void m3_metal_categorical_parameters_init(
    m3_metal_categorical_parameters *parameters,
    const m3_op_categorical *op, size_t row_count, size_t vocabulary)
{
    (void)memset(parameters, 0, sizeof(*parameters));
    m3_metal_view_parameters_init(
        &parameters->probabilities, op->probabilities);
    m3_metal_view_parameters_init(&parameters->uniforms, op->uniforms);
    m3_metal_view_parameters_init(&parameters->output, op->output);
    parameters->row_count = (uint64_t)row_count;
    parameters->vocabulary = (uint64_t)vocabulary;
}

static m3_status m3_metal_encode_categorical(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_op_categorical *op,
    m3_error *error)
{
    id<MTLComputePipelineState> pipeline = m3_metal_find_pipeline(
        context, m3_metal_categorical_kernel);
    id<MTLBuffer> probabilities = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->probabilities->storage);
    id<MTLBuffer> uniforms = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->uniforms->storage);
    id<MTLBuffer> output = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->output->storage);
    size_t vocabulary = (size_t)op->probabilities->metadata.shape[
        op->probabilities->metadata.rank - 1U];
    size_t row_count = op->uniforms->metadata.element_count;
    m3_metal_categorical_parameters parameters;

    if (row_count == 0U) {
        return M3_STATUS_OK;
    }
    if (probabilities == nil || uniforms == nil || output == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal categorical resources are unavailable");
    }
    m3_metal_categorical_parameters_init(
        &parameters, op, row_count, vocabulary);
    [encoder setBuffer:probabilities offset:0U atIndex:0U];
    [encoder setBuffer:uniforms offset:0U atIndex:1U];
    [encoder setBuffer:output offset:0U atIndex:2U];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3U];
    return m3_metal_dispatch_1d(encoder, pipeline, row_count, error);
}

m3_status m3_metal_encode_sampling(const m3_metal_context *context,
                                    id<MTLComputeCommandEncoder> encoder,
                                    const m3_command *command,
                                    bool *handled, m3_error *error)
{
    if (command->kind == M3_OP_TOP_K) {
        *handled = true;
        return m3_metal_encode_top_k(
            context, encoder, &command->descriptor.top_k, error);
    }
    if (command->kind == M3_OP_CATEGORICAL) {
        *handled = true;
        return m3_metal_encode_categorical(
            context, encoder, &command->descriptor.categorical, error);
    }
    *handled = false;
    return M3_STATUS_OK;
}

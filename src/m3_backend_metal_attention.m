/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_metal_internal.h"

#include "m3_op_internal.h"

#include <stdint.h>
#include <string.h>

static const char m3_metal_attention_kernel[] = "m3_attention";

typedef struct {
    m3_metal_view_parameters query;
    m3_metal_view_parameters key;
    m3_metal_view_parameters value;
    m3_metal_view_parameters mask;
    m3_metal_view_parameters output;
    int64_t causal_offset;
    float scale;
    uint32_t causal;
    uint32_t has_mask;
    uint32_t reserved;
} m3_metal_attention_parameters;

_Static_assert(sizeof(m3_metal_attention_parameters) == 784U,
               "Metal attention parameter ABI changed");

NSString *m3_metal_source_attention(void)
{
    NSString *support =
        @"#pragma clang fp contract(off)\n"
            "struct M3AttentionParameters {\n"
            "  M3ViewParameters query; M3ViewParameters key;\n"
            "  M3ViewParameters value; M3ViewParameters mask;\n"
            "  M3ViewParameters output; long causal_offset;\n"
            "  float scale; uint causal; uint has_mask; uint reserved;\n"
            "};\n"
            "static_assert(sizeof(M3AttentionParameters) == 784, "
            "\"Metal attention parameter size changed\");\n"
            "bool m3_attention_causal_allowed(ulong query, ulong key, "
            "long offset) {\n"
            "  if (offset >= 0) {\n"
            "    ulong positive = ulong(offset);\n"
            "    return query > ~ulong(0) - positive || "
            "key <= query + positive;\n"
            "  }\n"
            "  ulong magnitude = ulong(-(offset + 1l)) + 1ul;\n"
            "  return query >= magnitude && key <= query - magnitude;\n"
            "}\n"
            "ulong m3_attention_mask_base(\n"
            "    constant M3AttentionParameters& p, ulong batch,\n"
            "    ulong head, ulong query) {\n"
            "  ulong mask_batch = p.mask.shape[0] == 1ul ? 0ul : batch;\n"
            "  ulong mask_head = p.mask.shape[1] == 1ul ? 0ul : head;\n"
            "  ulong mask_query = p.mask.shape[2] == 1ul ? 0ul : query;\n"
            "  ulong flat = ((mask_batch * p.mask.shape[1] + mask_head) "
            "* p.mask.shape[2] + mask_query) * p.mask.shape[3];\n"
            "  return m3_view_logical_offset(p.mask, flat);\n"
            "}\n"
            "float m3_attention_score(\n"
            "    device const uchar* query_data,\n"
            "    device const uchar* key_data,\n"
            "    device const uchar* mask_data,\n"
            "    constant M3AttentionParameters& p, ulong query_base,\n"
            "    ulong keys_base, ulong mask_base, ulong query, "
            "ulong key) {\n"
            "  float negative_infinity = as_type<float>(0xff800000u);\n"
            "  if (p.causal != 0u && "
            "!m3_attention_causal_allowed(query, key, p.causal_offset))\n"
            "    return negative_infinity;\n"
            "  ulong depth = p.query.shape[3];\n"
            "  ulong key_base = keys_base + "
            "key * p.key.byte_strides[2];\n"
            "  float sum = 0.0f;\n"
            "  for (ulong channel = 0ul; channel < depth; ++channel) {\n"
            "    ulong query_offset = query_base + "
            "channel * p.query.byte_strides[3];\n"
            "    ulong key_offset = key_base + "
            "channel * p.key.byte_strides[3];\n"
            "    float query_value = "
            "m3_load_float(query_data + query_offset, p.query.dtype);\n"
            "    float key_value = "
            "m3_load_float(key_data + key_offset, p.key.dtype);\n"
            "    float product = query_value * key_value;\n"
            "    sum = sum + product;\n"
            "  }\n"
            "  sum = sum * p.scale;\n"
            "  if (p.has_mask != 0u) {\n"
            "    ulong mask_offset = mask_base + "
            "key * p.mask.byte_strides[3];\n"
            "    float mask_value = "
            "m3_load_float(mask_data + mask_offset, p.mask.dtype);\n"
            "    sum = sum + mask_value;\n"
            "  }\n"
            "  return sum;\n"
            "}\n";
    NSString *kernel =
        @"kernel void m3_attention(\n"
            "    device const uchar* query_data [[buffer(0)]],\n"
            "    device const uchar* key_data [[buffer(1)]],\n"
            "    device const uchar* value_data [[buffer(2)]],\n"
            "    device const uchar* mask_data [[buffer(3)]],\n"
            "    device uchar* output_data [[buffer(4)]],\n"
            "    constant M3AttentionParameters& p [[buffer(5)]],\n"
            "    uint gid [[thread_position_in_grid]]) {\n"
            "  ulong depth = p.query.shape[3];\n"
            "  ulong rows = p.query.element_count / depth;\n"
            "  ulong row = ulong(gid);\n"
            "  if (row >= rows) return;\n"
            "  ulong query_count = p.query.shape[2];\n"
            "  ulong query_heads = p.query.shape[1];\n"
            "  ulong key_count = p.key.shape[2];\n"
            "  ulong kv_heads = p.key.shape[1];\n"
            "  ulong query = row % query_count;\n"
            "  ulong outer = row / query_count;\n"
            "  ulong query_head = outer % query_heads;\n"
            "  ulong batch = outer / query_heads;\n"
            "  ulong kv_head = query_head / (query_heads / kv_heads);\n"
            "  ulong query_base =\n"
            "    m3_view_logical_offset(p.query, row * depth);\n"
            "  ulong kv_flat =\n"
            "    ((batch * kv_heads + kv_head) * key_count) * depth;\n"
            "  ulong keys_base =\n"
            "    m3_view_logical_offset(p.key, kv_flat);\n"
            "  ulong values_base =\n"
            "    m3_view_logical_offset(p.value, kv_flat);\n"
            "  ulong mask_base = p.has_mask == 0u ? 0ul\n"
            "    : m3_attention_mask_base(\n"
            "        p, batch, query_head, query);\n"
            "  float positive_infinity = as_type<float>(0x7f800000u);\n"
            "  float negative_infinity = as_type<float>(0xff800000u);\n"
            "  float maximum = negative_infinity;\n"
            "  ulong positive_infinities = 0ul;\n"
            "  for (ulong key = 0ul; key < key_count; ++key) {\n"
            "    float score = m3_attention_score(query_data, key_data, "
            "mask_data, p, query_base, keys_base, mask_base, query, "
            "key);\n"
            "    if (score > maximum) maximum = score;\n"
            "    if (score == positive_infinity) ++positive_infinities;\n"
            "  }\n"
            "  float denominator = 0.0f;\n"
            "  if (maximum != negative_infinity && "
            "maximum != positive_infinity) {\n"
            "    for (ulong key = 0ul; key < key_count; ++key) {\n"
            "      float score = m3_attention_score("
            "query_data, key_data, mask_data, p, query_base, "
            "keys_base, mask_base, query, key);\n"
            "      float exponential = exp(score - maximum);\n"
            "      denominator = denominator + exponential;\n"
            "    }\n"
            "  }\n"
            "  for (ulong channel = 0ul; channel < depth; ++channel) {\n"
            "    float sum = 0.0f;\n"
            "    if (maximum != negative_infinity) {\n"
            "      for (ulong key = 0ul; key < key_count; ++key) {\n"
            "        float score = m3_attention_score("
            "query_data, key_data, mask_data, p, query_base, "
            "keys_base, mask_base, query, key);\n"
            "        float probability;\n"
            "        if (maximum == positive_infinity) {\n"
            "          probability = score == positive_infinity\n"
            "              ? 1.0f / float(positive_infinities) : 0.0f;\n"
            "        } else {\n"
            "          probability = exp(score - maximum) / denominator;\n"
            "        }\n"
            "        probability = "
            "m3_round_float(p.query.dtype, probability);\n"
            "        ulong value_offset = values_base + "
            "key * p.value.byte_strides[2] + "
            "channel * p.value.byte_strides[3];\n"
            "        float value = "
            "m3_load_float(value_data + value_offset, p.value.dtype);\n"
            "        float product = probability * value;\n"
            "        sum = sum + product;\n"
            "      }\n"
            "    }\n"
            "    ulong output_offset = "
            "m3_view_contiguous_offset(p.output, row * depth + channel);\n"
            "    m3_store_float(output_data + output_offset, "
            "p.output.dtype, sum);\n"
            "  }\n"
            "}\n"
            "#pragma clang fp contract(on)\n";

    return [support stringByAppendingString:kernel];
}

m3_status m3_metal_prepare_attention(m3_metal_context *context,
                                      m3_error *error)
{
    return m3_metal_register_pipeline(context, m3_metal_attention_kernel,
                                      error);
}

bool m3_metal_attention_writes_storage(const m3_command *command,
                                        const m3_storage *storage)
{
    return command->kind == M3_OP_ATTENTION && storage != NULL &&
           command->descriptor.attention.output->storage == storage;
}

m3_status m3_metal_preflight_attention(const m3_command *commands,
                                        size_t command_index,
                                        m3_error *error)
{
    const m3_command *command = &commands[command_index];
    const m3_op_attention *op;

    if (command->kind != M3_OP_ATTENTION) {
        return M3_STATUS_OK;
    }
    op = &command->descriptor.attention;
    if (m3_metal_has_prior_writer(commands, command_index,
                                  op->query->storage)) {
        return m3_error_set(
            error, M3_STATUS_UNSUPPORTED,
            "Metal attention query depends on an earlier output");
    }
    if (m3_metal_has_prior_writer(commands, command_index,
                                  op->key->storage)) {
        return m3_error_set(
            error, M3_STATUS_UNSUPPORTED,
            "Metal attention key depends on an earlier output");
    }
    if (m3_metal_has_prior_writer(commands, command_index,
                                  op->value->storage)) {
        return m3_error_set(
            error, M3_STATUS_UNSUPPORTED,
            "Metal attention value depends on an earlier output");
    }
    if (op->mask != NULL &&
        m3_metal_has_prior_writer(commands, command_index,
                                  op->mask->storage)) {
        return m3_error_set(
            error, M3_STATUS_UNSUPPORTED,
            "Metal attention mask depends on an earlier output");
    }
    return m3_command_data_preflight(command, error);
}

static void m3_metal_attention_parameters_init(
    m3_metal_attention_parameters *parameters,
    const m3_op_attention *op)
{
    (void)memset(parameters, 0, sizeof(*parameters));
    m3_metal_view_parameters_init(&parameters->query, op->query);
    m3_metal_view_parameters_init(&parameters->key, op->key);
    m3_metal_view_parameters_init(&parameters->value, op->value);
    if (op->mask != NULL) {
        m3_metal_view_parameters_init(&parameters->mask, op->mask);
        parameters->has_mask = 1U;
    }
    m3_metal_view_parameters_init(&parameters->output, op->output);
    parameters->causal_offset = op->causal_offset;
    parameters->scale = op->scale;
    parameters->causal = op->causal ? 1U : 0U;
}

m3_status m3_metal_encode_attention(const m3_metal_context *context,
                                     id<MTLComputeCommandEncoder> encoder,
                                     const m3_command *command,
                                     bool *handled, m3_error *error)
{
    const m3_op_attention *op;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> query;
    id<MTLBuffer> key;
    id<MTLBuffer> value;
    id<MTLBuffer> mask;
    id<MTLBuffer> output;
    m3_metal_attention_parameters parameters;
    size_t work_count;

    if (command->kind != M3_OP_ATTENTION) {
        *handled = false;
        return M3_STATUS_OK;
    }
    *handled = true;
    op = &command->descriptor.attention;
    work_count = op->query->metadata.element_count /
                 (size_t)op->query->metadata.shape[3];
    if (work_count == 0U) {
        return m3_metal_dispatch_1d(encoder, nil, 0U, error);
    }
    pipeline = m3_metal_find_pipeline(context, m3_metal_attention_kernel);
    query = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->query->storage);
    key = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->key->storage);
    value = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->value->storage);
    mask = op->mask == NULL
               ? query
               : (__bridge id<MTLBuffer>)
                     m3_storage_handle_internal(op->mask->storage);
    output = (__bridge id<MTLBuffer>)
        m3_storage_handle_internal(op->output->storage);
    if (pipeline == nil || query == nil || key == nil || value == nil ||
        mask == nil || output == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal attention resources are unavailable");
    }
    m3_metal_attention_parameters_init(&parameters, op);
    [encoder setBuffer:query offset:0U atIndex:0U];
    [encoder setBuffer:key offset:0U atIndex:1U];
    [encoder setBuffer:value offset:0U atIndex:2U];
    [encoder setBuffer:mask offset:0U atIndex:3U];
    [encoder setBuffer:output offset:0U atIndex:4U];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5U];
    return m3_metal_dispatch_1d(encoder, pipeline, work_count, error);
}

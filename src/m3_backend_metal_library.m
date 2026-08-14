/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_metal_internal.h"

#include <stdint.h>
#include <string.h>

#define M3_METAL_FAMILY_SOURCE_BYTE_LIMIT (128U * 1024U)
#define M3_METAL_LIBRARY_SOURCE_BYTE_LIMIT (768U * 1024U)

typedef NSString *(*m3_metal_source_function)(void);
typedef m3_status (*m3_metal_prepare_function)(m3_metal_context *context,
                                               m3_error *error);

typedef struct {
    const char *name;
    m3_metal_source_function source;
    m3_metal_prepare_function prepare;
} m3_metal_family;

static const m3_metal_family m3_metal_families[] = {
    {"basic", m3_metal_source_basic, m3_metal_prepare_basic},
    {"dense", m3_metal_source_dense, m3_metal_prepare_dense},
    {"attention", m3_metal_source_attention, m3_metal_prepare_attention},
    {"sampling", m3_metal_source_sampling, m3_metal_prepare_sampling},
    {"convolution", m3_metal_source_convolution,
     m3_metal_prepare_convolution}
};

static const size_t m3_metal_family_count =
    sizeof(m3_metal_families) / sizeof(m3_metal_families[0]);

static NSString *m3_metal_common_source(void)
{
    NSString *prefix =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "ushort m3_load_u16(device const uchar* p) {\n"
         "  return ushort(p[0]) | (ushort(p[1]) << 8);\n"
         "}\n"
         "uint m3_load_u32(device const uchar* p) {\n"
         "  return uint(p[0]) | (uint(p[1]) << 8) | "
         "(uint(p[2]) << 16) | (uint(p[3]) << 24);\n"
         "}\n"
         "void m3_store_u16(device uchar* p, ushort v) {\n"
         "  p[0] = uchar(v); p[1] = uchar(v >> 8);\n"
         "}\n"
         "void m3_store_u32(device uchar* p, uint v) {\n"
         "  p[0] = uchar(v); p[1] = uchar(v >> 8); "
         "p[2] = uchar(v >> 16); p[3] = uchar(v >> 24);\n"
         "}\n"
         "uint m3_f16_to_f32_bits(ushort value) {\n"
         "  uint sign = (uint(value) & 0x8000u) << 16;\n"
         "  uint exponent = (uint(value) >> 10) & 0x1fu;\n"
         "  uint fraction = uint(value) & 0x03ffu;\n"
         "  if (exponent == 0u) {\n"
         "    if (fraction == 0u) return sign; exponent = 113u;\n"
         "    while ((fraction & 0x0400u) == 0u) { "
         "fraction <<= 1; --exponent; }\n"
         "    return sign | (exponent << 23) | "
         "((fraction & 0x03ffu) << 13);\n"
         "  }\n"
         "  if (exponent == 0x1fu) return sign | 0x7f800000u | "
         "(fraction << 13);\n"
         "  return sign | ((exponent + 112u) << 23) | "
         "(fraction << 13);\n"
         "}\n";

    return [prefix stringByAppendingString:
        @"ushort m3_f32_to_f16_bits(uint bits) {\n"
         "  ushort sign = ushort((bits >> 16) & 0x8000u);\n"
         "  uint exponent = (bits >> 23) & 0xffu;\n"
         "  uint fraction = bits & 0x007fffffu;\n"
         "  if (exponent == 0xffu) {\n"
         "    if (fraction == 0u) return ushort(sign | 0x7c00u);\n"
         "    fraction >>= 13; if (fraction == 0u) fraction = 1u;\n"
         "    return ushort(sign | 0x7c00u | fraction | 0x0200u);\n"
         "  }\n"
         "  int unbiased = int(exponent) - 127;\n"
         "  if (unbiased > 15) return ushort(sign | 0x7c00u);\n"
         "  if (unbiased >= -14) {\n"
         "    uint result = fraction >> 13; "
         "uint rem = fraction & 0x1fffu;\n"
         "    uint out_exp = uint(unbiased + 15);\n"
         "    if (rem > 0x1000u || "
         "(rem == 0x1000u && (result & 1u))) {\n"
         "      ++result; if (result == 0x0400u) { result = 0u; "
         "++out_exp; }\n"
         "    }\n"
         "    if (out_exp >= 0x1fu) return ushort(sign | 0x7c00u);\n"
         "    return ushort(sign | (out_exp << 10) | result);\n"
         "  }\n"
         "  if (unbiased >= -25) {\n"
         "    uint significand = 0x00800000u | fraction;\n"
         "    uint shift = uint(-unbiased - 1);\n"
         "    uint result = significand >> shift;\n"
         "    uint mask = (1u << shift) - 1u; "
         "uint rem = significand & mask;\n"
         "    uint halfway = 1u << (shift - 1u);\n"
         "    if (rem > halfway || "
         "(rem == halfway && (result & 1u))) ++result;\n"
         "    return ushort(sign | result);\n"
         "  } return sign;\n"
         "}\n"
         "ushort m3_f32_to_bf16_bits(uint bits) {\n"
         "  uint exponent = bits & 0x7f800000u; "
         "uint fraction = bits & 0x007fffffu;\n"
         "  if (exponent == 0x7f800000u && fraction != 0u) "
         "return ushort((bits >> 16) | 0x0040u);\n"
         "  bits += 0x7fffu + ((bits >> 16) & 1u); "
         "return ushort(bits >> 16);\n"
         "}\n"
         "uint m3_dtype_size(uint type) {\n"
         "  return type == 1u || type == 2u ? 2u : 4u;\n"
         "}\n"
         "int m3_load_i32(device const uchar* p) {\n"
         "  return as_type<int>(m3_load_u32(p));\n"
         "}\n"
         "void m3_store_i32(device uchar* p, int value) {\n"
         "  m3_store_u32(p, as_type<uint>(value));\n"
         "}\n"
         "uint m3_load_float_bits(device const uchar* p, uint type) {\n"
         "  if (type == 0u) return m3_load_u32(p);\n"
         "  ushort bits = m3_load_u16(p);\n"
         "  return type == 1u ? m3_f16_to_f32_bits(bits) : "
         "uint(bits) << 16;\n"
         "}\n"
         "float m3_load_float(device const uchar* p, uint type) {\n"
         "  if (type == 3u) return float(m3_load_i32(p));\n"
         "  return as_type<float>(m3_load_float_bits(p, type));\n"
         "}\n"
         "void m3_store_float_bits(device uchar* p, uint type, "
         "uint bits) {\n"
         "  if (type == 0u) m3_store_u32(p, bits);\n"
         "  else if (type == 1u) m3_store_u16(p, "
         "m3_f32_to_f16_bits(bits));\n"
         "  else m3_store_u16(p, m3_f32_to_bf16_bits(bits));\n"
         "}\n"
         "void m3_store_float(device uchar* p, uint type, float value) {\n"
         "  if (type == 3u) m3_store_i32(p, int(value));\n"
         "  else m3_store_float_bits(p, type, as_type<uint>(value));\n"
         "}\n"
         "ulong m3_logical_offset(ulong flat, ulong base, "
         "constant ulong* shape, constant ulong* strides, uint rank) {\n"
         "  ulong invalid = ~ulong(0); if (rank > 8u) return invalid;\n"
         "  ulong offset = base;\n"
         "  for (uint axis = rank; axis > 0u; --axis) {\n"
         "    ulong dimension = shape[axis - 1u];\n"
         "    if (dimension == 0u) return invalid;\n"
         "    ulong coordinate = flat % dimension; flat /= dimension;\n"
         "    ulong stride = strides[axis - 1u];\n"
         "    if (coordinate != 0u && "
         "stride > (invalid - offset) / coordinate) return invalid;\n"
         "    offset += coordinate * stride;\n"
         "  }\n"
         "  return flat == 0u ? offset : invalid;\n"
         "}\n"];
}

static m3_status m3_metal_source_append(
    NSMutableString *source, size_t *source_bytes,
    const m3_metal_family *family, m3_error *error)
{
    NSString *fragment = family->source();
    NSUInteger fragment_bytes;

    if (fragment == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal %s source fragment is null",
                            family->name);
    }
    fragment_bytes =
        [fragment lengthOfBytesUsingEncoding:NSUTF8StringEncoding];
    if (fragment_bytes > M3_METAL_FAMILY_SOURCE_BYTE_LIMIT) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal %s source fragment exceeds its limit",
                            family->name);
    }
    if ((size_t)fragment_bytes >
        M3_METAL_LIBRARY_SOURCE_BYTE_LIMIT - *source_bytes) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal library source exceeds its limit");
    }
    [source appendString:fragment];
    *source_bytes += (size_t)fragment_bytes;
    return M3_STATUS_OK;
}

static NSString *m3_metal_kernel_source(m3_status *status,
                                         m3_error *error)
{
    NSString *common = m3_metal_common_source();
    NSMutableString *source = [NSMutableString stringWithString:common];
    size_t source_bytes =
        (size_t)[common lengthOfBytesUsingEncoding:NSUTF8StringEncoding];
    size_t index;

    if (source_bytes > M3_METAL_LIBRARY_SOURCE_BYTE_LIMIT) {
        *status = m3_error_set(error, M3_STATUS_INTERNAL,
                               "Metal common source exceeds its limit");
        return nil;
    }
    for (index = 0U; index < m3_metal_family_count; ++index) {
        *status = m3_metal_source_append(
            source, &source_bytes, &m3_metal_families[index], error);
        if (*status != M3_STATUS_OK) {
            return nil;
        }
    }
    *status = M3_STATUS_OK;
    return [source copy];
}

static m3_status m3_metal_pipeline_error(m3_error *error,
                                          const char *stage,
                                          NSError *metal_error)
{
    long code = metal_error == nil ? 0L : (long)metal_error.code;

    return m3_error_set(error, M3_STATUS_INTERNAL,
                        "Metal %s failed (error %ld)", stage, code);
}

m3_status m3_metal_register_pipeline(m3_metal_context *context,
                                      const char *name,
                                      m3_error *error)
{
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLFunction> function;
    id<MTLComputePipelineState> pipeline;
    NSString *function_name;
    NSError *metal_error = nil;
    size_t name_length;
    size_t index;

    if (context == NULL || context->library == NULL || name == NULL) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal pipeline registry is unavailable");
    }
    name_length = strnlen(name, M3_METAL_PIPELINE_NAME_CAPACITY);
    if (name_length == 0U ||
        name_length == M3_METAL_PIPELINE_NAME_CAPACITY) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal pipeline name is invalid");
    }
    for (index = 0U; index < context->pipeline_count; ++index) {
        if (strcmp(context->pipelines[index].name, name) == 0) {
            return m3_error_set(error, M3_STATUS_INTERNAL,
                                "Metal pipeline is already registered");
        }
    }
    if (context->pipeline_count >= M3_METAL_PIPELINE_CAPACITY) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal pipeline registry is full");
    }
    device = (__bridge id<MTLDevice>)context->device;
    library = (__bridge id<MTLLibrary>)context->library;
    function_name = [NSString stringWithUTF8String:name];
    if (device == nil || library == nil || function_name == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal pipeline registry is unavailable");
    }
    function = [library newFunctionWithName:function_name];
    if (function == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal library is missing a required kernel");
    }
    pipeline = [device newComputePipelineStateWithFunction:function
                                                      error:&metal_error];
    if (pipeline == nil) {
        return m3_metal_pipeline_error(error, "pipeline creation",
                                       metal_error);
    }
    index = context->pipeline_count;
    (void)memcpy(context->pipelines[index].name, name, name_length + 1U);
    context->pipelines[index].state = CFBridgingRetain(pipeline);
    context->pipeline_count = index + 1U;
    return M3_STATUS_OK;
}

id<MTLComputePipelineState> m3_metal_find_pipeline(
    const m3_metal_context *context, const char *name)
{
    size_t index;

    if (context == NULL || name == NULL) {
        return nil;
    }
    for (index = 0U; index < context->pipeline_count; ++index) {
        if (strcmp(context->pipelines[index].name, name) == 0) {
            return (__bridge id<MTLComputePipelineState>)
                context->pipelines[index].state;
        }
    }
    return nil;
}

m3_status m3_metal_compile_library(m3_metal_context *context,
                                    m3_error *error)
{
    id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    NSString *source;
    id<MTLLibrary> library;
    NSError *metal_error = nil;
    m3_status status;
    size_t index;

    if (context->library != NULL || context->pipeline_count != 0U) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal library is already prepared");
    }
    source = m3_metal_kernel_source(&status, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    options.mathMode = MTLMathModeSafe;
    library = [device newLibraryWithSource:source
                                   options:options
                                     error:&metal_error];
    if (library == nil) {
        return m3_metal_pipeline_error(error, "library compilation",
                                       metal_error);
    }
    context->library = CFBridgingRetain(library);
    for (index = 0U; index < m3_metal_family_count; ++index) {
        status = m3_metal_families[index].prepare(context, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    return M3_STATUS_OK;
}

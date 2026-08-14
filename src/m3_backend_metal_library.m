/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_metal_internal.h"

static NSString *m3_metal_kernel_source(void)
{
    NSString *prefix = @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "struct Params { ulong count; ulong in_offset; ulong out_offset; "
            "ulong shape[8]; ulong strides[8]; uint rank; uint in_type; "
            "uint out_type; uint element_size; };\n"
            "ulong input_offset(constant Params& p, ulong flat) {\n"
            "  ulong offset = p.in_offset;\n"
            "  for (uint axis = p.rank; axis > 0; --axis) {\n"
            "    ulong dimension = p.shape[axis - 1];\n"
            "    ulong coordinate = flat % dimension; flat /= dimension;\n"
            "    offset += coordinate * p.strides[axis - 1];\n"
            "  } return offset;\n"
            "}\n"
            "ushort load_u16(device const uchar* p) {\n"
            "  return ushort(p[0]) | (ushort(p[1]) << 8);\n"
            "}\n"
            "uint load_u32(device const uchar* p) {\n"
            "  return uint(p[0]) | (uint(p[1]) << 8) | "
            "(uint(p[2]) << 16) | (uint(p[3]) << 24);\n"
            "}\n"
            "void store_u16(device uchar* p, ushort v) {\n"
            "  p[0] = uchar(v); p[1] = uchar(v >> 8);\n"
            "}\n"
            "void store_u32(device uchar* p, uint v) {\n"
            "  p[0] = uchar(v); p[1] = uchar(v >> 8); "
            "p[2] = uchar(v >> 16); p[3] = uchar(v >> 24);\n"
            "}\n"
            "uint f16_to_f32_bits(ushort value) {\n"
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
            @"ushort f32_to_f16_bits(uint bits) {\n"
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
            "    uint result = fraction >> 13; uint rem = fraction & 0x1fffu;\n"
            "    uint out_exp = uint(unbiased + 15);\n"
            "    if (rem > 0x1000u || (rem == 0x1000u && (result & 1u))) {\n"
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
            "    uint mask = (1u << shift) - 1u; uint rem = significand & mask;\n"
            "    uint halfway = 1u << (shift - 1u);\n"
            "    if (rem > halfway || (rem == halfway && (result & 1u))) "
            "++result;\n"
            "    return ushort(sign | result);\n"
            "  } return sign;\n"
            "}\n"
            "ushort f32_to_bf16_bits(uint bits) {\n"
            "  uint exponent = bits & 0x7f800000u; "
            "uint fraction = bits & 0x007fffffu;\n"
            "  if (exponent == 0x7f800000u && fraction != 0u) "
            "return ushort((bits >> 16) | 0x0040u);\n"
            "  bits += 0x7fffu + ((bits >> 16) & 1u); "
            "return ushort(bits >> 16);\n"
            "}\n"
            "uint load_float_bits(device const uchar* p, uint type) {\n"
            "  if (type == 0u) return load_u32(p);\n"
            "  ushort bits = load_u16(p);\n"
            "  return type == 1u ? f16_to_f32_bits(bits) : uint(bits) << 16;\n"
            "}\n"
            "kernel void m3_copy(device const uchar* input [[buffer(0)]], "
            "device uchar* output [[buffer(1)]], "
            "constant Params& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
            "  if (ulong(gid) >= p.count) return; "
            "ulong src = input_offset(p, ulong(gid));\n"
            "  ulong dst = p.out_offset + ulong(gid) * ulong(p.element_size);\n"
            "  for (uint b = 0; b < p.element_size; ++b) "
            "output[dst + b] = input[src + b];\n"
            "}\n"
            "kernel void m3_cast(device const uchar* input [[buffer(0)]], "
            "device uchar* output [[buffer(1)]], "
            "constant Params& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
            "  if (ulong(gid) >= p.count) return; "
            "ulong src = input_offset(p, ulong(gid));\n"
            "  ulong dst = p.out_offset + ulong(gid) * "
            "ulong(p.out_type == 1u || p.out_type == 2u ? 2u : 4u);\n"
            "  if (p.in_type == 3u) {\n"
            "    int integer = as_type<int>(load_u32(input + src));\n"
            "    if (p.out_type == 3u) { store_u32(output + dst, "
            "as_type<uint>(integer)); return; }\n"
            "    uint bits = as_type<uint>(float(integer));\n"
            "    if (p.out_type == 0u) store_u32(output + dst, bits);\n"
            "    else if (p.out_type == 1u) store_u16(output + dst, "
            "f32_to_f16_bits(bits));\n"
            "    else store_u16(output + dst, f32_to_bf16_bits(bits));\n"
            "    return;\n"
            "  }\n"
            "  uint bits = load_float_bits(input + src, p.in_type);\n"
            "  if (p.out_type == 3u) store_u32(output + dst, "
            "as_type<uint>(int(as_type<float>(bits))));\n"
            "  else if (p.out_type == 0u) store_u32(output + dst, bits);\n"
            "  else if (p.out_type == 1u) store_u16(output + dst, "
            "f32_to_f16_bits(bits));\n"
            "  else store_u16(output + dst, f32_to_bf16_bits(bits));\n"
            "}\n"];
}

static m3_status m3_metal_pipeline_error(m3_error *error,
                                          const char *stage,
                                          NSError *metal_error)
{
    long code = metal_error == nil ? 0L : (long)metal_error.code;

    return m3_error_set(error, M3_STATUS_INTERNAL,
                        "Metal %s failed (error %ld)", stage, code);
}

m3_status m3_metal_compile_library(m3_metal_context *context,
                                    m3_error *error)
{
    id<MTLDevice> device = (__bridge id<MTLDevice>)context->device;
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> library;
    id<MTLFunction> copy_function;
    id<MTLFunction> cast_function;
    id<MTLComputePipelineState> copy_pipeline;
    id<MTLComputePipelineState> cast_pipeline;
    NSError *metal_error = nil;

    options.mathMode = MTLMathModeSafe;
    library = [device newLibraryWithSource:m3_metal_kernel_source()
                                   options:options
                                     error:&metal_error];
    if (library == nil) {
        return m3_metal_pipeline_error(error, "library compilation",
                                       metal_error);
    }
    copy_function = [library newFunctionWithName:@"m3_copy"];
    cast_function = [library newFunctionWithName:@"m3_cast"];
    if (copy_function == nil || cast_function == nil) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Metal library is missing a required kernel");
    }
    metal_error = nil;
    copy_pipeline =
        [device newComputePipelineStateWithFunction:copy_function
                                              error:&metal_error];
    if (copy_pipeline == nil) {
        return m3_metal_pipeline_error(error, "COPY pipeline creation",
                                       metal_error);
    }
    metal_error = nil;
    cast_pipeline =
        [device newComputePipelineStateWithFunction:cast_function
                                              error:&metal_error];
    if (cast_pipeline == nil) {
        return m3_metal_pipeline_error(error, "CAST pipeline creation",
                                       metal_error);
    }
    context->library = CFBridgingRetain(library);
    context->copy_pipeline = CFBridgingRetain(copy_pipeline);
    context->cast_pipeline = CFBridgingRetain(cast_pipeline);
    return M3_STATUS_OK;
}

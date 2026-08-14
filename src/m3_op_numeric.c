/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"

#include "m3_backend.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static uint32_t m3_op_float_bits(float value)
{
    uint32_t bits;

    (void)memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float m3_op_bits_float(uint32_t bits)
{
    float value;

    (void)memcpy(&value, &bits, sizeof(value));
    return value;
}

static float m3_op_f16_to_f32(uint16_t half)
{
    uint32_t sign = ((uint32_t)half & 0x8000U) << 16U;
    uint32_t exponent = ((uint32_t)half >> 10U) & 0x1fU;
    uint32_t fraction = (uint32_t)half & 0x03ffU;
    uint32_t bits;

    if (exponent == 0U) {
        if (fraction == 0U) {
            return m3_op_bits_float(sign);
        }
        exponent = 113U;
        while ((fraction & 0x0400U) == 0U) {
            fraction <<= 1U;
            --exponent;
        }
        fraction &= 0x03ffU;
        bits = sign | (exponent << 23U) | (fraction << 13U);
        return m3_op_bits_float(bits);
    }
    if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (fraction << 13U);
        return m3_op_bits_float(bits);
    }
    bits = sign | ((exponent + 112U) << 23U) | (fraction << 13U);
    return m3_op_bits_float(bits);
}

static uint16_t m3_op_f32_to_f16(float value)
{
    uint32_t bits = m3_op_float_bits(value);
    uint16_t sign = (uint16_t)((bits >> 16U) & 0x8000U);
    uint32_t exponent = (bits >> 23U) & 0xffU;
    uint32_t fraction = bits & 0x007fffffU;
    int unbiased;

    if (exponent == 0xffU) {
        if (fraction == 0U) {
            return (uint16_t)(sign | 0x7c00U);
        }
        fraction >>= 13U;
        if (fraction == 0U) {
            fraction = 1U;
        }
        fraction |= 0x0200U;
        return (uint16_t)(sign | 0x7c00U | (uint16_t)fraction);
    }
    unbiased = (int)exponent - 127;
    if (unbiased > 15) {
        return (uint16_t)(sign | 0x7c00U);
    }
    if (unbiased >= -14) {
        uint32_t result_fraction = fraction >> 13U;
        uint32_t remainder = fraction & 0x1fffU;
        uint32_t result_exponent = (uint32_t)(unbiased + 15);

        if (remainder > 0x1000U ||
            (remainder == 0x1000U && (result_fraction & 1U) != 0U)) {
            ++result_fraction;
            if (result_fraction == 0x0400U) {
                result_fraction = 0U;
                ++result_exponent;
            }
        }
        if (result_exponent >= 0x1fU) {
            return (uint16_t)(sign | 0x7c00U);
        }
        return (uint16_t)(sign | (uint16_t)(result_exponent << 10U) |
                          (uint16_t)result_fraction);
    }
    if (unbiased >= -25) {
        uint32_t significand = 0x00800000U | fraction;
        unsigned int shift = (unsigned int)(-unbiased - 1);
        uint32_t result_fraction = significand >> shift;
        uint32_t mask = (UINT32_C(1) << shift) - 1U;
        uint32_t remainder = significand & mask;
        uint32_t halfway = UINT32_C(1) << (shift - 1U);

        if (remainder > halfway ||
            (remainder == halfway && (result_fraction & 1U) != 0U)) {
            ++result_fraction;
        }
        return (uint16_t)(sign | (uint16_t)result_fraction);
    }
    return sign;
}

static float m3_op_bf16_to_f32(uint16_t value)
{
    return m3_op_bits_float((uint32_t)value << 16U);
}

static uint16_t m3_op_f32_to_bf16(float value)
{
    uint32_t bits = m3_op_float_bits(value);
    uint32_t exponent = bits & 0x7f800000U;
    uint32_t fraction = bits & 0x007fffffU;

    if (exponent == 0x7f800000U && fraction != 0U) {
        return (uint16_t)((bits >> 16U) | 0x0040U);
    }
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return (uint16_t)(bits >> 16U);
}

size_t m3_op_element_offset(const m3_tensor_view *view, size_t flat_index)
{
    size_t offset = view->byte_offset;
    uint8_t axis = view->metadata.rank;

    while (axis > 0U) {
        size_t coordinate;
        size_t dimension;

        --axis;
        dimension = (size_t)view->metadata.shape[axis];
        coordinate = flat_index % dimension;
        flat_index /= dimension;
        offset += coordinate * view->byte_strides[axis];
    }
    return offset;
}

size_t m3_op_broadcast_offset(const m3_tensor_view *input,
                              const m3_tensor_view *output,
                              size_t output_flat_index)
{
    size_t offset = input->byte_offset;
    uint8_t output_axis = output->metadata.rank;

    while (output_axis > 0U) {
        size_t coordinate;
        size_t dimension;
        int input_axis;

        --output_axis;
        dimension = (size_t)output->metadata.shape[output_axis];
        coordinate = output_flat_index % dimension;
        output_flat_index /= dimension;
        input_axis = (int)output_axis -
                     (int)(output->metadata.rank - input->metadata.rank);
        if (input_axis >= 0 &&
            input->metadata.shape[(uint8_t)input_axis] != 1U) {
            offset += coordinate * input->byte_strides[(uint8_t)input_axis];
        }
    }
    return offset;
}

float m3_op_load_float(const m3_tensor_view *view, size_t byte_offset)
{
    const uint8_t *base = m3_storage_const_data(view->storage);

    if (view->metadata.dtype == M3_DTYPE_F32) {
        float value;

        (void)memcpy(&value, base + byte_offset, sizeof(value));
        return value;
    }
    if (view->metadata.dtype == M3_DTYPE_F16) {
        uint16_t value;

        (void)memcpy(&value, base + byte_offset, sizeof(value));
        return m3_op_f16_to_f32(value);
    }
    if (view->metadata.dtype == M3_DTYPE_BF16) {
        uint16_t value;

        (void)memcpy(&value, base + byte_offset, sizeof(value));
        return m3_op_bf16_to_f32(value);
    }
    return (float)m3_op_load_i32(view, byte_offset);
}

int32_t m3_op_load_i32(const m3_tensor_view *view, size_t byte_offset)
{
    const uint8_t *base = m3_storage_const_data(view->storage);
    int32_t value;

    (void)memcpy(&value, base + byte_offset, sizeof(value));
    return value;
}

void m3_op_store_float(m3_tensor_view *view, size_t byte_offset, float value)
{
    uint8_t *base = m3_storage_data(view->storage);

    if (view->metadata.dtype == M3_DTYPE_F32) {
        (void)memcpy(base + byte_offset, &value, sizeof(value));
    } else if (view->metadata.dtype == M3_DTYPE_F16) {
        uint16_t converted = m3_op_f32_to_f16(value);

        (void)memcpy(base + byte_offset, &converted, sizeof(converted));
    } else if (view->metadata.dtype == M3_DTYPE_BF16) {
        uint16_t converted = m3_op_f32_to_bf16(value);

        (void)memcpy(base + byte_offset, &converted, sizeof(converted));
    } else {
        int32_t converted = (int32_t)value;

        (void)memcpy(base + byte_offset, &converted, sizeof(converted));
    }
}

void m3_op_store_i32(m3_tensor_view *view, size_t byte_offset,
                     int32_t value)
{
    uint8_t *base = m3_storage_data(view->storage);

    (void)memcpy(base + byte_offset, &value, sizeof(value));
}

/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    const void *data;
    size_t count;
} m3_metal_cast_vector;

static bool m3_test_metal_cast_fixture_init(m3_test_context *test,
                                            m3_op_test_fixture *fixture)
{
    m3_error error;
    m3_status status;

    (void)memset(fixture, 0, sizeof(*fixture));
    status = m3_backend_create_metal(&fixture->backend, &error);
    if (status == M3_STATUS_UNSUPPORTED) {
        M3_TEST_SKIP(test, m3_error_message(&error));
        return false;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create Metal backend for CAST parity");
    return status == M3_STATUS_OK;
}

static m3_metal_cast_vector m3_test_metal_cast_values(
    m3_dtype input_dtype, m3_dtype output_dtype)
{
    static const uint32_t f32_edges[] = {
        0x00000000U, 0x80000000U, 0x3f800000U, 0xbf800000U,
        0x00000001U, 0x007fffffU, 0x7f800000U, 0xff800000U,
        0x7f800001U, 0x7fc12345U, 0x3f801000U, 0x3f803000U
    };
    static const float f32_i32[] = {
        0.0F, -0.0F, 2.9F, -2.9F, 2147483520.0F,
        -2147483648.0F, 0x1p-149F, -0x1p-149F
    };
    static const uint16_t f16_edges[] = {
        0x0000U, 0x8000U, 0x3c00U, 0xbc00U, 0x0001U, 0x03ffU,
        0x0400U, 0x7bffU, 0x7c00U, 0xfc00U, 0x7c01U, 0x7e55U
    };
    static const uint16_t f16_i32[] = {
        0x0000U, 0x8000U, 0x41ccU, 0xc1ccU,
        0x7bffU, 0xfbffU, 0x0001U, 0x8001U
    };
    static const uint16_t bf16_edges[] = {
        0x0000U, 0x8000U, 0x3f80U, 0xbf80U, 0x0001U, 0x007fU,
        0x0080U, 0x7f7fU, 0x7f80U, 0xff80U, 0x7f81U, 0x7fc5U
    };
    static const uint16_t bf16_i32[] = {
        0x0000U, 0x8000U, 0x4039U, 0xc039U,
        0x4effU, 0xcf00U, 0x0001U, 0x8001U
    };
    static const int32_t i32_edges[] = {
        INT32_MIN, INT32_MAX, 0, 1, -1, 16777215,
        16777217, -16777217, 65504, -65504
    };
    m3_metal_cast_vector vector;

    if (input_dtype == M3_DTYPE_F32) {
        vector.data = output_dtype == M3_DTYPE_I32
                          ? (const void *)f32_i32
                          : (const void *)f32_edges;
        vector.count = output_dtype == M3_DTYPE_I32
                           ? sizeof(f32_i32) / sizeof(f32_i32[0])
                           : sizeof(f32_edges) / sizeof(f32_edges[0]);
    } else if (input_dtype == M3_DTYPE_F16) {
        vector.data = output_dtype == M3_DTYPE_I32
                          ? (const void *)f16_i32
                          : (const void *)f16_edges;
        vector.count = output_dtype == M3_DTYPE_I32
                           ? sizeof(f16_i32) / sizeof(f16_i32[0])
                           : sizeof(f16_edges) / sizeof(f16_edges[0]);
    } else if (input_dtype == M3_DTYPE_BF16) {
        vector.data = output_dtype == M3_DTYPE_I32
                          ? (const void *)bf16_i32
                          : (const void *)bf16_edges;
        vector.count = output_dtype == M3_DTYPE_I32
                           ? sizeof(bf16_i32) / sizeof(bf16_i32[0])
                           : sizeof(bf16_edges) / sizeof(bf16_edges[0]);
    } else {
        vector.data = i32_edges;
        vector.count = sizeof(i32_edges) / sizeof(i32_edges[0]);
    }
    return vector;
}

static bool m3_test_metal_cast_pair(
    m3_test_context *test, m3_op_test_fixture *metal,
    m3_op_test_fixture *host, m3_dtype input_dtype,
    m3_dtype output_dtype)
{
    m3_metal_cast_vector vector =
        m3_test_metal_cast_values(input_dtype, output_dtype);
    uint8_t sentinel[64];
    uint64_t shape[] = {(uint64_t)vector.count};
    m3_tensor_view metal_input;
    m3_tensor_view metal_output;
    m3_tensor_view host_input;
    m3_tensor_view host_output;
    m3_command metal_command;
    m3_command host_command;
    m3_error error;
    size_t output_bytes = vector.count * m3_dtype_size(output_dtype);
    m3_status metal_status;
    m3_status host_status;

    m3_tensor_view_init(&metal_input);
    m3_tensor_view_init(&metal_output);
    m3_tensor_view_init(&host_input);
    m3_tensor_view_init(&host_output);
    (void)memset(sentinel, 0xa5, sizeof(sentinel));
    M3_TEST_EXPECT(test, output_bytes <= sizeof(sentinel),
                   "CAST parity vector fits its sentinel buffer");
    if (output_bytes > sizeof(sentinel)) {
        return false;
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(metal, &metal_input, input_dtype, 1U,
                                     shape, vector.data) &&
                       m3_op_test_tensor(metal, &metal_output, output_dtype,
                                         1U, shape, sentinel) &&
                       m3_op_test_tensor(host, &host_input, input_dtype, 1U,
                                         shape, vector.data) &&
                       m3_op_test_tensor(host, &host_output, output_dtype,
                                         1U, shape, sentinel),
                   "create host and Metal CAST parity tensors");
    if (metal_output.storage == NULL || host_output.storage == NULL) {
        return false;
    }
    metal_command.kind = M3_OP_CAST;
    metal_command.descriptor.cast.input = &metal_input;
    metal_command.descriptor.cast.output = &metal_output;
    host_command.kind = M3_OP_CAST;
    host_command.descriptor.cast.input = &host_input;
    host_command.descriptor.cast.output = &host_output;
    host_status = m3_backend_execute(host->backend, &host_command, 1U, NULL,
                                     &error);
    metal_status = m3_backend_execute(metal->backend, &metal_command, 1U,
                                      NULL, &error);
    M3_TEST_EXPECT(test,
                   host_status == M3_STATUS_OK &&
                       metal_status == M3_STATUS_OK,
                   "host and Metal accept the CAST pair");
    M3_TEST_EXPECT(test,
                   memcmp(m3_storage_const_data(metal_output.storage),
                          m3_storage_const_data(host_output.storage),
                          output_bytes) == 0,
                   "Metal CAST bits exactly match the host");
    return host_status == M3_STATUS_OK && metal_status == M3_STATUS_OK;
}

static bool m3_test_metal_fixed_rounding(m3_test_context *test,
                                         m3_op_test_fixture *fixture)
{
    const uint64_t seven[] = {7U};
    const uint64_t four[] = {4U};
    const float f16_source[] = {
        1.00048828125F, 1.00146484375F, INFINITY, -INFINITY, NAN,
        0x1p-24F, 0x1p-25F
    };
    const uint16_t expected_f16[] = {
        0x3c00U, 0x3c02U, 0x7c00U, 0xfc00U, 0x7e00U, 0x0001U,
        0x0000U
    };
    const float bf16_source[] = {
        1.00390625F, 1.01171875F, INFINITY, NAN
    };
    const uint16_t expected_bf16[] = {
        0x3f80U, 0x3f82U, 0x7f80U, 0x7fc0U
    };
    const uint16_t zeros_f16[7] = {0U};
    const uint16_t zeros_bf16[4] = {0U};
    m3_tensor_view f16_input;
    m3_tensor_view f16_output;
    m3_tensor_view bf16_input;
    m3_tensor_view bf16_output;
    m3_command commands[2];
    m3_error error;

    m3_tensor_view_init(&f16_input);
    m3_tensor_view_init(&f16_output);
    m3_tensor_view_init(&bf16_input);
    m3_tensor_view_init(&bf16_output);
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(fixture, &f16_input, M3_DTYPE_F32, 1U,
                                     seven, f16_source) &&
                       m3_op_test_tensor(fixture, &f16_output, M3_DTYPE_F16,
                                         1U, seven, zeros_f16) &&
                       m3_op_test_tensor(fixture, &bf16_input, M3_DTYPE_F32,
                                         1U, four, bf16_source) &&
                       m3_op_test_tensor(fixture, &bf16_output,
                                         M3_DTYPE_BF16, 1U, four,
                                         zeros_bf16),
                   "create fixed Metal rounding tensors");
    if (bf16_output.storage == NULL) {
        return false;
    }
    commands[0].kind = M3_OP_CAST;
    commands[0].descriptor.cast.input = &f16_input;
    commands[0].descriptor.cast.output = &f16_output;
    commands[1].kind = M3_OP_CAST;
    commands[1].descriptor.cast.input = &bf16_input;
    commands[1].descriptor.cast.output = &bf16_output;
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture->backend, commands, 2U, NULL,
                                      &error) == M3_STATUS_OK,
                   "execute fixed F16 and BF16 Metal casts");
    M3_TEST_EXPECT(test,
                   memcmp(m3_op_test_u16(&f16_output), expected_f16,
                          sizeof(expected_f16)) == 0,
                   "Metal F16 handles RN-even, infinities, NaN, and subnormals");
    M3_TEST_EXPECT(test,
                   memcmp(m3_op_test_u16(&bf16_output), expected_bf16,
                          sizeof(expected_bf16)) == 0,
                   "Metal BF16 handles RN-even, infinity, and NaN");
    return true;
}

static bool m3_test_metal_cast_aliases(m3_test_context *test,
                                       m3_op_test_fixture *fixture)
{
    const uint64_t two[] = {2U};
    const uint64_t three[] = {3U};
    const float float_values[] = {2.9F, -2.9F};
    const uint16_t half_values[] = {0x3c00U, 0x7c01U, 0x0001U};
    const uint16_t expected_brain[] = {0x3f80U, 0x7fc0U, 0x3380U};
    m3_storage *float_storage = NULL;
    m3_storage *half_storage = NULL;
    m3_tensor_view float_input;
    m3_tensor_view integer_output;
    m3_tensor_view half_input;
    m3_tensor_view brain_output;
    m3_command command;
    m3_error error;

    m3_tensor_view_init(&float_input);
    m3_tensor_view_init(&integer_output);
    m3_tensor_view_init(&half_input);
    m3_tensor_view_init(&brain_output);
    M3_TEST_EXPECT(test,
                   m3_op_test_storage(fixture, sizeof(float_values),
                                      &float_storage) &&
                       m3_storage_write(float_storage, 0U, float_values,
                                        sizeof(float_values), &error) ==
                           M3_STATUS_OK &&
                       m3_tensor_view_contiguous(
                           &float_input, float_storage, M3_DTYPE_F32, 1U, two,
                           0U, &error) == M3_STATUS_OK &&
                       m3_tensor_view_contiguous(
                           &integer_output, float_storage, M3_DTYPE_I32, 1U,
                           two, 0U, &error) == M3_STATUS_OK,
                   "create exact F32-to-I32 Metal alias");
    command.kind = M3_OP_CAST;
    command.descriptor.cast.input = &float_input;
    command.descriptor.cast.output = &integer_output;
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture->backend, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK &&
                       m3_op_test_i32(&integer_output)[0] == 2 &&
                       m3_op_test_i32(&integer_output)[1] == -2,
                   "Metal casts an exact four-byte alias in place");
    M3_TEST_EXPECT(test,
                   m3_op_test_storage(fixture, sizeof(half_values),
                                      &half_storage) &&
                       m3_storage_write(half_storage, 0U, half_values,
                                        sizeof(half_values), &error) ==
                           M3_STATUS_OK &&
                       m3_tensor_view_contiguous(
                           &half_input, half_storage, M3_DTYPE_F16, 1U, three,
                           0U, &error) == M3_STATUS_OK &&
                       m3_tensor_view_contiguous(
                           &brain_output, half_storage, M3_DTYPE_BF16, 1U,
                           three, 0U, &error) == M3_STATUS_OK,
                   "create exact F16-to-BF16 Metal alias");
    command.descriptor.cast.input = &half_input;
    command.descriptor.cast.output = &brain_output;
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture->backend, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK &&
                       memcmp(m3_op_test_u16(&brain_output), expected_brain,
                              sizeof(expected_brain)) == 0,
                   "Metal casts an exact two-byte alias in place");
    return true;
}

static bool m3_test_metal_invalid_i32_atomic(
    m3_test_context *test, m3_op_test_fixture *fixture)
{
    const uint64_t shape[] = {4U};
    const uint64_t matrix_shape[] = {1U, 1U};
    const float invalid_values[] = {1.0F, NAN, 2.0F, INFINITY};
    const float valid_values[] = {9.0F, 8.0F, 7.0F, 6.0F};
    const int32_t sentinels[] = {77, 78, 79, 80};
    m3_tensor_view input;
    m3_tensor_view middle;
    m3_tensor_view output;
    m3_tensor_view dense_input;
    m3_tensor_view dense_output;
    m3_command command;
    m3_command commands[2];
    m3_error error;
    m3_status status;

    m3_tensor_view_init(&input);
    m3_tensor_view_init(&middle);
    m3_tensor_view_init(&output);
    m3_tensor_view_init(&dense_input);
    m3_tensor_view_init(&dense_output);
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(fixture, &input, M3_DTYPE_F32, 1U,
                                     shape, invalid_values) &&
                       m3_op_test_tensor(fixture, &middle, M3_DTYPE_F32, 1U,
                                         shape, valid_values) &&
                       m3_op_test_tensor(fixture, &output, M3_DTYPE_I32, 1U,
                                         shape, sentinels),
                   "create invalid float-to-I32 Metal cast");
    if (output.storage == NULL) {
        return false;
    }
    command.kind = M3_OP_CAST;
    command.descriptor.cast.input = &input;
    command.descriptor.cast.output = &output;
    status = m3_backend_execute(fixture->backend, &command, 1U, NULL, &error);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_OUT_OF_RANGE &&
                       memcmp(m3_op_test_i32(&output), sentinels,
                              sizeof(sentinels)) == 0,
                   "invalid float-to-I32 preflight is atomic");
    status = m3_backend_execute(fixture->backend, &command, 1U, NULL, NULL);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_OUT_OF_RANGE &&
                       memcmp(m3_op_test_i32(&output), sentinels,
                              sizeof(sentinels)) == 0,
                   "invalid Metal cast accepts a null error sink");
    commands[0].kind = M3_OP_COPY;
    commands[0].descriptor.copy.input = &input;
    commands[0].descriptor.copy.output = &middle;
    commands[1].kind = M3_OP_CAST;
    commands[1].descriptor.cast.input = &middle;
    commands[1].descriptor.cast.output = &output;
    status = m3_backend_execute(fixture->backend, commands, 2U, NULL, &error);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_UNSUPPORTED &&
                       memcmp(m3_op_test_f32(&middle), valid_values,
                              sizeof(valid_values)) == 0 &&
                       memcmp(m3_op_test_i32(&output), sentinels,
                              sizeof(sentinels)) == 0,
                   "produced float-to-I32 input is rejected before commit");
    M3_TEST_EXPECT(test,
                   m3_tensor_view_contiguous(
                       &dense_input, input.storage, M3_DTYPE_F32, 2U,
                       matrix_shape, 0U, &error) == M3_STATUS_OK &&
                       m3_tensor_view_contiguous(
                           &dense_output, middle.storage, M3_DTYPE_F32, 2U,
                           matrix_shape, 0U, &error) == M3_STATUS_OK,
                   "create ordering-test dense views");
    commands[0].kind = M3_OP_MATMUL;
    commands[0].descriptor.matmul.left = &dense_input;
    commands[0].descriptor.matmul.right = &dense_input;
    commands[0].descriptor.matmul.output = &dense_output;
    commands[1] = command;
    status = m3_backend_execute(fixture->backend, commands, 2U, NULL, &error);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_UNSUPPORTED &&
                       memcmp(m3_op_test_f32(&middle), valid_values,
                              sizeof(valid_values)) == 0 &&
                       memcmp(m3_op_test_i32(&output), sentinels,
                              sizeof(sentinels)) == 0,
                   "earlier unsupported command wins without mutation");
    commands[0] = command;
    commands[1].kind = M3_OP_MATMUL;
    commands[1].descriptor.matmul.left = &dense_input;
    commands[1].descriptor.matmul.right = &dense_input;
    commands[1].descriptor.matmul.output = &dense_output;
    status = m3_backend_execute(fixture->backend, commands, 2U, NULL, &error);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_OUT_OF_RANGE &&
                       memcmp(m3_op_test_f32(&middle), valid_values,
                              sizeof(valid_values)) == 0 &&
                       memcmp(m3_op_test_i32(&output), sentinels,
                              sizeof(sentinels)) == 0,
                   "earlier invalid cast wins without mutation");
    return true;
}

void m3_test_metal_cast_commands(m3_test_context *test)
{
    m3_op_test_fixture metal;
    m3_op_test_fixture host;
    int input_value;
    int output_value;

    if (!m3_test_metal_cast_fixture_init(test, &metal)) {
        return;
    }
    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&host),
                   "create host backend for Metal CAST parity");
    if (host.backend == NULL) {
        m3_op_test_fixture_dispose(&metal);
        return;
    }
    for (input_value = (int)M3_DTYPE_F32;
         input_value <= (int)M3_DTYPE_I32; ++input_value) {
        for (output_value = (int)M3_DTYPE_F32;
             output_value <= (int)M3_DTYPE_I32; ++output_value) {
            if (!m3_test_metal_cast_pair(
                    test, &metal, &host, (m3_dtype)input_value,
                    (m3_dtype)output_value)) {
                m3_op_test_fixture_dispose(&host);
                m3_op_test_fixture_dispose(&metal);
                return;
            }
        }
    }
    if (!m3_test_metal_fixed_rounding(test, &metal) ||
        !m3_test_metal_cast_aliases(test, &metal) ||
        !m3_test_metal_invalid_i32_atomic(test, &metal)) {
        m3_op_test_fixture_dispose(&host);
        m3_op_test_fixture_dispose(&metal);
        return;
    }
    m3_op_test_fixture_dispose(&host);
    m3_op_test_fixture_dispose(&metal);
}

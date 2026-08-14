/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

void m3_test_op_conversions(m3_test_context *test)
{
    const uint64_t seven[] = {7U};
    const uint64_t two[] = {2U};
    const uint64_t four[] = {4U};
    const float f16_source[] = {
        1.00048828125F, 1.00146484375F, INFINITY, -INFINITY, NAN,
        0x1p-24F, 0x1p-25F
    };
    const uint16_t zeros16[7] = {0U};
    const uint16_t expected_f16[] = {
        0x3c00U, 0x3c02U, 0x7c00U, 0xfc00U, 0x7e00U, 0x0001U,
        0x0000U
    };
    const float bf16_source[] = {
        1.00390625F, 1.01171875F, INFINITY, NAN
    };
    const uint16_t zeros_bf16[4] = {0U};
    const int32_t zeros_i32[] = {0, 0};
    const float valid_i32_source[] = {2.9F, -2.9F};
    const float invalid_i32_source[] = {
        NAN, INFINITY, -INFINITY, 2147483648.0F
    };
    const int32_t sentinel_i32[] = {77, 77, 77, 77};
    m3_tensor_view f32_input;
    m3_tensor_view f16_output;
    m3_tensor_view f16_roundtrip;
    m3_tensor_view bf32_input;
    m3_tensor_view bf16_output;
    m3_tensor_view valid_input;
    m3_tensor_view valid_output;
    m3_tensor_view invalid_input;
    m3_tensor_view invalid_output;
    m3_op_test_fixture fixture;
    m3_command command;
    m3_error error;
    uint16_t *half_bits;
    uint16_t *brain_bits;
    float *roundtrip;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create conversion fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &f32_input, M3_DTYPE_F32, 1U,
                                     seven, f16_source) &&
                       m3_op_test_tensor(&fixture, &f16_output, M3_DTYPE_F16,
                                         1U, seven, zeros16) &&
                       m3_op_test_tensor(&fixture, &f16_roundtrip,
                                         M3_DTYPE_F32, 1U, seven,
                                         f16_source),
                   "create F16 conversion tensors");
    command.kind = M3_OP_CAST;
    command.descriptor.cast.input = &f32_input;
    command.descriptor.cast.output = &f16_output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                       M3_STATUS_OK,
                   "cast F32 values to F16");
    half_bits = m3_op_test_u16(&f16_output);
    M3_TEST_EXPECT(test,
                   half_bits[0] == expected_f16[0] &&
                       half_bits[1] == expected_f16[1],
                   "F16 halfway cases round to nearest even");
    M3_TEST_EXPECT(test,
                   half_bits[2] == expected_f16[2] &&
                       half_bits[3] == expected_f16[3] &&
                       (half_bits[4] & 0x7c00U) == 0x7c00U &&
                       (half_bits[4] & 0x03ffU) != 0U,
                   "F16 infinity and NaN encodings are preserved");
    M3_TEST_EXPECT(test,
                   half_bits[5] == expected_f16[5] &&
                       half_bits[6] == expected_f16[6],
                   "F16 minimum subnormal and halfway-to-zero use RN-even");
    command.descriptor.cast.input = &f16_output;
    command.descriptor.cast.output = &f16_roundtrip;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OK,
                   "cast F16 values back to F32");
    roundtrip = m3_op_test_f32(&f16_roundtrip);
    M3_TEST_EXPECT(test,
                   roundtrip[0] == 1.0F &&
                       roundtrip[1] == 1.001953125F &&
                       isinf(roundtrip[2]) && roundtrip[2] > 0.0F &&
                       isinf(roundtrip[3]) && roundtrip[3] < 0.0F &&
                       isnan(roundtrip[4]) && roundtrip[5] == 0x1p-24F &&
                       roundtrip[6] == 0.0F,
                   "F16 decoding covers finite, infinity, and NaN values");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &bf32_input, M3_DTYPE_F32, 1U,
                                     four, bf16_source) &&
                       m3_op_test_tensor(&fixture, &bf16_output,
                                         M3_DTYPE_BF16, 1U, four,
                                         zeros_bf16),
                   "create BF16 conversion tensors");
    command.descriptor.cast.input = &bf32_input;
    command.descriptor.cast.output = &bf16_output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                       M3_STATUS_OK,
                   "cast F32 values to BF16");
    brain_bits = m3_op_test_u16(&bf16_output);
    M3_TEST_EXPECT(test,
                   brain_bits[0] == 0x3f80U && brain_bits[1] == 0x3f82U,
                   "BF16 halfway cases round to nearest even");
    M3_TEST_EXPECT(test,
                   brain_bits[2] == 0x7f80U &&
                       (brain_bits[3] & 0x7f80U) == 0x7f80U &&
                       (brain_bits[3] & 0x007fU) != 0U,
                   "BF16 infinity and NaN encodings are preserved");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &valid_input, M3_DTYPE_F32,
                                     1U, two, valid_i32_source) &&
                       m3_op_test_tensor(&fixture, &valid_output,
                                         M3_DTYPE_I32, 1U, two, zeros_i32),
                   "create valid integer cast tensors");
    command.descriptor.cast.input = &valid_input;
    command.descriptor.cast.output = &valid_output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OK &&
                       m3_op_test_i32(&valid_output)[0] == 2 &&
                       m3_op_test_i32(&valid_output)[1] == -2,
                   "float-to-I32 truncates toward zero");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &invalid_input, M3_DTYPE_F32,
                                     1U, four, invalid_i32_source) &&
                       m3_op_test_tensor(&fixture, &invalid_output,
                                         M3_DTYPE_I32, 1U, four,
                                         sentinel_i32),
                   "create invalid integer cast tensors");
    command.descriptor.cast.input = &invalid_input;
    command.descriptor.cast.output = &invalid_output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OUT_OF_RANGE &&
                       memcmp(m3_op_test_i32(&invalid_output), sentinel_i32,
                              sizeof(sentinel_i32)) == 0,
                   "invalid float-to-I32 cast leaves all output unchanged");
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, NULL) ==
                           M3_STATUS_OUT_OF_RANGE &&
                       memcmp(m3_op_test_i32(&invalid_output), sentinel_i32,
                              sizeof(sentinel_i32)) == 0,
                   "invalid data preflight works without an error sink");
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_op_broadcast_embedding(m3_test_context *test)
{
    const uint64_t matrix_shape[] = {2U, 3U};
    const uint64_t vector_shape[] = {3U};
    const uint64_t scalar_shape[] = {1U};
    const size_t transposed_strides[] = {4U, 8U};
    const float transposed_backing[] = {1, 4, 2, 5, 3, 6};
    const float vector[] = {10, 20, 30};
    const float zeros6[6] = {0};
    const float scalar[] = {2};
    const float expected_add[] = {11, 22, 33, 14, 25, 36};
    const uint64_t ids_shape[] = {2U};
    const uint64_t table_shape[] = {3U, 2U};
    const uint64_t embedded_shape[] = {2U, 2U};
    const int32_t ids[] = {2, 0};
    const float table[] = {1, 2, 3, 4, 5, 6};
    const float sentinel4[] = {-9, -9, -9, -9};
    const int32_t bad_ids[] = {-1, 3};
    const uint64_t empty_shape[] = {0U};
    const float empty_scalar[] = {3.0F};
    m3_op_test_fixture fixture;
    m3_storage *transposed_storage = NULL;
    m3_storage *overlap_storage = NULL;
    m3_tensor_view left;
    m3_tensor_view right;
    m3_tensor_view output;
    m3_tensor_view scale;
    m3_tensor_view overlap_input;
    m3_tensor_view overlap_output;
    m3_tensor_view id_view;
    m3_tensor_view table_view;
    m3_tensor_view embedded;
    m3_tensor_view empty_left;
    m3_tensor_view empty_right;
    m3_tensor_view empty_output;
    m3_command command;
    m3_error error;
    float *values;
    size_t index;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create broadcast fixture");
    if (fixture.backend == NULL) {
        return;
    }
    m3_tensor_view_init(&left);
    m3_tensor_view_init(&overlap_input);
    m3_tensor_view_init(&overlap_output);
    M3_TEST_EXPECT(test,
                   m3_op_test_storage(&fixture, sizeof(transposed_backing),
                                      &transposed_storage) &&
                       m3_storage_write(transposed_storage, 0U,
                                        transposed_backing,
                                        sizeof(transposed_backing), &error) ==
                           M3_STATUS_OK &&
                       m3_tensor_view_strided(
                           &left, transposed_storage, M3_DTYPE_F32, 2U,
                           matrix_shape, transposed_strides, 0U, &error) ==
                           M3_STATUS_OK &&
                       m3_op_test_tensor(&fixture, &right, M3_DTYPE_F32, 1U,
                                         vector_shape, vector) &&
                       m3_op_test_tensor(&fixture, &output, M3_DTYPE_F32, 2U,
                                         matrix_shape, zeros6),
                   "create strided broadcast tensors");
    command.kind = M3_OP_COPY;
    command.descriptor.copy.input = &left;
    command.descriptor.copy.output = &output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OK &&
                       m3_op_test_f32(&output)[0] == 1.0F &&
                       m3_op_test_f32(&output)[5] == 6.0F,
                   "copy gathers a strided input into contiguous output");
    command.kind = M3_OP_ADD;
    command.descriptor.add.left = &left;
    command.descriptor.add.right = &right;
    command.descriptor.add.output = &output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                       M3_STATUS_OK,
                   "execute right-aligned broadcast add");
    values = m3_op_test_f32(&output);
    for (index = 0U; index < 6U; ++index) {
        M3_TEST_EXPECT_F32(test, values[index], expected_add[index], 0.0F,
                           0.0F, "strided broadcast fixture value");
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &scale, M3_DTYPE_F32, 1U,
                                     scalar_shape, scalar),
                   "create scalar broadcast input");
    command.kind = M3_OP_MUL;
    command.descriptor.mul.left = &output;
    command.descriptor.mul.right = &scale;
    command.descriptor.mul.output = &output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OK &&
                       values[0] == 22.0F && values[5] == 72.0F,
                   "exact output alias is allowed for shaped broadcast input");
    M3_TEST_EXPECT(test,
                   m3_op_test_storage(&fixture, 20U, &overlap_storage) &&
                       m3_tensor_view_contiguous(
                           &overlap_input, overlap_storage, M3_DTYPE_F32, 1U,
                           embedded_shape, 0U, &error) == M3_STATUS_OK &&
                       m3_tensor_view_contiguous(
                           &overlap_output, overlap_storage, M3_DTYPE_F32, 1U,
                           embedded_shape, 4U, &error) == M3_STATUS_OK,
                   "create partially overlapping views");
    command.kind = M3_OP_COPY;
    command.descriptor.copy.input = &overlap_input;
    command.descriptor.copy.output = &overlap_output;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "partial storage overlap is rejected structurally");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &id_view, M3_DTYPE_I32, 1U,
                                     ids_shape, ids) &&
                       m3_op_test_tensor(&fixture, &table_view, M3_DTYPE_F32,
                                         2U, table_shape, table) &&
                       m3_op_test_tensor(&fixture, &embedded, M3_DTYPE_F32,
                                         2U, embedded_shape, sentinel4),
                   "create embedding tensors");
    command.kind = M3_OP_EMBEDDING;
    command.descriptor.embedding.ids = &id_view;
    command.descriptor.embedding.table = &table_view;
    command.descriptor.embedding.output = &embedded;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OK &&
                       m3_op_test_f32(&embedded)[0] == 5.0F &&
                       m3_op_test_f32(&embedded)[1] == 6.0F &&
                       m3_op_test_f32(&embedded)[2] == 1.0F &&
                       m3_op_test_f32(&embedded)[3] == 2.0F,
                   "embedding gathers rows in ID order");
    (void)m3_storage_write(id_view.storage, 0U, bad_ids, sizeof(bad_ids),
                           &error);
    (void)m3_storage_write(embedded.storage, 0U, sentinel4,
                           sizeof(sentinel4), &error);
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OUT_OF_RANGE &&
                       memcmp(m3_op_test_f32(&embedded), sentinel4,
                              sizeof(sentinel4)) == 0,
                   "negative and out-of-range IDs leave embedding unchanged");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &empty_left, M3_DTYPE_F32, 1U,
                                     empty_shape, NULL) &&
                       m3_op_test_tensor(&fixture, &empty_right,
                                         M3_DTYPE_F32, 1U, scalar_shape,
                                         empty_scalar) &&
                       m3_op_test_tensor(&fixture, &empty_output,
                                         M3_DTYPE_F32, 1U, empty_shape,
                                         NULL),
                   "create empty-dimension broadcast tensors");
    command.kind = M3_OP_ADD;
    command.descriptor.add.left = &empty_left;
    command.descriptor.add.right = &empty_right;
    command.descriptor.add.output = &empty_output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OK &&
                       empty_output.metadata.element_count == 0U,
                   "broadcasting dimensions zero and one produces zero");
    m3_op_test_fixture_dispose(&fixture);
}

/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"

#include <stdint.h>

static m3_command m3_test_resize_command(const m3_tensor_view *input,
                                         m3_tensor_view *output)
{
    m3_command command = {0};

    command.kind = M3_OP_NEAREST_RESIZE1D;
    command.descriptor.nearest_resize1d.input = input;
    command.descriptor.nearest_resize1d.output = output;
    return command;
}

void m3_test_op_nearest_resize1d(m3_test_context *test)
{
    const uint64_t input_shape[] = {1U, 1U, 3U};
    const uint64_t upsample_shape[] = {1U, 1U, 8U};
    const uint16_t input_values[] = {0x4900U, 0x4d00U, 0x4f80U};
    const uint16_t upsample_zeros[8] = {0};
    const uint16_t upsample_expected[] = {
        0x4120U, 0x4120U, 0x4120U, 0x41a0U,
        0x41a0U, 0x41a0U, 0x41f0U, 0x41f0U
    };
    const uint64_t downsample_input_shape[] = {1U, 1U, 5U};
    const uint64_t downsample_output_shape[] = {1U, 1U, 2U};
    const float downsample_values[] = {1, 2, 3, 4, 5};
    const float downsample_zeros[2] = {0};
    const float downsample_expected[] = {1, 3};
    const float identity_values[] = {7, 8, 9};
    m3_op_test_fixture fixture;
    m3_tensor_view input;
    m3_tensor_view upsample_output;
    m3_tensor_view downsample_input;
    m3_tensor_view downsample_output;
    m3_tensor_view identity;
    m3_command command;
    m3_error error;
    size_t scratch_bytes = SIZE_MAX;
    size_t index;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create nearest resize fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &input, M3_DTYPE_F16, 3U,
                          input_shape, input_values) &&
            m3_op_test_tensor(&fixture, &upsample_output, M3_DTYPE_BF16, 3U,
                              upsample_shape, upsample_zeros),
        "create mixed-dtype nearest upsample tensors");
    command = m3_test_resize_command(&input, &upsample_output);
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U,
                                      &scratch_bytes, &error) ==
                           M3_STATUS_OK &&
                       scratch_bytes == 0U,
                   "execute nearest upsample without scratch");
    for (index = 0U; index < 8U; ++index) {
        M3_TEST_EXPECT(test, m3_op_test_u16(&upsample_output)[index] ==
                                 upsample_expected[index],
                       "nearest upsample hard-coded mapping");
    }

    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &downsample_input, M3_DTYPE_F32, 3U,
                          downsample_input_shape, downsample_values) &&
            m3_op_test_tensor(&fixture, &downsample_output, M3_DTYPE_F32, 3U,
                              downsample_output_shape, downsample_zeros),
        "create nearest downsample tensors");
    command = m3_test_resize_command(&downsample_input, &downsample_output);
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK,
                   "execute nearest downsample");
    for (index = 0U; index < 2U; ++index) {
        M3_TEST_EXPECT_F32(test, m3_op_test_f32(&downsample_output)[index],
                           downsample_expected[index], 0.0F, 0.0F,
                           "nearest downsample hard-coded mapping");
    }

    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &identity, M3_DTYPE_F32, 3U,
                                     input_shape, identity_values),
                   "create identity nearest resize tensor");
    command = m3_test_resize_command(&identity, &identity);
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL,
                                      &error) == M3_STATUS_OK,
                   "allow exact identity nearest resize view");
    for (index = 0U; index < 3U; ++index) {
        M3_TEST_EXPECT_F32(test, m3_op_test_f32(&identity)[index],
                           identity_values[index], 0.0F, 0.0F,
                           "identity nearest resize preserves value");
    }
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_op_resize_safety(m3_test_context *test)
{
    const uint64_t shape[] = {1U, 1U, 3U};
    const size_t strides[] = {12U, 12U, 4U};
    const float backing_values[] = {1, 2, 3, 4};
    const uint64_t empty_huge_input_shape[] = {0U, 1U, UINT64_MAX};
    const uint64_t empty_huge_output_shape[] = {
        0U, 1U, UINT64_MAX - 1U
    };
    m3_op_test_fixture fixture;
    m3_storage *overlap_storage = NULL;
    m3_storage *reinterpret_storage = NULL;
    m3_tensor_view overlap_input;
    m3_tensor_view overlap_output;
    m3_tensor_view f16_input;
    m3_tensor_view bf16_output;
    m3_tensor_view huge_input;
    m3_tensor_view huge_output;
    m3_command command;
    m3_error error;
    size_t scratch_bytes = SIZE_MAX;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create nearest resize safety fixture");
    if (fixture.backend == NULL) {
        return;
    }
    m3_tensor_view_init(&overlap_input);
    m3_tensor_view_init(&overlap_output);
    M3_TEST_EXPECT(
        test,
        m3_op_test_storage(&fixture, sizeof(backing_values),
                           &overlap_storage) &&
            m3_storage_write(overlap_storage, 0U, backing_values,
                             sizeof(backing_values), &error) ==
                M3_STATUS_OK &&
            m3_tensor_view_strided(&overlap_input, overlap_storage,
                                   M3_DTYPE_F32, 3U, shape, strides, 0U,
                                   &error) == M3_STATUS_OK &&
            m3_tensor_view_strided(&overlap_output, overlap_storage,
                                   M3_DTYPE_F32, 3U, shape, strides, 4U,
                                   &error) == M3_STATUS_OK,
        "create partially overlapping nearest resize views");
    command = m3_test_resize_command(&overlap_input, &overlap_output);
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject partial nearest resize overlap");

    m3_tensor_view_init(&f16_input);
    m3_tensor_view_init(&bf16_output);
    M3_TEST_EXPECT(
        test,
        m3_op_test_storage(&fixture, 6U, &reinterpret_storage) &&
            m3_tensor_view_contiguous(&f16_input, reinterpret_storage,
                                      M3_DTYPE_F16, 3U, shape, 0U,
                                      &error) == M3_STATUS_OK &&
            m3_tensor_view_contiguous(&bf16_output, reinterpret_storage,
                                      M3_DTYPE_BF16, 3U, shape, 0U,
                                      &error) == M3_STATUS_OK,
        "create exact mixed-dtype nearest resize views");
    command = m3_test_resize_command(&f16_input, &bf16_output);
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject exact overlap that would require a cast");

    M3_TEST_EXPECT(
        test,
        m3_op_test_tensor(&fixture, &huge_input, M3_DTYPE_F32, 3U,
                          empty_huge_input_shape, NULL) &&
            m3_op_test_tensor(&fixture, &huge_output, M3_DTYPE_F32, 3U,
                              empty_huge_output_shape, NULL),
        "create empty huge-length nearest resize tensors");
    command = m3_test_resize_command(&huge_input, &huge_output);
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U,
                                      &scratch_bytes, &error) ==
                           M3_STATUS_OK &&
                       scratch_bytes == 0U,
                   "resize validates huge metadata without multiplication");
    m3_op_test_fixture_dispose(&fixture);
}

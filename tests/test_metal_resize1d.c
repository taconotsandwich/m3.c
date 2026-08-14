/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "metal_convolution_test.h"

#include <stdint.h>

static m3_command m3_test_metal_resize_command(
    const m3_tensor_view *input, m3_tensor_view *output)
{
    m3_command command = {0};

    command.kind = M3_OP_NEAREST_RESIZE1D;
    command.descriptor.nearest_resize1d.input = input;
    command.descriptor.nearest_resize1d.output = output;
    return command;
}

static bool m3_test_metal_resize_mixed(
    m3_test_context *test, m3_metal_convolution_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 1U, 3U};
    const uint64_t output_shape[] = {1U, 1U, 8U};
    const uint16_t input_values[] = {0x4900U, 0x4d00U, 0x4f80U};
    const uint16_t zeros[8] = {0};
    const uint16_t expected[] = {
        0x4120U, 0x4120U, 0x4120U, 0x41a0U,
        0x41a0U, 0x41a0U, 0x41f0U, 0x41f0U
    };
    m3_metal_convolution_view input;
    m3_metal_convolution_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(test,
                   m3_metal_convolution_tensor(
                       fixture, &input, M3_DTYPE_F16, 3U, input_shape,
                       input_values) &&
                       m3_metal_convolution_tensor(
                           fixture, &output, M3_DTYPE_BF16, 3U,
                           output_shape, zeros),
                   "create mixed-dtype Metal nearest upsample");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_resize_command(&input.host, &output.host);
    metal = m3_test_metal_resize_command(&input.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_convolution_equal(&output),
                   "mixed Metal nearest resize matches host exactly");
    for (index = 0U; index < 8U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_op_test_u16(&output.metal)[index] ==
                           expected[index],
                       "Metal nearest upsample has expected BF16 bits");
    }
    return true;
}

static bool m3_test_metal_resize_strided_rows(
    m3_test_context *test, m3_metal_convolution_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 2U, 7U};
    const uint64_t output_shape[] = {1U, 2U, 3U};
    const size_t input_strides[] = {128U, 64U, 8U};
    const float input_backing[] = {
        -90, 1, -91, 2, -92, 3, -93, 4, -94, 5,
        -95, 6, -96, 7, -97, -98, -99, 11, -80, 12,
        -81, 13, -82, 14, -83, 15, -84, 16, -85, 17
    };
    const float sentinels[] = {-9, -9, -9, -9, -9, -9};
    const float expected[] = {1, 3, 5, 11, 13, 15};
    m3_metal_convolution_view input;
    m3_metal_convolution_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(
        test,
        m3_metal_convolution_strided(
            fixture, &input, M3_DTYPE_F32, 3U, input_shape,
            input_strides, sizeof(float), sizeof(input_backing),
            input_backing) &&
            m3_metal_convolution_tensor(
                fixture, &output, M3_DTYPE_F32, 3U, output_shape,
                sentinels),
        "create multi-row strided Metal nearest resize");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_resize_command(&input.host, &output.host);
    metal = m3_test_metal_resize_command(&input.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_convolution_equal(&output),
                   "Metal resize uses host quotient-remainder boundaries");
    for (index = 0U; index < 6U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_op_test_f32(&output.metal)[index] ==
                           expected[index],
                       "Metal nearest downsample has expected source index");
    }
    return true;
}

static bool m3_test_metal_resize_low_precision(
    m3_test_context *test, m3_metal_convolution_fixture *fixture,
    m3_dtype dtype, const uint64_t *input_shape,
    const uint64_t *output_shape, const uint16_t *input_values,
    const uint16_t *output_values, const char *message)
{
    m3_metal_convolution_view input;
    m3_metal_convolution_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_convolution_tensor(
                       fixture, &input, dtype, 3U, input_shape,
                       input_values) &&
                       m3_metal_convolution_tensor(
                           fixture, &output, dtype, 3U, output_shape,
                           output_values),
                   message);
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_resize_command(&input.host, &output.host);
    metal = m3_test_metal_resize_command(&input.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_convolution_equal(&output),
                   message);
    return true;
}

static bool m3_test_metal_resize_identity(
    m3_test_context *test, m3_metal_convolution_fixture *fixture)
{
    const uint64_t shape[] = {1U, 2U, 3U};
    const float values[] = {1, 2, 3, 4, 5, 6};
    m3_metal_convolution_view view;
    m3_command host;
    m3_command metal;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_convolution_tensor(
                       fixture, &view, M3_DTYPE_F32, 3U, shape, values),
                   "create exact-alias Metal nearest resize");
    if (view.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_resize_command(&view.host, &view.host);
    metal = m3_test_metal_resize_command(&view.metal, &view.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_convolution_equal(&view),
                   "Metal nearest identity is safe in place");
    return true;
}

static bool m3_test_metal_resize_empty(
    m3_test_context *test, m3_metal_convolution_fixture *fixture)
{
    const uint64_t huge_input_shape[] = {0U, 1U, UINT64_MAX};
    const uint64_t huge_output_shape[] = {0U, 1U, UINT64_MAX - 1U};
    const uint64_t channel_input_shape[] = {1U, 0U, 1U};
    const uint64_t channel_output_shape[] = {1U, 0U, 7U};
    m3_metal_convolution_view huge_input;
    m3_metal_convolution_view huge_output = {0};
    m3_metal_convolution_view channel_input;
    m3_metal_convolution_view channel_output = {0};
    m3_command host[2];
    m3_command metal[2];
    m3_error error;

    M3_TEST_EXPECT(
        test,
        m3_metal_convolution_tensor(
            fixture, &huge_input, M3_DTYPE_F32, 3U, huge_input_shape,
            NULL) &&
            m3_metal_convolution_tensor(
                fixture, &huge_output, M3_DTYPE_F32, 3U,
                huge_output_shape, NULL) &&
            m3_metal_convolution_tensor(
                fixture, &channel_input, M3_DTYPE_F32, 3U,
                channel_input_shape, NULL) &&
            m3_metal_convolution_tensor(
                fixture, &channel_output, M3_DTYPE_F32, 3U,
                channel_output_shape, NULL),
        "create empty Metal resize dimensions");
    if (huge_input.host.storage == NULL) {
        return false;
    }
    host[0] = m3_test_metal_resize_command(
        &huge_input.host, &huge_output.host);
    metal[0] = m3_test_metal_resize_command(
        &huge_input.metal, &huge_output.metal);
    host[1] = m3_test_metal_resize_command(
        &channel_input.host, &channel_output.host);
    metal[1] = m3_test_metal_resize_command(
        &channel_input.metal, &channel_output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, host, metal, 2U, &error) &&
                       m3_metal_convolution_equal(&huge_output) &&
                       m3_metal_convolution_equal(&channel_output),
                   "empty Metal resize avoids buffers and products");
    return true;
}

void m3_test_metal_nearest_resize1d(m3_test_context *test)
{
    const uint64_t f16_input_shape[] = {1U, 1U, 1U};
    const uint64_t f16_output_shape[] = {1U, 1U, 5U};
    const uint16_t f16_input[] = {0x4200U};
    const uint16_t f16_output[5] = {0};
    const uint64_t bf16_input_shape[] = {1U, 1U, 5U};
    const uint64_t bf16_output_shape[] = {1U, 1U, 1U};
    const uint16_t bf16_input[] = {
        0x3f80U, 0x4000U, 0x4040U, 0x4080U, 0x40a0U
    };
    const uint16_t bf16_output[] = {0U};
    m3_metal_convolution_fixture fixture;

    if (!m3_metal_convolution_fixture_init(test, &fixture)) {
        return;
    }
    if (!m3_test_metal_resize_mixed(test, &fixture) ||
        !m3_test_metal_resize_strided_rows(test, &fixture) ||
        !m3_test_metal_resize_low_precision(
            test, &fixture, M3_DTYPE_F16, f16_input_shape,
            f16_output_shape, f16_input, f16_output,
            "width-one F16 Metal nearest resize matches host") ||
        !m3_test_metal_resize_low_precision(
            test, &fixture, M3_DTYPE_BF16, bf16_input_shape,
            bf16_output_shape, bf16_input, bf16_output,
            "width-one BF16 Metal nearest resize matches host") ||
        !m3_test_metal_resize_identity(test, &fixture) ||
        !m3_test_metal_resize_empty(test, &fixture)) {
        m3_metal_convolution_fixture_dispose(&fixture);
        return;
    }
    m3_metal_convolution_fixture_dispose(&fixture);
}

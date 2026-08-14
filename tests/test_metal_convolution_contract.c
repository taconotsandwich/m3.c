/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_backend_metal_convolution_internal.h"
#include "m3_backend_metal_helpers.h"
#include "metal_convolution_test.h"

#include <stdint.h>
#include <string.h>

static m3_command m3_test_metal_contract_conv(
    const m3_tensor_view *input, const m3_tensor_view *weight,
    m3_tensor_view *output, uint64_t groups)
{
    m3_command command = {0};

    command.kind = M3_OP_CONV1D;
    command.descriptor.conv1d.input = input;
    command.descriptor.conv1d.weight = weight;
    command.descriptor.conv1d.bias = NULL;
    command.descriptor.conv1d.output = output;
    command.descriptor.conv1d.groups = groups;
    command.descriptor.conv1d.stride = 1U;
    command.descriptor.conv1d.dilation = 1U;
    command.descriptor.conv1d.pad_left = 0U;
    command.descriptor.conv1d.pad_right = 0U;
    return command;
}

static m3_command m3_test_metal_contract_resize(
    const m3_tensor_view *input, m3_tensor_view *output)
{
    m3_command command = {0};

    command.kind = M3_OP_NEAREST_RESIZE1D;
    command.descriptor.nearest_resize1d.input = input;
    command.descriptor.nearest_resize1d.output = output;
    return command;
}

static m3_command m3_test_metal_contract_transpose(
    const m3_tensor_view *input, const m3_tensor_view *weight,
    m3_tensor_view *output)
{
    m3_command command = {0};

    command.kind = M3_OP_CONV_TRANSPOSE1D;
    command.descriptor.conv_transpose1d.input = input;
    command.descriptor.conv_transpose1d.weight = weight;
    command.descriptor.conv_transpose1d.bias = NULL;
    command.descriptor.conv_transpose1d.output = output;
    command.descriptor.conv_transpose1d.groups = 1U;
    command.descriptor.conv_transpose1d.stride = 1U;
    command.descriptor.conv_transpose1d.dilation = 1U;
    return command;
}

static bool m3_test_metal_convolution_dependency(
    m3_test_context *test, m3_metal_convolution_fixture *fixture)
{
    const uint64_t input_shape[] = {1U, 1U, 2U};
    const uint64_t weight_shape[] = {1U, 1U, 1U};
    const uint64_t output_shape[] = {1U, 1U, 5U};
    const float input_values[] = {1, 2};
    const float weight_values[] = {2};
    const float middle_zeros[] = {0, 0};
    const float output_zeros[] = {0, 0, 0, 0, 0};
    const float expected[] = {2, 2, 2, 4, 4};
    m3_metal_convolution_view input;
    m3_metal_convolution_view weight;
    m3_metal_convolution_view middle;
    m3_metal_convolution_view output = {0};
    m3_command host[2];
    m3_command metal[2];
    m3_command transpose;
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(test,
                   m3_metal_convolution_tensor(
                       fixture, &input, M3_DTYPE_F32, 3U, input_shape,
                       input_values) &&
                       m3_metal_convolution_tensor(
                           fixture, &weight, M3_DTYPE_F32, 3U,
                           weight_shape, weight_values) &&
                       m3_metal_convolution_tensor(
                           fixture, &middle, M3_DTYPE_F32, 3U,
                           input_shape, middle_zeros) &&
                       m3_metal_convolution_tensor(
                           fixture, &output, M3_DTYPE_F32, 3U,
                           output_shape, output_zeros),
                   "create dependent Metal convolution list");
    if (output.host.storage == NULL) {
        return false;
    }
    host[0] = m3_test_metal_contract_conv(
        &input.host, &weight.host, &middle.host, 1U);
    metal[0] = m3_test_metal_contract_conv(
        &input.metal, &weight.metal, &middle.metal, 1U);
    host[1] = m3_test_metal_contract_resize(
        &middle.host, &output.host);
    metal[1] = m3_test_metal_contract_resize(
        &middle.metal, &output.metal);
    transpose = m3_test_metal_contract_transpose(
        &input.metal, &weight.metal, &middle.metal);
    M3_TEST_EXPECT(
        test,
        m3_metal_command_writes_storage(
            &metal[0], middle.metal.storage) &&
            !m3_metal_command_writes_storage(
                &metal[0], output.metal.storage) &&
            m3_metal_command_writes_storage(
                &transpose, middle.metal.storage) &&
            m3_metal_command_writes_storage(
                &metal[1], output.metal.storage),
        "Metal convolution registry reports family outputs");
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_execute(
                       fixture, host, metal, 2U, &error) &&
                       m3_metal_convolution_equal(&middle) &&
                       m3_metal_convolution_equal(&output),
                   "dependent Metal convolution commands stay on GPU");
    for (index = 0U; index < 5U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_op_test_f32(&output.metal)[index] ==
                           expected[index],
                       "dependent Metal resize observes convolution output");
    }
    return true;
}

static bool m3_test_metal_convolution_atomic(
    m3_test_context *test, m3_metal_convolution_fixture *fixture)
{
    const uint64_t shape[] = {1U, 1U, 2U};
    const uint64_t weight_shape[] = {1U, 1U, 1U};
    const float source_values[] = {7, 8};
    const float weight_values[] = {2};
    const float middle_sentinels[] = {-41, -42};
    const float output_sentinels[] = {-51, -52};
    m3_metal_convolution_view source;
    m3_metal_convolution_view weight;
    m3_metal_convolution_view middle;
    m3_metal_convolution_view output = {0};
    m3_command commands[2];
    m3_error error;
    m3_status status;

    M3_TEST_EXPECT(test,
                   m3_metal_convolution_tensor(
                       fixture, &source, M3_DTYPE_F32, 3U, shape,
                       source_values) &&
                       m3_metal_convolution_tensor(
                           fixture, &weight, M3_DTYPE_F32, 3U,
                           weight_shape, weight_values) &&
                       m3_metal_convolution_tensor(
                           fixture, &middle, M3_DTYPE_F32, 3U, shape,
                           middle_sentinels) &&
                       m3_metal_convolution_tensor(
                           fixture, &output, M3_DTYPE_F32, 3U, shape,
                           output_sentinels),
                   "create invalid-list Metal convolution tensors");
    if (output.metal.storage == NULL) {
        return false;
    }
    commands[0] = m3_test_metal_contract_resize(
        &source.metal, &middle.metal);
    commands[1] = m3_test_metal_contract_conv(
        &middle.metal, &weight.metal, &output.metal, 0U);
    status = m3_backend_execute(
        fixture->metal.backend, commands, 2U, NULL, &error);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_INVALID_ARGUMENT &&
            memcmp(m3_op_test_f32(&middle.metal), middle_sentinels,
                   sizeof(middle_sentinels)) == 0 &&
            memcmp(m3_op_test_f32(&output.metal), output_sentinels,
                   sizeof(output_sentinels)) == 0,
        "whole-list convolution validation is atomic");
    status = m3_backend_execute(
        fixture->metal.backend, commands, 2U, NULL, NULL);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_INVALID_ARGUMENT &&
            memcmp(m3_op_test_f32(&middle.metal), middle_sentinels,
                   sizeof(middle_sentinels)) == 0 &&
            memcmp(m3_op_test_f32(&output.metal), output_sentinels,
                   sizeof(output_sentinels)) == 0,
        "invalid Metal convolution list accepts null error sink");
    return true;
}

static bool m3_test_metal_convolution_limits(m3_test_context *test)
{
    m3_error error;
    size_t work = 77U;

    M3_TEST_EXPECT(test,
                   m3_metal_convolution_work(
                       (uint64_t)UINT32_MAX + 1U, &work, &error) ==
                           M3_STATUS_OUT_OF_RANGE &&
                       work == 0U,
                   "Metal convolution rejects grid above uint32");
    work = 77U;
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_work(
                       UINT32_MAX, &work, &error) == M3_STATUS_OK &&
                       work == (size_t)UINT32_MAX,
                   "Metal convolution accepts maximum indexed grid");
    work = 77U;
    M3_TEST_EXPECT(test,
                   m3_metal_convolution_work(0U, &work, NULL) ==
                           M3_STATUS_OK &&
                       work == 0U,
                   "empty Metal convolution bypasses dispatch limits");
    return true;
}

void m3_test_metal_convolution_contract(m3_test_context *test)
{
    m3_metal_convolution_fixture fixture;
    m3_backend_allocation_stats stats;
    m3_error error;

    if (!m3_metal_convolution_fixture_init(test, &fixture)) {
        return;
    }
    if (!m3_test_metal_convolution_dependency(test, &fixture) ||
        !m3_test_metal_convolution_atomic(test, &fixture) ||
        !m3_test_metal_convolution_limits(test)) {
        m3_metal_convolution_fixture_dispose(&fixture);
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_backend_get_allocation_stats(
            fixture.metal.backend, &stats, &error) == M3_STATUS_OK &&
            stats.live_storage_count == fixture.metal.storage_count &&
            stats.live_storage_count != 0U &&
            stats.live_allocated_bytes != 0U,
        "Metal convolution storage stays live and accounted");
    m3_backend_free(fixture.metal.backend);
    fixture.metal.backend = NULL;
    fixture.metal.storage_count = 0U;
    M3_TEST_EXPECT(test, true,
                   "Metal convolution backend releases live storage");
    m3_metal_convolution_fixture_dispose(&fixture);
}

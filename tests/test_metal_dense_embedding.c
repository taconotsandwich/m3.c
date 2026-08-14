/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_backend_metal_dense_internal.h"
#include "metal_dense_test.h"

#include <stdint.h>
#include <string.h>

static m3_command m3_test_metal_embedding_command(
    const m3_tensor_view *ids, const m3_tensor_view *table,
    m3_tensor_view *output)
{
    m3_command command = {0};

    command.kind = M3_OP_EMBEDDING;
    command.descriptor.embedding.ids = ids;
    command.descriptor.embedding.table = table;
    command.descriptor.embedding.output = output;
    return command;
}

static m3_command m3_test_metal_copy_command(
    const m3_tensor_view *input, m3_tensor_view *output)
{
    m3_command command = {0};

    command.kind = M3_OP_COPY;
    command.descriptor.copy.input = input;
    command.descriptor.copy.output = output;
    return command;
}

static m3_command m3_test_metal_embedding_linear_command(
    const m3_tensor_view *input, const m3_tensor_view *weight,
    m3_tensor_view *output)
{
    m3_command command = {0};

    command.kind = M3_OP_LINEAR;
    command.descriptor.linear.input = input;
    command.descriptor.linear.weight = weight;
    command.descriptor.linear.bias = NULL;
    command.descriptor.linear.output = output;
    return command;
}

static bool m3_test_metal_embedding_strided(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t ids_shape[] = {2U};
    const uint64_t table_shape[] = {3U, 2U};
    const uint64_t output_shape[] = {2U, 2U};
    const size_t ids_strides[] = {2U * sizeof(int32_t)};
    const size_t table_strides[] = {3U * sizeof(float), sizeof(float)};
    const int32_t ids_backing[] = {2, 91, 0};
    const float table_backing[] = {1, 2, 92, 3, 4, 93, 5, 6};
    const float sentinels[] = {-9, -9, -9, -9};
    const float expected[] = {5, 6, 1, 2};
    m3_metal_dense_view ids;
    m3_metal_dense_view table;
    m3_metal_dense_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_strided(
                       fixture, &ids, M3_DTYPE_I32, 1U, ids_shape,
                       ids_strides, sizeof(ids_backing), ids_backing) &&
                       m3_metal_dense_strided(
                           fixture, &table, M3_DTYPE_F32, 2U, table_shape,
                           table_strides, sizeof(table_backing),
                           table_backing) &&
                       m3_metal_dense_tensor(
                           fixture, &output, M3_DTYPE_F32, 2U,
                           output_shape, sentinels),
                   "create strided F32 Metal embedding");
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_embedding_command(
        &ids.host, &table.host, &output.host);
    metal = m3_test_metal_embedding_command(
        &ids.metal, &table.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output),
                   "Metal embedding matches host for strided inputs");
    for (index = 0U; index < 4U; ++index) {
        M3_TEST_EXPECT(test,
                       m3_op_test_f32(&output.metal)[index] ==
                           expected[index],
                       "Metal embedding gathers rows in ID order");
    }
    return true;
}

static bool m3_test_metal_embedding_low_precision(
    m3_test_context *test, m3_metal_dense_fixture *fixture,
    m3_dtype dtype, const void *table_values, const char *message)
{
    const uint64_t ids_shape[] = {2U};
    const uint64_t table_shape[] = {3U, 2U};
    const uint64_t output_shape[] = {2U, 2U};
    const int32_t ids_values[] = {2, 0};
    const uint16_t zeros[] = {0U, 0U, 0U, 0U};
    m3_metal_dense_view ids;
    m3_metal_dense_view table;
    m3_metal_dense_view output = {0};
    m3_command host;
    m3_command metal;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &ids, M3_DTYPE_I32, 1U, ids_shape,
                       ids_values) &&
                       m3_metal_dense_tensor(
                           fixture, &table, dtype, 2U, table_shape,
                           table_values) &&
                       m3_metal_dense_tensor(
                           fixture, &output, dtype, 2U, output_shape,
                           zeros),
                   message);
    if (output.host.storage == NULL) {
        return false;
    }
    host = m3_test_metal_embedding_command(
        &ids.host, &table.host, &output.host);
    metal = m3_test_metal_embedding_command(
        &ids.metal, &table.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, &host, &metal, 1U, &error) &&
                       m3_metal_dense_equal(&output),
                   message);
    return true;
}

static bool m3_test_metal_embedding_invalid(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t ids_shape[] = {2U};
    const uint64_t table_shape[] = {3U, 2U};
    const uint64_t output_shape[] = {2U, 2U};
    const int32_t ids_values[] = {-1, 3};
    const float table_values[] = {1, 2, 3, 4, 5, 6};
    const float sentinels[] = {-9, -8, -7, -6};
    m3_metal_dense_view ids;
    m3_metal_dense_view table;
    m3_metal_dense_view output = {0};
    m3_command command;
    m3_error error;
    m3_status status;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &ids, M3_DTYPE_I32, 1U, ids_shape,
                       ids_values) &&
                       m3_metal_dense_tensor(
                           fixture, &table, M3_DTYPE_F32, 2U, table_shape,
                           table_values) &&
                       m3_metal_dense_tensor(
                           fixture, &output, M3_DTYPE_F32, 2U,
                           output_shape, sentinels),
                   "create invalid-ID Metal embedding");
    if (output.metal.storage == NULL) {
        return false;
    }
    command = m3_test_metal_embedding_command(
        &ids.metal, &table.metal, &output.metal);
    status = m3_backend_execute(
        fixture->metal.backend, &command, 1U, NULL, &error);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_OUT_OF_RANGE &&
                       memcmp(m3_op_test_f32(&output.metal), sentinels,
                              sizeof(sentinels)) == 0,
                   "invalid Metal embedding IDs leave output unchanged");
    status = m3_backend_execute(
        fixture->metal.backend, &command, 1U, NULL, NULL);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_OUT_OF_RANGE &&
                       memcmp(m3_op_test_f32(&output.metal), sentinels,
                              sizeof(sentinels)) == 0,
                   "invalid Metal embedding accepts a null error sink");
    return true;
}

static bool m3_test_metal_embedding_empty_output(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t ids_shape[] = {2U};
    const uint64_t table_shape[] = {3U, 0U};
    const uint64_t output_shape[] = {2U, 0U};
    const int32_t ids_values[] = {-1, 3};
    m3_metal_dense_view ids;
    m3_metal_dense_view table;
    m3_metal_dense_view output = {0};
    m3_command command;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &ids, M3_DTYPE_I32, 1U, ids_shape,
                       ids_values) &&
                       m3_metal_dense_tensor(
                           fixture, &table, M3_DTYPE_F32, 2U, table_shape,
                           NULL) &&
                       m3_metal_dense_tensor(
                           fixture, &output, M3_DTYPE_F32, 2U,
                           output_shape, NULL),
                   "create empty-output Metal embedding");
    if (ids.metal.storage == NULL) {
        return false;
    }
    command = m3_test_metal_embedding_command(
        &ids.metal, &table.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_backend_execute(
                       fixture->metal.backend, &command, 1U, NULL,
                       &error) == M3_STATUS_OUT_OF_RANGE,
                   "Metal checks IDs even when embedding output is empty");
    M3_TEST_EXPECT(test,
                   m3_backend_execute(
                       fixture->metal.backend, &command, 1U, NULL,
                       NULL) == M3_STATUS_OUT_OF_RANGE,
                   "empty invalid embedding accepts a null error sink");
    return true;
}

static bool m3_test_metal_embedding_prior_writer(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t ids_shape[] = {2U};
    const uint64_t table_shape[] = {3U, 2U};
    const uint64_t output_shape[] = {2U, 2U};
    const int32_t source_values[] = {2, 0};
    const int32_t stale_ids[] = {-1, 3};
    const float table_values[] = {1, 2, 3, 4, 5, 6};
    const float sentinels[] = {-9, -8, -7, -6};
    m3_metal_dense_view source;
    m3_metal_dense_view ids;
    m3_metal_dense_view table;
    m3_metal_dense_view output = {0};
    m3_command commands[2];
    m3_error error;
    m3_status status;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &source, M3_DTYPE_I32, 1U, ids_shape,
                       source_values) &&
                       m3_metal_dense_tensor(
                           fixture, &ids, M3_DTYPE_I32, 1U, ids_shape,
                           stale_ids) &&
                       m3_metal_dense_tensor(
                           fixture, &table, M3_DTYPE_F32, 2U, table_shape,
                           table_values) &&
                       m3_metal_dense_tensor(
                           fixture, &output, M3_DTYPE_F32, 2U,
                           output_shape, sentinels),
                   "create produced-ID Metal embedding");
    if (output.metal.storage == NULL) {
        return false;
    }
    commands[0] = m3_test_metal_copy_command(
        &source.metal, &ids.metal);
    commands[1] = m3_test_metal_embedding_command(
        &ids.metal, &table.metal, &output.metal);
    status = m3_backend_execute(
        fixture->metal.backend, commands, 2U, NULL, &error);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_UNSUPPORTED &&
                       memcmp(m3_op_test_i32(&ids.metal), stale_ids,
                              sizeof(stale_ids)) == 0 &&
                       memcmp(m3_op_test_f32(&output.metal), sentinels,
                              sizeof(sentinels)) == 0,
                   "Metal rejects produced embedding IDs before stale data");
    status = m3_backend_execute(
        fixture->metal.backend, commands, 2U, NULL, NULL);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_UNSUPPORTED &&
                       memcmp(m3_op_test_i32(&ids.metal), stale_ids,
                              sizeof(stale_ids)) == 0 &&
                       memcmp(m3_op_test_f32(&output.metal), sentinels,
                              sizeof(sentinels)) == 0,
                   "produced-ID rejection accepts a null error sink");
    return true;
}

static bool m3_test_metal_embedding_dependency(
    m3_test_context *test, m3_metal_dense_fixture *fixture)
{
    const uint64_t ids_shape[] = {2U};
    const uint64_t table_shape[] = {3U, 2U};
    const uint64_t embedded_shape[] = {2U, 2U};
    const uint64_t weight_shape[] = {1U, 2U};
    const uint64_t output_shape[] = {2U, 1U};
    const int32_t ids_values[] = {2, 0};
    const float table_values[] = {1, 2, 3, 4, 5, 6};
    const float weight_values[] = {1, -1};
    const float embedded_zeros[] = {0, 0, 0, 0};
    const float output_zeros[] = {0, 0};
    m3_metal_dense_view ids;
    m3_metal_dense_view table;
    m3_metal_dense_view embedded;
    m3_metal_dense_view weight;
    m3_metal_dense_view output = {0};
    m3_command host[2];
    m3_command metal[2];
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_tensor(
                       fixture, &ids, M3_DTYPE_I32, 1U, ids_shape,
                       ids_values) &&
                       m3_metal_dense_tensor(
                           fixture, &table, M3_DTYPE_F32, 2U, table_shape,
                           table_values) &&
                       m3_metal_dense_tensor(
                           fixture, &embedded, M3_DTYPE_F32, 2U,
                           embedded_shape, embedded_zeros) &&
                       m3_metal_dense_tensor(
                           fixture, &weight, M3_DTYPE_F32, 2U,
                           weight_shape, weight_values) &&
                       m3_metal_dense_tensor(
                           fixture, &output, M3_DTYPE_F32, 2U,
                           output_shape, output_zeros),
                   "create dependent embedding-linear Metal list");
    if (output.host.storage == NULL) {
        return false;
    }
    host[0] = m3_test_metal_embedding_command(
        &ids.host, &table.host, &embedded.host);
    metal[0] = m3_test_metal_embedding_command(
        &ids.metal, &table.metal, &embedded.metal);
    host[1] = m3_test_metal_embedding_linear_command(
        &embedded.host, &weight.host, &output.host);
    metal[1] = m3_test_metal_embedding_linear_command(
        &embedded.metal, &weight.metal, &output.metal);
    M3_TEST_EXPECT(test,
                   m3_metal_dense_execute(
                       fixture, host, metal, 2U, &error) &&
                       m3_metal_dense_equal(&embedded) &&
                       m3_metal_dense_equal(&output) &&
                       m3_op_test_f32(&output.metal)[0] == -1.0F &&
                       m3_op_test_f32(&output.metal)[1] == -1.0F,
                   "dependent Metal dense commands execute in list order");
    return true;
}

static bool m3_test_metal_dense_limits(m3_test_context *test)
{
    m3_error error;
    size_t work = 77U;

    M3_TEST_EXPECT(test,
                   m3_metal_dense_work(
                       (uint64_t)UINT32_MAX + 1U, &work, &error) ==
                           M3_STATUS_OUT_OF_RANGE &&
                       work == 0U,
                   "Metal dense grid rejects metadata beyond uint32");
    work = 77U;
    M3_TEST_EXPECT(test,
                   m3_metal_dense_product(
                       UINT64_MAX, 2U, &work, &error) ==
                           M3_STATUS_OVERFLOW &&
                       work == 0U,
                   "Metal dense parameter products reject uint64 overflow");
    work = 77U;
    M3_TEST_EXPECT(test,
                   m3_metal_dense_product(
                       0U, UINT64_MAX, &work, NULL) == M3_STATUS_OK &&
                       work == 0U,
                   "empty Metal dense dimensions bypass unused bounds");
    return true;
}

void m3_test_metal_dense_embedding(m3_test_context *test)
{
    const uint16_t f16_table[] = {
        0x3c00U, 0x4000U, 0x4200U, 0x4400U, 0x4500U, 0x4600U
    };
    const uint16_t bf16_table[] = {
        0x3f80U, 0x4000U, 0x4040U, 0x4080U, 0x40a0U, 0x40c0U
    };
    m3_metal_dense_fixture fixture;

    if (!m3_metal_dense_fixture_init(test, &fixture)) {
        return;
    }
    if (!m3_test_metal_embedding_strided(test, &fixture) ||
        !m3_test_metal_embedding_low_precision(
            test, &fixture, M3_DTYPE_F16, f16_table,
            "F16 Metal embedding matches host exactly") ||
        !m3_test_metal_embedding_low_precision(
            test, &fixture, M3_DTYPE_BF16, bf16_table,
            "BF16 Metal embedding matches host exactly") ||
        !m3_test_metal_embedding_invalid(test, &fixture) ||
        !m3_test_metal_embedding_empty_output(test, &fixture) ||
        !m3_test_metal_embedding_prior_writer(test, &fixture) ||
        !m3_test_metal_embedding_dependency(test, &fixture) ||
        !m3_test_metal_dense_limits(test)) {
        m3_metal_dense_fixture_dispose(&fixture);
        return;
    }
    m3_metal_dense_fixture_dispose(&fixture);
}

/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"

#include <math.h>
#include <string.h>

static bool m3_test_attention_run(m3_op_test_fixture *fixture,
                                  m3_tensor_view *query,
                                  m3_tensor_view *key,
                                  m3_tensor_view *value,
                                  m3_tensor_view *mask,
                                  m3_tensor_view *output, bool causal,
                                  int64_t causal_offset,
                                  size_t *scratch_bytes,
                                  m3_error *error)
{
    m3_command command;

    command.kind = M3_OP_ATTENTION;
    command.descriptor.attention.query = query;
    command.descriptor.attention.key = key;
    command.descriptor.attention.value = value;
    command.descriptor.attention.mask = mask;
    command.descriptor.attention.output = output;
    command.descriptor.attention.scale = 1.0F;
    command.descriptor.attention.causal_offset = causal_offset;
    command.descriptor.attention.causal = causal;
    return m3_op_test_execute(fixture, &command, 1U, scratch_bytes, error) ==
           M3_STATUS_OK;
}

void m3_test_op_attention(m3_test_context *test)
{
    const uint64_t query_shape[] = {1U, 1U, 2U, 1U};
    const uint64_t kv_shape[] = {1U, 1U, 2U, 1U};
    const float causal_query_values[] = {1, 1};
    const float causal_key_values[] = {0, 0};
    const float causal_value_values[] = {10, 20};
    const float zeros2[2] = {0};
    const uint64_t gqa_query_shape[] = {1U, 2U, 1U, 1U};
    const float gqa_query_values[] = {1, -1};
    const float gqa_key_values[] = {0, 1};
    const float gqa_value_values[] = {2, 6};
    const uint64_t mask_shape[] = {1U, 1U, 2U, 2U};
    const float mask_values[] = {0, -INFINITY, -INFINITY, -INFINITY};
    const float masked_query_values[] = {0, 0};
    const float masked_key_values[] = {0, 0};
    const float masked_value_values[] = {2, 6};
    const float nan_query_values[] = {NAN, 0};
    const float sentinel[] = {-9, -9};
    m3_op_test_fixture fixture;
    m3_tensor_view query;
    m3_tensor_view key;
    m3_tensor_view value;
    m3_tensor_view output;
    m3_tensor_view gqa_query;
    m3_tensor_view gqa_key;
    m3_tensor_view gqa_value;
    m3_tensor_view gqa_output;
    m3_tensor_view masked_query;
    m3_tensor_view masked_key;
    m3_tensor_view masked_value;
    m3_tensor_view mask;
    m3_tensor_view masked_output;
    m3_error error;
    size_t scratch_bytes = 0U;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create attention fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &query, M3_DTYPE_F32, 4U,
                                     query_shape, causal_query_values) &&
                       m3_op_test_tensor(&fixture, &key, M3_DTYPE_F32, 4U,
                                         kv_shape, causal_key_values) &&
                       m3_op_test_tensor(&fixture, &value, M3_DTYPE_F32, 4U,
                                         kv_shape, causal_value_values) &&
                       m3_op_test_tensor(&fixture, &output, M3_DTYPE_F32, 4U,
                                         query_shape, zeros2),
                   "create causal attention tensors");
    M3_TEST_EXPECT(test,
                   m3_test_attention_run(&fixture, &query, &key, &value,
                                         NULL, &output, true, 0,
                                         &scratch_bytes, &error),
                   "execute causal attention");
    M3_TEST_EXPECT(test, scratch_bytes >= 2U * sizeof(float),
                   "attention reports one key-row scratch buffer");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&output)[0], 10.0F, 0.0F, 0.0F,
                       "causal first query sees only first key");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&output)[1], 15.0F, 1.0e-6F,
                       1.0e-6F, "causal second query sees both keys");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &gqa_query, M3_DTYPE_F32, 4U,
                                     gqa_query_shape, gqa_query_values) &&
                       m3_op_test_tensor(&fixture, &gqa_key, M3_DTYPE_F32, 4U,
                                         kv_shape, gqa_key_values) &&
                       m3_op_test_tensor(&fixture, &gqa_value, M3_DTYPE_F32,
                                         4U, kv_shape, gqa_value_values) &&
                       m3_op_test_tensor(&fixture, &gqa_output, M3_DTYPE_F32,
                                         4U, gqa_query_shape, zeros2),
                   "create grouped-query attention tensors");
    M3_TEST_EXPECT(test,
                   m3_test_attention_run(&fixture, &gqa_query, &gqa_key,
                                         &gqa_value, NULL, &gqa_output, false,
                                         0, NULL, &error),
                   "execute noncausal grouped-query attention");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&gqa_output)[0], 4.92423439F,
                       1.0e-6F, 1.0e-6F,
                       "first query head shares the KV head");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&gqa_output)[1], 3.07576561F,
                       1.0e-6F, 1.0e-6F,
                       "second query head shares the KV head");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &masked_query, M3_DTYPE_F32,
                                     4U, query_shape,
                                     masked_query_values) &&
                       m3_op_test_tensor(&fixture, &masked_key, M3_DTYPE_F32,
                                         4U, kv_shape,
                                         masked_key_values) &&
                       m3_op_test_tensor(&fixture, &masked_value,
                                         M3_DTYPE_F32, 4U, kv_shape,
                                         masked_value_values) &&
                       m3_op_test_tensor(&fixture, &mask, M3_DTYPE_F32, 4U,
                                         mask_shape, mask_values) &&
                       m3_op_test_tensor(&fixture, &masked_output,
                                         M3_DTYPE_F32, 4U, query_shape,
                                         sentinel),
                   "create additive-mask attention tensors");
    M3_TEST_EXPECT(test,
                   m3_test_attention_run(
                       &fixture, &masked_query, &masked_key, &masked_value,
                       &mask, &masked_output, false, 0, NULL, &error),
                   "execute noncausal additive-mask attention");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&masked_output)[0], 2.0F, 0.0F,
                       0.0F, "additive mask excludes one key");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&masked_output)[1], 0.0F, 0.0F,
                       0.0F, "all-masked attention row produces zeros");
    (void)m3_storage_write(masked_query.storage, 0U, nan_query_values,
                           sizeof(nan_query_values), &error);
    (void)m3_storage_write(masked_output.storage, 0U, sentinel,
                           sizeof(sentinel), &error);
    {
        m3_command command;

        command.kind = M3_OP_ATTENTION;
        command.descriptor.attention.query = &masked_query;
        command.descriptor.attention.key = &masked_key;
        command.descriptor.attention.value = &masked_value;
        command.descriptor.attention.mask = &mask;
        command.descriptor.attention.output = &masked_output;
        command.descriptor.attention.scale = 1.0F;
        command.descriptor.attention.causal_offset = 0;
        command.descriptor.attention.causal = false;
        M3_TEST_EXPECT(
            test,
            m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                    M3_STATUS_OUT_OF_RANGE &&
                memcmp(m3_op_test_f32(&masked_output), sentinel,
                       sizeof(sentinel)) == 0,
            "attention data failure leaves the full output unchanged");
    }
    m3_op_test_fixture_dispose(&fixture);
}

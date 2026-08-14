/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"
#include "m3_runtime_workspace.h"

#include <stdint.h>

static void m3_runtime_attention_command(
    m3_command *command, const m3_tensor_view *query,
    const m3_tensor_view *key, const m3_tensor_view *value,
    m3_tensor_view *output)
{
    command->kind = M3_OP_ATTENTION;
    command->descriptor.attention.query = query;
    command->descriptor.attention.key = key;
    command->descriptor.attention.value = value;
    command->descriptor.attention.mask = NULL;
    command->descriptor.attention.output = output;
    command->descriptor.attention.scale = 1.0F;
    command->descriptor.attention.causal_offset = 0;
    command->descriptor.attention.causal = false;
}

void m3_test_runtime_command_executor(m3_test_context *test)
{
    const uint64_t query_shape[] = {1U, 1U, 1U, 1U};
    const uint64_t small_kv_shape[] = {1U, 1U, 2U, 1U};
    const uint64_t large_kv_shape[] = {1U, 1U, 4U, 1U};
    const float query_values[] = {0.0F};
    const float small_key_values[] = {0.0F, 0.0F};
    const float small_value_values[] = {2.0F, 6.0F};
    const float large_key_values[] = {0.0F, 0.0F, 0.0F, 0.0F};
    const float large_value_values[] = {1.0F, 3.0F, 5.0F, 7.0F};
    const float output_values[] = {-1.0F};
    m3_op_test_fixture fixture;
    m3_command_executor executor;
    m3_tensor_view query;
    m3_tensor_view small_key;
    m3_tensor_view small_value;
    m3_tensor_view small_output;
    m3_tensor_view large_key;
    m3_tensor_view large_value;
    m3_tensor_view large_output;
    m3_command small;
    m3_command large;
    m3_command invalid = {0};
    m3_error error;
    void *small_scratch;
    size_t small_capacity;
    bool ready;

    m3_command_executor_init(NULL, NULL);
    m3_command_executor_dispose(NULL);
    ready = m3_op_test_fixture_init(&fixture);
    M3_TEST_EXPECT(test, ready, "create reusable executor host fixture");
    if (!ready) {
        return;
    }
    m3_command_executor_init(&executor, fixture.backend);
    M3_TEST_EXPECT(
        test,
        m3_command_executor_execute(&executor, NULL, 0U, &error) ==
                M3_STATUS_OK &&
            executor.scratch == NULL && executor.scratch_capacity == 0U,
        "executor accepts an empty command list without allocation");
    ready = m3_op_test_tensor(&fixture, &query, M3_DTYPE_F32, 4U,
                              query_shape, query_values) &&
            m3_op_test_tensor(&fixture, &small_key, M3_DTYPE_F32, 4U,
                              small_kv_shape, small_key_values) &&
            m3_op_test_tensor(&fixture, &small_value, M3_DTYPE_F32, 4U,
                              small_kv_shape, small_value_values) &&
            m3_op_test_tensor(&fixture, &small_output, M3_DTYPE_F32, 4U,
                              query_shape, output_values) &&
            m3_op_test_tensor(&fixture, &large_key, M3_DTYPE_F32, 4U,
                              large_kv_shape, large_key_values) &&
            m3_op_test_tensor(&fixture, &large_value, M3_DTYPE_F32, 4U,
                              large_kv_shape, large_value_values) &&
            m3_op_test_tensor(&fixture, &large_output, M3_DTYPE_F32, 4U,
                              query_shape, output_values);
    M3_TEST_EXPECT(test, ready, "create scratch-backed executor tensors");
    if (!ready) {
        m3_command_executor_dispose(&executor);
        m3_op_test_fixture_dispose(&fixture);
        return;
    }
    m3_runtime_attention_command(&small, &query, &small_key, &small_value,
                                 &small_output);
    M3_TEST_EXPECT(
        test,
        m3_command_executor_execute(&executor, &small, 1U, &error) ==
                M3_STATUS_OK &&
            executor.scratch != NULL && executor.scratch_capacity != 0U &&
            (uintptr_t)executor.scratch % 64U == 0U,
        "executor allocates aligned scratch for a checked command");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&small_output)[0], 4.0F, 0.0F,
                       0.0F, "scratch-backed attention executes exactly");
    small_scratch = executor.scratch;
    small_capacity = executor.scratch_capacity;
    M3_TEST_EXPECT(
        test,
        m3_command_executor_execute(&executor, &small, 1U, &error) ==
                M3_STATUS_OK &&
            executor.scratch == small_scratch &&
            executor.scratch_capacity == small_capacity,
        "executor reuses sufficient scratch without reallocating");
    invalid.kind = (m3_op_kind)999;
    M3_TEST_EXPECT(
        test,
        m3_command_executor_execute(&executor, &invalid, 1U, &error) ==
                M3_STATUS_UNSUPPORTED &&
            executor.scratch == small_scratch &&
            executor.scratch_capacity == small_capacity,
        "command validation failure preserves reusable scratch");
    M3_TEST_EXPECT(
        test,
        m3_command_executor_execute(&executor, &invalid, 1U, NULL) ==
            M3_STATUS_UNSUPPORTED,
        "executor validation works without an error sink");
    m3_runtime_attention_command(&large, &query, &large_key, &large_value,
                                 &large_output);
    M3_TEST_EXPECT(
        test,
        m3_command_executor_execute(&executor, &large, 1U, &error) ==
                M3_STATUS_OK &&
            executor.scratch != small_scratch &&
            executor.scratch_capacity > small_capacity &&
            (uintptr_t)executor.scratch % 64U == 0U,
        "executor transactionally grows scratch for a larger command");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&large_output)[0], 4.0F, 0.0F,
                       0.0F, "grown scratch executes the larger attention");
    M3_TEST_EXPECT(
        test,
        m3_command_executor_execute(&executor, NULL, 0U, &error) ==
                M3_STATUS_OK &&
            executor.scratch_capacity > small_capacity,
        "empty execution retains reusable scratch");
    m3_command_executor_dispose(&executor);
    M3_TEST_EXPECT(test,
                   executor.backend == NULL && executor.scratch == NULL &&
                       executor.scratch_capacity == 0U,
                   "executor disposal releases and clears owned scratch");
    m3_command_executor_init(&executor, NULL);
    M3_TEST_EXPECT(
        test,
        m3_command_executor_execute(&executor, NULL, 0U, NULL) ==
            M3_STATUS_INVALID_ARGUMENT,
        "executor rejects a missing borrowed backend without diagnostics");
    m3_command_executor_dispose(&executor);
    m3_op_test_fixture_dispose(&fixture);
}

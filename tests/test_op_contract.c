/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"

#include <stdint.h>
#include <string.h>

void m3_test_op_validation_atomicity(m3_test_context *test)
{
    const uint64_t two[] = {2U};
    const uint64_t one[] = {1U};
    const float left_values[] = {1, 2};
    const float right_values[] = {3, 4};
    const float sentinel[] = {-8, -8};
    const float narrow_sentinel[] = {-3};
    m3_op_test_fixture fixture;
    m3_tensor_view left;
    m3_tensor_view right;
    m3_tensor_view output;
    m3_tensor_view invalid_output;
    m3_command commands[2];
    m3_error error;
    size_t scratch_bytes = 99U;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create command-list fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &left, M3_DTYPE_F32, 1U, two,
                                     left_values) &&
                       m3_op_test_tensor(&fixture, &right, M3_DTYPE_F32, 1U,
                                         two, right_values) &&
                       m3_op_test_tensor(&fixture, &output, M3_DTYPE_F32, 1U,
                                         two, sentinel) &&
                       m3_op_test_tensor(&fixture, &invalid_output,
                                         M3_DTYPE_F32, 1U, one,
                                         narrow_sentinel),
                   "create command-list tensors");
    commands[0].kind = M3_OP_ADD;
    commands[0].descriptor.add.left = &left;
    commands[0].descriptor.add.right = &right;
    commands[0].descriptor.add.output = &output;
    commands[1].kind = M3_OP_SOFTMAX;
    commands[1].descriptor.softmax.input = &left;
    commands[1].descriptor.softmax.output = &invalid_output;
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture.backend, commands, 2U, NULL,
                                      &error) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       memcmp(m3_op_test_f32(&output), sentinel,
                              sizeof(sentinel)) == 0,
                   "late structural error prevents every earlier write");
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture.backend, NULL, 0U, NULL,
                                      &error) == M3_STATUS_OK,
                   "empty command list is a successful backend no-op");
    M3_TEST_EXPECT(test,
                   m3_commands_scratch_bytes(NULL, NULL, 0U,
                                             &scratch_bytes, &error) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       scratch_bytes == 99U,
                   "empty scratch query still requires a backend");
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_op_scratch_rejections(m3_test_context *test)
{
    const uint64_t logits_shape[] = {3U};
    const uint64_t top_shape[] = {2U};
    const float logits_values[] = {3, 2, 1};
    const float value_sentinel[] = {-7, -7};
    const int32_t index_sentinel[] = {-7, -7};
    const uint64_t query_shape[] = {0U, 1U, 0U, 1U};
    const uint64_t key_shape[] = {0U, 1U, (uint64_t)SIZE_MAX, 1U};
    uint8_t scratch_memory[64];
    m3_op_test_fixture fixture;
    m3_tensor_view logits;
    m3_tensor_view values;
    m3_tensor_view indices;
    m3_tensor_view query;
    m3_tensor_view key;
    m3_tensor_view value;
    m3_tensor_view attention_output;
    m3_storage *empty_storage = NULL;
    m3_command command;
    m3_scratch_arena arena;
    m3_error error;
    size_t required = 0U;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create scratch fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &logits, M3_DTYPE_F32, 1U,
                                     logits_shape, logits_values) &&
                       m3_op_test_tensor(&fixture, &values, M3_DTYPE_F32, 1U,
                                         top_shape, value_sentinel) &&
                       m3_op_test_tensor(&fixture, &indices, M3_DTYPE_I32, 1U,
                                         top_shape, index_sentinel),
                   "create top-k scratch tensors");
    command.kind = M3_OP_TOP_K;
    command.descriptor.top_k.logits = &logits;
    command.descriptor.top_k.values = &values;
    command.descriptor.top_k.indices = &indices;
    command.descriptor.top_k.k = 2U;
    M3_TEST_EXPECT(test,
                   m3_commands_scratch_bytes(fixture.backend, &command, 1U,
                                             &required, &error) ==
                           M3_STATUS_OK &&
                       required > 1U && required <= sizeof(scratch_memory),
                   "query checked top-k scratch bytes");
    M3_TEST_EXPECT(test,
                   m3_scratch_arena_init(&arena, scratch_memory,
                                         required - 1U, &error) ==
                       M3_STATUS_OK,
                   "create deliberately short scratch arena");
    M3_TEST_EXPECT(test,
                   m3_backend_execute(fixture.backend, &command, 1U, &arena,
                                      &error) == M3_STATUS_OUT_OF_MEMORY &&
                       memcmp(m3_op_test_f32(&values), value_sentinel,
                              sizeof(value_sentinel)) == 0 &&
                       memcmp(m3_op_test_i32(&indices), index_sentinel,
                              sizeof(index_sentinel)) == 0,
                   "scratch insufficiency is detected before output writes");
    m3_tensor_view_init(&query);
    m3_tensor_view_init(&key);
    m3_tensor_view_init(&value);
    m3_tensor_view_init(&attention_output);
    M3_TEST_EXPECT(test,
                   m3_op_test_storage(&fixture, 0U, &empty_storage) &&
                       m3_tensor_view_contiguous(
                           &query, empty_storage, M3_DTYPE_F32, 4U,
                           query_shape, 0U, &error) == M3_STATUS_OK &&
                       m3_tensor_view_contiguous(
                           &key, empty_storage, M3_DTYPE_F32, 4U, key_shape,
                           0U, &error) == M3_STATUS_OK &&
                       m3_tensor_view_contiguous(
                           &value, empty_storage, M3_DTYPE_F32, 4U,
                           key_shape, 0U, &error) == M3_STATUS_OK &&
                       m3_tensor_view_contiguous(
                           &attention_output, empty_storage, M3_DTYPE_F32, 4U,
                           query_shape, 0U, &error) == M3_STATUS_OK,
                   "create metadata-only overflowing attention views");
    command.kind = M3_OP_ATTENTION;
    command.descriptor.attention.query = &query;
    command.descriptor.attention.key = &key;
    command.descriptor.attention.value = &value;
    command.descriptor.attention.mask = NULL;
    command.descriptor.attention.output = &attention_output;
    command.descriptor.attention.scale = 1.0F;
    command.descriptor.attention.causal_offset = 0;
    command.descriptor.attention.causal = false;
    M3_TEST_EXPECT(test,
                   m3_commands_scratch_bytes(fixture.backend, &command, 1U,
                                             &required, &error) ==
                       M3_STATUS_OVERFLOW,
                   "attention scratch multiplication overflow is rejected");
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_op_sampling_shape_rejections(m3_test_context *test)
{
    const uint64_t empty[] = {0U};
    const uint64_t huge_probability[] = {
        0U, (uint64_t)INT32_MAX + 1U
    };
    const float scalar_value = 1.0F;
    const int32_t scalar_index = 0;
    m3_op_test_fixture fixture;
    m3_tensor_view scalar_logits;
    m3_tensor_view scalar_values;
    m3_tensor_view scalar_indices;
    m3_tensor_view probabilities;
    m3_tensor_view uniforms;
    m3_tensor_view categorical_output;
    m3_storage *empty_storage = NULL;
    m3_command command;
    m3_error error;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create sampling rejection fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &scalar_logits, M3_DTYPE_F32,
                                     0U, NULL, &scalar_value) &&
                       m3_op_test_tensor(&fixture, &scalar_values,
                                         M3_DTYPE_F32, 0U, NULL,
                                         &scalar_value) &&
                       m3_op_test_tensor(&fixture, &scalar_indices,
                                         M3_DTYPE_I32, 0U, NULL,
                                         &scalar_index),
                   "create scalar top-k views");
    command.kind = M3_OP_TOP_K;
    command.descriptor.top_k.logits = &scalar_logits;
    command.descriptor.top_k.values = &scalar_values;
    command.descriptor.top_k.indices = &scalar_indices;
    command.descriptor.top_k.k = 1U;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "scalar top-k is rejected without indexing shape[-1]");
    m3_tensor_view_init(&probabilities);
    m3_tensor_view_init(&uniforms);
    m3_tensor_view_init(&categorical_output);
    M3_TEST_EXPECT(test,
                   m3_op_test_storage(&fixture, 0U, &empty_storage) &&
                       m3_tensor_view_contiguous(
                           &probabilities, empty_storage, M3_DTYPE_F32, 2U,
                           huge_probability, 0U, &error) == M3_STATUS_OK &&
                       m3_tensor_view_contiguous(
                           &uniforms, empty_storage, M3_DTYPE_F32, 1U, empty,
                           0U, &error) == M3_STATUS_OK &&
                       m3_tensor_view_contiguous(
                           &categorical_output, empty_storage, M3_DTYPE_I32,
                           1U, empty, 0U, &error) == M3_STATUS_OK,
                   "create metadata-only huge categorical views");
    command.kind = M3_OP_CATEGORICAL;
    command.descriptor.categorical.probabilities = &probabilities;
    command.descriptor.categorical.uniforms = &uniforms;
    command.descriptor.categorical.output = &categorical_output;
    M3_TEST_EXPECT(test,
                   m3_command_validate(fixture.backend, &command, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "categorical vocabulary must fit I32 output indices");
    m3_op_test_fixture_dispose(&fixture);
}

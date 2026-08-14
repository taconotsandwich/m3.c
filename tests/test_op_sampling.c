/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_op_test.h"

#include <math.h>
#include <string.h>

void m3_test_op_softmax(m3_test_context *test)
{
    const uint64_t two[] = {2U};
    const uint64_t three[] = {3U};
    const float shifted_values[] = {1000, 1001};
    const float negative_infinities[] = {-INFINITY, -INFINITY};
    const float positive_infinities[] = {INFINITY, 1, INFINITY};
    const float zeros2[2] = {0};
    const float zeros3[3] = {0};
    const float nan_values[] = {0, NAN};
    const float sentinel[] = {-7, -7};
    m3_op_test_fixture fixture;
    m3_tensor_view shifted;
    m3_tensor_view all_negative;
    m3_tensor_view all_negative_output;
    m3_tensor_view positive;
    m3_tensor_view positive_output;
    m3_tensor_view nan_input;
    m3_tensor_view nan_output;
    m3_command command;
    m3_error error;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create softmax fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &shifted, M3_DTYPE_F32, 1U,
                                     two, shifted_values),
                   "create shifted in-place softmax tensor");
    command.kind = M3_OP_SOFTMAX;
    command.descriptor.softmax.input = &shifted;
    command.descriptor.softmax.output = &shifted;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                       M3_STATUS_OK,
                   "execute exact-alias stable softmax");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&shifted)[0], 0.268941432F,
                       1.0e-6F, 1.0e-6F,
                       "softmax subtracts a large row maximum");
    M3_TEST_EXPECT_F32(test, m3_op_test_f32(&shifted)[1], 0.731058598F,
                       1.0e-6F, 1.0e-6F,
                       "softmax normalized shifted maximum");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &all_negative, M3_DTYPE_F32,
                                     1U, two, negative_infinities) &&
                       m3_op_test_tensor(&fixture, &all_negative_output,
                                         M3_DTYPE_F32, 1U, two, zeros2),
                   "create all-negative-infinity softmax row");
    command.descriptor.softmax.input = &all_negative;
    command.descriptor.softmax.output = &all_negative_output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OK &&
                       m3_op_test_f32(&all_negative_output)[0] == 0.0F &&
                       m3_op_test_f32(&all_negative_output)[1] == 0.0F,
                   "all-negative-infinity softmax row becomes all zeros");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &positive, M3_DTYPE_F32, 1U,
                                     three, positive_infinities) &&
                       m3_op_test_tensor(&fixture, &positive_output,
                                         M3_DTYPE_F32, 1U, three, zeros3),
                   "create positive-infinity softmax row");
    command.descriptor.softmax.input = &positive;
    command.descriptor.softmax.output = &positive_output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OK &&
                       m3_op_test_f32(&positive_output)[0] == 0.5F &&
                       m3_op_test_f32(&positive_output)[1] == 0.0F &&
                       m3_op_test_f32(&positive_output)[2] == 0.5F,
                   "only positive-infinity entries split probability mass");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &nan_input, M3_DTYPE_F32, 1U,
                                     two, nan_values) &&
                       m3_op_test_tensor(&fixture, &nan_output,
                                         M3_DTYPE_F32, 1U, two, sentinel),
                   "create NaN softmax rejection tensors");
    command.descriptor.softmax.input = &nan_input;
    command.descriptor.softmax.output = &nan_output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OUT_OF_RANGE &&
                       memcmp(m3_op_test_f32(&nan_output), sentinel,
                              sizeof(sentinel)) == 0,
                   "NaN softmax failure leaves output unchanged");
    m3_op_test_fixture_dispose(&fixture);
}

void m3_test_op_sampling(m3_test_context *test)
{
    const uint64_t logits_shape[] = {1U, 4U};
    const uint64_t top_shape[] = {1U, 3U};
    const float logits_values[] = {3, 5, 5, 1};
    const float top_sentinel[] = {-9, -9, -9};
    const int32_t index_sentinel[] = {-9, -9, -9};
    const float nan_logits[] = {3, NAN, 5, 1};
    const uint64_t probability_shape[] = {4U, 3U};
    const uint64_t row_shape[] = {4U};
    const float probabilities[] = {
        0.2F, 0.3F, 0.5F, 0.2F, 0.3F, 0.5F,
        0.2F, 0.3F, 0.5F, 0.2F, 0.3F, 0.5F
    };
    const float uniforms[] = {0.0F, 0.2F, 0.5F, 0.999F};
    const int32_t categorical_sentinel[] = {-4, -4, -4, -4};
    const int32_t categorical_expected[] = {0, 1, 2, 2};
    m3_op_test_fixture fixture;
    m3_tensor_view logits;
    m3_tensor_view top_values;
    m3_tensor_view top_indices;
    m3_tensor_view probability_view;
    m3_tensor_view uniform_view;
    m3_tensor_view categorical_output;
    m3_command command;
    m3_error error;
    size_t scratch_bytes = 0U;

    M3_TEST_EXPECT(test, m3_op_test_fixture_init(&fixture),
                   "create sampling fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &logits, M3_DTYPE_F32, 2U,
                                     logits_shape, logits_values) &&
                       m3_op_test_tensor(&fixture, &top_values,
                                         M3_DTYPE_F32, 2U, top_shape,
                                         top_sentinel) &&
                       m3_op_test_tensor(&fixture, &top_indices,
                                         M3_DTYPE_I32, 2U, top_shape,
                                         index_sentinel),
                   "create top-k tensors");
    command.kind = M3_OP_TOP_K;
    command.descriptor.top_k.logits = &logits;
    command.descriptor.top_k.values = &top_values;
    command.descriptor.top_k.indices = &top_indices;
    command.descriptor.top_k.k = 3U;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U,
                                      &scratch_bytes, &error) == M3_STATUS_OK,
                   "execute top-k selection");
    M3_TEST_EXPECT(test, scratch_bytes >= 3U * 8U,
                   "top-k reports K pair scratch storage");
    M3_TEST_EXPECT(test,
                   m3_op_test_f32(&top_values)[0] == 5.0F &&
                       m3_op_test_f32(&top_values)[1] == 5.0F &&
                       m3_op_test_f32(&top_values)[2] == 3.0F &&
                       m3_op_test_i32(&top_indices)[0] == 1 &&
                       m3_op_test_i32(&top_indices)[1] == 2 &&
                       m3_op_test_i32(&top_indices)[2] == 0,
                   "top-k sorts descending and breaks ties by lower index");
    (void)m3_storage_write(logits.storage, 0U, nan_logits,
                           sizeof(nan_logits), &error);
    (void)m3_storage_write(top_values.storage, 0U, top_sentinel,
                           sizeof(top_sentinel), &error);
    (void)m3_storage_write(top_indices.storage, 0U, index_sentinel,
                           sizeof(index_sentinel), &error);
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OUT_OF_RANGE &&
                       memcmp(m3_op_test_f32(&top_values), top_sentinel,
                              sizeof(top_sentinel)) == 0 &&
                       memcmp(m3_op_test_i32(&top_indices), index_sentinel,
                              sizeof(index_sentinel)) == 0,
                   "top-k NaN rejection leaves both outputs unchanged");
    M3_TEST_EXPECT(test,
                   m3_op_test_tensor(&fixture, &probability_view,
                                     M3_DTYPE_F32, 2U, probability_shape,
                                     probabilities) &&
                       m3_op_test_tensor(&fixture, &uniform_view,
                                         M3_DTYPE_F32, 1U, row_shape,
                                         uniforms) &&
                       m3_op_test_tensor(&fixture, &categorical_output,
                                         M3_DTYPE_I32, 1U, row_shape,
                                         categorical_sentinel),
                   "create categorical tensors");
    command.kind = M3_OP_CATEGORICAL;
    command.descriptor.categorical.probabilities = &probability_view;
    command.descriptor.categorical.uniforms = &uniform_view;
    command.descriptor.categorical.output = &categorical_output;
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OK &&
                       memcmp(m3_op_test_i32(&categorical_output),
                              categorical_expected,
                              sizeof(categorical_expected)) == 0,
                   "categorical boundaries use half-open cumulative bins");
    m3_op_test_f32(&probability_view)[10] = -0.1F;
    (void)m3_storage_write(categorical_output.storage, 0U,
                           categorical_sentinel,
                           sizeof(categorical_sentinel), &error);
    M3_TEST_EXPECT(test,
                   m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                           M3_STATUS_OUT_OF_RANGE &&
                       memcmp(m3_op_test_i32(&categorical_output),
                              categorical_sentinel,
                              sizeof(categorical_sentinel)) == 0,
                   "invalid later categorical row leaves all rows unchanged");
    m3_op_test_fixture_dispose(&fixture);
}

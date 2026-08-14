/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_test.h"
#include "m3_qwen_internal.h"
#include "m3_test.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

void m3_test_qwen_semantic_embedding(m3_test_context *test)
{
    const uint64_t rows = M3_QWEN_SEMANTIC_TOKEN_START +
                          M3_QWEN_SEMANTIC_TOKEN_COUNT;
    const uint64_t shape[] = {rows, 1U};
    const size_t first_offset =
        (size_t)M3_QWEN_SEMANTIC_TOKEN_START * sizeof(uint16_t);
    const size_t last_offset =
        (size_t)(rows - 1U) * sizeof(uint16_t);
    m3_op_test_fixture fixture;
    m3_qwen_runtime runtime;
    m3_tensor_view table;
    m3_tensor_view first;
    m3_tensor_view last;
    m3_tensor_view unchanged;
    m3_tensor_view output;
    m3_storage *storage = NULL;
    uint16_t *bits;
    m3_error error;
    bool ready;

    (void)memset(&runtime, 0, sizeof(runtime));
    (void)memset(&unchanged, 0x5a, sizeof(unchanged));
    output = unchanged;
    m3_tensor_view_init(&table);
    m3_tensor_view_init(&first);
    m3_tensor_view_init(&last);
    ready = m3_op_test_fixture_init(&fixture) &&
            m3_op_test_storage(
                &fixture, (size_t)rows * sizeof(uint16_t), &storage) &&
            m3_tensor_view_contiguous(
                &table, storage, M3_DTYPE_BF16, 2U, shape, 0U,
                &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready,
                   "create sparse official semantic embedding fixture");
    if (!ready) {
        m3_op_test_fixture_dispose(&fixture);
        return;
    }
    bits = m3_storage_data(storage);
    (void)memset(bits, 0, (size_t)rows * sizeof(*bits));
    bits[M3_QWEN_SEMANTIC_TOKEN_START] = 0x3f01U;
    bits[rows - 1U] = 0xbf23U;
    runtime.dimensions.hidden_size = 1U;
    runtime.dimensions.semantic_token_start =
        M3_QWEN_SEMANTIC_TOKEN_START;
    runtime.dimensions.semantic_token_count =
        M3_QWEN_SEMANTIC_TOKEN_COUNT;
    runtime.weights.embedding = &table;
    runtime.forward.token_count = 37U;
    runtime.forward.published.count = 5U;

    M3_TEST_EXPECT(
        test,
        m3_qwen_runtime_semantic_embedding(
            &runtime, 0U, &first, &error) == M3_STATUS_OK &&
            first.storage == storage && first.metadata.dtype == M3_DTYPE_BF16 &&
            first.metadata.rank == 1U && first.metadata.shape[0] == 1U &&
            first.byte_offset == first_offset &&
            m3_op_test_u16(&first)[0] == 0x3f01U,
        "map semantic code zero to official language-model row 151675");
    M3_TEST_EXPECT(
        test,
        m3_qwen_runtime_semantic_embedding(
            &runtime, M3_QWEN_SEMANTIC_TOKEN_COUNT - 1U, &last,
            &error) == M3_STATUS_OK &&
            last.storage == storage && last.metadata.rank == 1U &&
            last.metadata.shape[0] == 1U &&
            last.byte_offset == last_offset &&
            m3_op_test_u16(&last)[0] == 0xbf23U,
        "map semantic code 16383 to official language-model row 168058");
    M3_TEST_EXPECT(
        test,
        m3_qwen_runtime_semantic_embedding(
            &runtime, M3_QWEN_SEMANTIC_TOKEN_COUNT, &output,
            NULL) == M3_STATUS_OUT_OF_RANGE &&
            memcmp(&output, &unchanged, sizeof(output)) == 0 &&
            runtime.forward.token_count == 37U &&
            runtime.forward.published.count == 5U,
        "reject code 16384 without advancing or invalidating state");
    m3_op_test_fixture_dispose(&fixture);
}

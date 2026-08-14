/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3.h"
#include "m3_tensor.h"

#include <stdint.h>
#include <string.h>

void m3_test_tensor_contract(m3_test_context *test)
{
    const uint64_t shape[] = {2U, 3U, 4U};
    const uint64_t zero_shape[] = {7U, 0U, UINT64_MAX};
    const uint64_t element_overflow_shape[] = {(uint64_t)SIZE_MAX, 2U};
    const uint64_t byte_overflow_shape[] = {
        (uint64_t)(SIZE_MAX / 4U) + 1U
    };
    uint64_t maximum_rank_shape[M3_TENSOR_MAX_RANK];
    m3_tensor_metadata metadata;
    m3_error error;
    size_t index;

    for (index = 0U; index < M3_TENSOR_MAX_RANK; ++index) {
        maximum_rank_shape[index] = 1U;
    }

    M3_TEST_EXPECT(test, m3_dtype_size(M3_DTYPE_F32) == 4U,
                   "F32 byte size");
    M3_TEST_EXPECT(test, m3_dtype_size(M3_DTYPE_F16) == 2U,
                   "F16 byte size");
    M3_TEST_EXPECT(test, m3_dtype_size(M3_DTYPE_BF16) == 2U,
                   "BF16 byte size");
    M3_TEST_EXPECT(test, m3_dtype_size(M3_DTYPE_I32) == 4U,
                   "I32 byte size");
    M3_TEST_EXPECT(test, strcmp(m3_dtype_name(M3_DTYPE_I32), "I32") == 0,
                   "I32 dtype name");
    M3_TEST_EXPECT(test, strcmp(m3_dtype_name(M3_DTYPE_BF16), "BF16") == 0,
                   "BF16 dtype name");
    M3_TEST_EXPECT(test, m3_dtype_size((m3_dtype)99) == 0U,
                   "unknown dtype size");

    M3_TEST_EXPECT(test,
                   m3_tensor_metadata_init(&metadata, M3_DTYPE_F32, 3U,
                                           shape, &error) == M3_STATUS_OK,
                   "initialize F32 tensor metadata");
    M3_TEST_EXPECT(test, metadata.rank == 3U, "tensor rank");
    M3_TEST_EXPECT(test, metadata.element_count == 24U,
                   "tensor element count");
    M3_TEST_EXPECT(test, metadata.byte_count == 96U, "tensor byte count");
    M3_TEST_EXPECT(test,
                   metadata.shape[0] == 2U && metadata.shape[2] == 4U,
                   "tensor shape copy");
    M3_TEST_EXPECT(test, error.status == M3_STATUS_OK,
                   "tensor success clears error");

    M3_TEST_EXPECT(test,
                   m3_tensor_metadata_init(&metadata, M3_DTYPE_BF16, 0U,
                                           NULL, &error) == M3_STATUS_OK,
                   "initialize scalar tensor metadata");
    M3_TEST_EXPECT(test,
                   metadata.element_count == 1U && metadata.byte_count == 2U,
                   "scalar tensor counts");

    M3_TEST_EXPECT(test,
                   m3_tensor_metadata_init(&metadata, M3_DTYPE_F16,
                                           M3_TENSOR_MAX_RANK,
                                           maximum_rank_shape,
                                           &error) == M3_STATUS_OK,
                   "initialize maximum-rank metadata");
    M3_TEST_EXPECT(test, metadata.element_count == 1U,
                   "maximum-rank element count");

    M3_TEST_EXPECT(test,
                   m3_tensor_metadata_init(&metadata, M3_DTYPE_F16, 3U,
                                           zero_shape, &error) == M3_STATUS_OK,
                   "initialize empty tensor metadata");
    M3_TEST_EXPECT(test,
                   metadata.element_count == 0U && metadata.byte_count == 0U,
                   "empty tensor counts");

    M3_TEST_EXPECT(test,
                   m3_tensor_metadata_init(&metadata, M3_DTYPE_F16,
                                           M3_TENSOR_MAX_RANK + 1U, shape,
                                           &error) == M3_STATUS_OUT_OF_RANGE,
                   "reject excessive tensor rank");
    M3_TEST_EXPECT(test, metadata.element_count == 0U,
                   "clear rejected tensor output");
    M3_TEST_EXPECT(test,
                   m3_tensor_metadata_init(&metadata, M3_DTYPE_F16, 1U,
                                           NULL, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject missing tensor shape");
    M3_TEST_EXPECT(test,
                   m3_tensor_metadata_init(&metadata, (m3_dtype)99, 0U,
                                           NULL, &error) ==
                       M3_STATUS_UNSUPPORTED,
                   "reject unknown tensor dtype");
    M3_TEST_EXPECT(test,
                   m3_tensor_metadata_init(&metadata, M3_DTYPE_F16, 2U,
                                           element_overflow_shape,
                                           &error) == M3_STATUS_OVERFLOW,
                   "reject tensor element overflow");
    M3_TEST_EXPECT(test,
                   m3_tensor_metadata_init(&metadata, M3_DTYPE_F32, 1U,
                                           byte_overflow_shape,
                                           &error) == M3_STATUS_OVERFLOW,
                   "reject tensor byte overflow");
    M3_TEST_EXPECT(test,
                   m3_tensor_metadata_init(NULL, M3_DTYPE_F32, 0U, NULL,
                                           &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject null tensor metadata output");
}

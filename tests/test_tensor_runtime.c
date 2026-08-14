/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_backend.h"
#include "m3_scratch.h"
#include "m3_tensor.h"
#include "m3_weights.h"

#include <stdint.h>
#include <string.h>

void m3_test_host_storage_lifetime(m3_test_context *test)
{
    const uint8_t source[] = {1U, 3U, 5U, 7U, 9U, 11U, 13U, 15U};
    uint8_t output[sizeof(source)] = {0U};
    m3_backend_allocation_stats stats;
    m3_backend_info info;
    m3_storage *storage = NULL;
    m3_storage *automatic = NULL;
    m3_backend *backend = NULL;
    m3_error error;

    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&backend, &error) == M3_STATUS_OK,
                   "create host storage backend");
    if (backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_backend_get_info(backend, &info, &error) ==
                       M3_STATUS_OK &&
                       info.kind == M3_BACKEND_HOST && info.unified_memory,
                   "host backend reports its storage contract");
    M3_TEST_EXPECT(test,
                   m3_storage_allocate(backend, sizeof(source), 32U,
                                       &storage, &error) == M3_STATUS_OK,
                   "allocate aligned host storage");
    M3_TEST_EXPECT(test,
                   storage != NULL && m3_storage_size(storage) ==
                                          sizeof(source) &&
                       m3_storage_backend(storage) == backend &&
                       (uintptr_t)m3_storage_data(storage) % 32U == 0U,
                   "host storage exposes checked raw properties");
    M3_TEST_EXPECT(test,
                   m3_storage_write(storage, 0U, source, sizeof(source),
                                    &error) == M3_STATUS_OK &&
                       m3_storage_read(storage, 0U, output, sizeof(output),
                                       &error) == M3_STATUS_OK &&
                       memcmp(source, output, sizeof(source)) == 0,
                   "host storage round trip is exact");
    M3_TEST_EXPECT(test,
                   m3_storage_write(storage, sizeof(source), source, 1U,
                                    &error) == M3_STATUS_OUT_OF_RANGE,
                   "host storage rejects an excessive range");
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(backend, &stats, &error) ==
                       M3_STATUS_OK &&
                       stats.live_allocated_bytes == sizeof(source) &&
                       stats.live_storage_count == 1U,
                   "host backend tracks its storage owner");
    m3_storage_free(storage);
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(backend, &stats, &error) ==
                       M3_STATUS_OK && stats.live_allocated_bytes == 0U &&
                       stats.live_storage_count == 0U,
                   "explicit storage free clears live ownership");
    M3_TEST_EXPECT(test,
                   m3_storage_allocate(backend, 17U, 16U, &automatic,
                                       &error) == M3_STATUS_OK,
                   "allocate storage owned until backend destruction");
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(NULL, &error) ==
                       M3_STATUS_INVALID_ARGUMENT &&
                       m3_storage_allocate(NULL, 1U, 8U, &storage, &error) ==
                           M3_STATUS_INVALID_ARGUMENT,
                   "backend lifetime APIs validate null owners");
    m3_backend_free(backend);
}

void m3_test_tensor_views(m3_test_context *test)
{
    const uint64_t shape[] = {2U, 3U};
    const uint64_t reshaped_shape[] = {3U, 2U};
    const uint8_t permutation[] = {1U, 0U};
    const uint64_t transposed_index[] = {2U, 1U};
    const uint64_t bad_index[] = {3U, 0U};
    m3_backend_allocation_stats stats;
    m3_tensor_view tensor;
    m3_tensor_view transposed;
    m3_tensor_view reshaped;
    m3_tensor_view slice;
    m3_storage *storage = NULL;
    m3_backend *backend = NULL;
    m3_error error;
    size_t offset = 0U;
    void *tensor_data = NULL;

    m3_tensor_view_init(&tensor);
    m3_tensor_view_init(&transposed);
    m3_tensor_view_init(&reshaped);
    m3_tensor_view_init(&slice);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
                       m3_storage_allocate(backend, 24U, 16U, &storage,
                                           &error) == M3_STATUS_OK,
                   "prepare tensor view storage");
    if (storage == NULL) {
        m3_backend_free(backend);
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_tensor_view_contiguous(&tensor, storage, M3_DTYPE_F32,
                                             2U, shape, 0U, &error) ==
                       M3_STATUS_OK,
                   "construct a contiguous tensor view");
    M3_TEST_EXPECT(test,
                   tensor.byte_strides[0] == 12U &&
                       tensor.byte_strides[1] == 4U &&
                       m3_tensor_is_contiguous(&tensor),
                   "contiguous view has canonical row-major byte strides");
    M3_TEST_EXPECT(test,
                   m3_tensor_data(&tensor, &tensor_data, &error) ==
                           M3_STATUS_OK &&
                       tensor_data == m3_storage_data(storage),
                   "tensor pointer addresses its storage offset");
    M3_TEST_EXPECT(test,
                   m3_tensor_permute(&tensor, permutation, &transposed,
                                     &error) == M3_STATUS_OK,
                   "transpose tensor axes as a view");
    M3_TEST_EXPECT(test,
                   transposed.metadata.shape[0] == 3U &&
                       transposed.metadata.shape[1] == 2U &&
                       transposed.byte_strides[0] == 4U &&
                       transposed.byte_strides[1] == 12U &&
                       !m3_tensor_is_contiguous(&transposed),
                   "transposed view preserves strided layout");
    M3_TEST_EXPECT(test,
                   m3_tensor_element_offset(&transposed, transposed_index,
                                            &offset, &error) == M3_STATUS_OK &&
                       offset == 20U,
                   "transposed element offset follows byte strides");
    M3_TEST_EXPECT(test,
                   m3_tensor_element_offset(&transposed, bad_index, &offset,
                                            &error) ==
                       M3_STATUS_OUT_OF_RANGE,
                   "element offset rejects out-of-range indices");
    M3_TEST_EXPECT(test,
                   m3_tensor_reshape(&tensor, 2U, reshaped_shape, &reshaped,
                                     &error) == M3_STATUS_OK &&
                       reshaped.byte_strides[0] == 8U &&
                       reshaped.byte_strides[1] == 4U,
                   "reshape creates canonical contiguous strides");
    M3_TEST_EXPECT(test,
                   m3_tensor_reshape(&transposed, 2U, reshaped_shape, &slice,
                                     &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reshape rejects non-contiguous input");
    M3_TEST_EXPECT(test,
                   m3_tensor_slice(&tensor, 0U, 1U, 1U, &slice, &error) ==
                           M3_STATUS_OK &&
                       slice.byte_offset == 12U &&
                       slice.metadata.shape[0] == 1U,
                   "slice derives a bounded non-owning view");
    m3_tensor_view_init(&tensor);
    m3_tensor_view_init(&transposed);
    m3_tensor_view_init(&reshaped);
    m3_tensor_view_init(&slice);
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(backend, &stats, &error) ==
                           M3_STATUS_OK &&
                       stats.live_storage_count == 1U,
                   "clearing views never frees their storage owner");
    m3_storage_free(storage);
    m3_backend_free(backend);
}

void m3_test_tensor_view_rejections(m3_test_context *test)
{
    const uint64_t shape[] = {2U, 3U};
    const uint64_t empty_shape[] = {0U, UINT64_MAX};
    const uint8_t duplicate_permutation[] = {0U, 0U};
    const size_t excessive_strides[] = {SIZE_MAX - 3U, 4U};
    const size_t misaligned_strides[] = {12U, 2U};
    m3_tensor_view tensor;
    m3_tensor_view rejected;
    m3_tensor_view empty;
    m3_storage *storage = NULL;
    m3_storage *empty_storage = NULL;
    m3_backend *backend = NULL;
    m3_error error;
    const void *empty_data = (const void *)(uintptr_t)1U;

    m3_tensor_view_init(&tensor);
    m3_tensor_view_init(&rejected);
    m3_tensor_view_init(&empty);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
                       m3_storage_allocate(backend, 24U, 16U, &storage,
                                           &error) == M3_STATUS_OK &&
                       m3_storage_allocate(backend, 0U, 16U, &empty_storage,
                                           &error) == M3_STATUS_OK,
                   "prepare rejection fixtures");
    if (storage == NULL || empty_storage == NULL) {
        m3_backend_free(backend);
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_tensor_view_contiguous(&tensor, storage, M3_DTYPE_F32,
                                             2U, shape, 0U, &error) ==
                       M3_STATUS_OK,
                   "construct rejection source view");
    M3_TEST_EXPECT(test,
                   m3_tensor_permute(&tensor, duplicate_permutation,
                                     &rejected, &error) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       rejected.storage == NULL,
                   "reject duplicate permutation axes atomically");
    M3_TEST_EXPECT(test,
                   m3_tensor_view_contiguous(&rejected, storage,
                                             M3_DTYPE_F32, 2U, shape, 1U,
                                             &error) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       rejected.storage == NULL,
                   "reject a misaligned tensor byte offset");
    M3_TEST_EXPECT(test,
                   m3_tensor_view_strided(&rejected, storage, M3_DTYPE_F32,
                                          2U, shape, misaligned_strides, 0U,
                                          &error) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       rejected.storage == NULL,
                   "reject a misaligned byte stride");
    M3_TEST_EXPECT(test,
                   m3_tensor_view_strided(&rejected, storage, M3_DTYPE_F32,
                                          2U, shape, excessive_strides, 0U,
                                          &error) == M3_STATUS_OVERFLOW &&
                       rejected.storage == NULL,
                   "reject overflowing maximum reachable byte");
    M3_TEST_EXPECT(test,
                   m3_tensor_view_contiguous(&rejected, storage,
                                             M3_DTYPE_F32, 2U, shape, 4U,
                                             &error) ==
                           M3_STATUS_OUT_OF_RANGE &&
                       rejected.storage == NULL,
                   "reject a contiguous view that exceeds storage");
    M3_TEST_EXPECT(test,
                   m3_tensor_view_contiguous(&empty, empty_storage,
                                             M3_DTYPE_I32, 2U, empty_shape,
                                             0U, &error) == M3_STATUS_OK &&
                       empty.metadata.byte_count == 0U &&
                       empty.byte_strides[0] == 0U &&
                       empty.byte_strides[1] == 0U &&
                       m3_tensor_is_contiguous(&empty),
                   "empty I32 view has checked zero strides");
    M3_TEST_EXPECT(test,
                   m3_tensor_const_data(&empty, &empty_data, &error) ==
                           M3_STATUS_OK &&
                       empty_data == NULL,
                   "empty storage exposes no invalid pointer");
    m3_storage_free(empty_storage);
    m3_storage_free(storage);
    m3_backend_free(backend);
}

void m3_test_scratch_arena(m3_test_context *test)
{
    uint8_t backing[64];
    m3_scratch_arena arena;
    m3_error error;
    void *first = NULL;
    void *second = NULL;
    void *failed = (void *)(uintptr_t)1U;
    size_t first_end;
    size_t mark;

    M3_TEST_EXPECT(test,
                   m3_scratch_arena_init(&arena, backing, sizeof(backing),
                                         &error) == M3_STATUS_OK,
                   "initialize caller-backed scratch arena");
    M3_TEST_EXPECT(test,
                   m3_scratch_arena_allocate(&arena, 7U, 16U, &first,
                                             &error) == M3_STATUS_OK &&
                       first != NULL && (uintptr_t)first % 16U == 0U,
                   "scratch allocation honors absolute alignment");
    first_end = arena.offset;
    mark = m3_scratch_arena_mark(&arena);
    M3_TEST_EXPECT(test, mark == first_end, "scratch mark captures offset");
    M3_TEST_EXPECT(test,
                   m3_scratch_arena_allocate(&arena, 5U, 8U, &second,
                                             &error) == M3_STATUS_OK &&
                       second != NULL && (uintptr_t)second % 8U == 0U,
                   "second scratch allocation is independently aligned");
    M3_TEST_EXPECT(test,
                   m3_scratch_arena_rewind(&arena, mark, &error) ==
                           M3_STATUS_OK &&
                       arena.offset == mark,
                   "scratch rewind restores a valid mark");
    M3_TEST_EXPECT(test,
                   m3_scratch_arena_allocate(&arena, sizeof(backing), 8U,
                                             &failed, &error) ==
                           M3_STATUS_OUT_OF_MEMORY &&
                       failed == NULL && arena.offset == mark,
                   "scratch exhaustion is atomic and reports out of memory");
    M3_TEST_EXPECT(test,
                   m3_scratch_arena_allocate(&arena, SIZE_MAX, 8U, &failed,
                                             &error) == M3_STATUS_OVERFLOW &&
                       failed == NULL && arena.offset == mark,
                   "scratch arithmetic overflow is distinct and atomic");
    M3_TEST_EXPECT(test,
                   m3_scratch_arena_allocate(&arena, 1U, 3U, &failed,
                                             &error) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       arena.offset == mark,
                   "scratch rejects non-power-of-two alignment atomically");
    M3_TEST_EXPECT(test,
                   m3_scratch_arena_rewind(&arena, mark + 1U, &error) ==
                           M3_STATUS_OUT_OF_RANGE &&
                       arena.offset == mark,
                   "scratch rejects a future rewind mark atomically");
    m3_scratch_arena_reset(&arena);
    M3_TEST_EXPECT(test, arena.offset == 0U,
                   "scratch reset preserves backing and clears offset");
}

static void m3_test_weight_fixture(m3_safetensors_metadata *metadata,
                                   m3_safetensors_tensor tensors[2],
                                   m3_error *error)
{
    const uint64_t alpha_shape[] = {2U, 2U};
    const uint64_t zeta_shape[] = {1U};

    (void)memset(metadata, 0, sizeof(*metadata));
    (void)memset(tensors, 0, 2U * sizeof(*tensors));
    tensors[0].name = "zeta.weight";
    (void)m3_tensor_metadata_init(&tensors[0].tensor, M3_DTYPE_F32, 1U,
                                  zeta_shape, error);
    tensors[0].data_start = 16U;
    tensors[0].data_end = 20U;
    tensors[1].name = "alpha.weight";
    (void)m3_tensor_metadata_init(&tensors[1].tensor, M3_DTYPE_F32, 2U,
                                  alpha_shape, error);
    tensors[1].data_start = 0U;
    tensors[1].data_end = 16U;
    metadata->tensors = tensors;
    metadata->tensor_count = 2U;
    metadata->tensor_bytes = 20U;
    metadata->data_section_offset = 128U;
}

void m3_test_weight_bindings(m3_test_context *test)
{
    m3_safetensors_tensor tensors[2];
    m3_safetensors_metadata metadata;
    m3_weight_shard shard;
    m3_weight_table table;
    m3_weight_requirement exact[2];
    m3_weight_requirement duplicate[2];
    m3_weight_requirement missing;
    const m3_weight_binding *alpha;
    m3_error error;

    m3_weight_table_init(&table);
    m3_test_weight_fixture(&metadata, tensors, &error);
    shard.shard_path = "model-00001-of-00001.safetensors";
    shard.metadata = &metadata;
    M3_TEST_EXPECT(test,
                   m3_weight_table_build(&table, &shard, 1U, &error) ==
                       M3_STATUS_OK,
                   "build metadata-only weight binding table");
    M3_TEST_EXPECT(test,
                   table.binding_count == 2U &&
                       strcmp(table.bindings[0].name, "alpha.weight") == 0 &&
                       strcmp(table.bindings[1].name, "zeta.weight") == 0,
                   "weight bindings are sorted for deterministic lookup");
    alpha = m3_weight_table_find(&table, "alpha.weight");
    M3_TEST_EXPECT(test,
                   alpha != NULL && alpha->absolute_file_start == 128U &&
                       alpha->absolute_file_end == 144U &&
                       strcmp(alpha->shard_path, shard.shard_path) == 0,
                   "binding stores an absolute checked shard byte range");
    exact[0].name = "alpha.weight";
    exact[0].tensor = tensors[1].tensor;
    exact[1].name = "zeta.weight";
    exact[1].tensor = tensors[0].tensor;
    M3_TEST_EXPECT(test,
                   m3_weight_table_validate_required(&table, exact, 2U,
                                                     &error) == M3_STATUS_OK &&
                       m3_weight_table_validate_no_extra(&table, exact, 2U,
                                                         &error) ==
                           M3_STATUS_OK,
                   "exact weight schema accepts all requested bindings");
    missing = exact[0];
    missing.name = "missing.weight";
    M3_TEST_EXPECT(test,
                   m3_weight_table_validate_required(&table, &missing, 1U,
                                                     &error) ==
                           M3_STATUS_INVALID_FORMAT &&
                       strstr(m3_error_message(&error), "missing required") !=
                           NULL,
                   "missing requested weight has an explicit diagnostic");
    M3_TEST_EXPECT(test,
                   m3_weight_table_validate_no_extra(&table, exact, 1U,
                                                     &error) ==
                           M3_STATUS_INVALID_FORMAT &&
                       strstr(m3_error_message(&error), "unexpected extra") !=
                           NULL,
                   "extra inspected weight has an explicit diagnostic");
    duplicate[0] = exact[0];
    duplicate[1] = exact[0];
    M3_TEST_EXPECT(test,
                   m3_weight_table_validate_required(&table, duplicate, 2U,
                                                     &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "duplicate requested names are rejected");
    exact[0].tensor.dtype = M3_DTYPE_F16;
    M3_TEST_EXPECT(test,
                   m3_weight_table_validate_required(&table, exact, 2U,
                                                     &error) ==
                       M3_STATUS_INVALID_FORMAT,
                   "weight schema requires exact dtype and shape");
    m3_weight_table_dispose(&table);
}

void m3_test_weight_binding_rejections(m3_test_context *test)
{
    m3_safetensors_tensor tensors[2];
    m3_safetensors_tensor duplicate_tensor;
    m3_safetensors_metadata metadata;
    m3_safetensors_metadata duplicate_metadata;
    m3_weight_shard shards[2];
    m3_weight_table table;
    m3_error error;

    m3_weight_table_init(&table);
    m3_test_weight_fixture(&metadata, tensors, &error);
    duplicate_tensor = tensors[0];
    duplicate_metadata = metadata;
    duplicate_metadata.tensors = &duplicate_tensor;
    duplicate_metadata.tensor_count = 1U;
    shards[0].shard_path = "first.safetensors";
    shards[0].metadata = &metadata;
    shards[1].shard_path = "second.safetensors";
    shards[1].metadata = &duplicate_metadata;
    M3_TEST_EXPECT(test,
                   m3_weight_table_build(&table, shards, 2U, &error) ==
                           M3_STATUS_INVALID_FORMAT &&
                       strstr(m3_error_message(&error), "duplicate weight") !=
                           NULL &&
                       table.binding_count == 0U,
                   "duplicate names across shards are rejected atomically");
    tensors[0].data_end = tensors[0].data_start + 2U;
    M3_TEST_EXPECT(test,
                   m3_weight_table_build(&table, shards, 1U, &error) ==
                           M3_STATUS_INVALID_FORMAT &&
                       strstr(m3_error_message(&error), "range disagrees") !=
                           NULL,
                   "weight byte range must agree with tensor metadata");
    m3_test_weight_fixture(&metadata, tensors, &error);
    metadata.data_section_offset = UINT64_MAX - 1U;
    M3_TEST_EXPECT(test,
                   m3_weight_table_build(&table, shards, 1U, &error) ==
                       M3_STATUS_OVERFLOW,
                   "absolute weight file range overflow is rejected");
    m3_test_weight_fixture(&metadata, tensors, &error);
    tensors[0].name = NULL;
    M3_TEST_EXPECT(test,
                   m3_weight_table_build(&table, shards, 1U, NULL) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "null weight name returns exact status without diagnostics");
    m3_weight_table_dispose(&table);
}

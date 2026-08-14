/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_test.h"

#include <stdlib.h>
#include <string.h>

bool m3_op_test_fixture_init(m3_op_test_fixture *fixture)
{
    m3_backend *backend = NULL;
    m3_error error;

    if (fixture == NULL) {
        return false;
    }
    if (m3_backend_create_host(&backend, &error) != M3_STATUS_OK) {
        return false;
    }
    return m3_op_test_fixture_init_backend(fixture, backend, true);
}

bool m3_op_test_fixture_init_backend(m3_op_test_fixture *fixture,
                                     m3_backend *backend,
                                     bool owns_backend)
{
    if (fixture == NULL || backend == NULL) {
        if (owns_backend) {
            m3_backend_free(backend);
        }
        return false;
    }
    (void)memset(fixture, 0, sizeof(*fixture));
    fixture->backend = backend;
    fixture->owns_backend = owns_backend;
    return true;
}

void m3_op_test_fixture_dispose(m3_op_test_fixture *fixture)
{
    size_t index;

    if (fixture == NULL) {
        return;
    }
    for (index = fixture->storage_count; index > 0U; --index) {
        m3_storage_free(fixture->storages[index - 1U]);
    }
    if (fixture->owns_backend) {
        m3_backend_free(fixture->backend);
    }
    (void)memset(fixture, 0, sizeof(*fixture));
}

bool m3_op_test_storage(m3_op_test_fixture *fixture, size_t byte_count,
                        m3_storage **storage)
{
    m3_error error;

    if (fixture == NULL || storage == NULL ||
        fixture->storage_count >= M3_OP_TEST_STORAGE_CAPACITY) {
        return false;
    }
    *storage = NULL;
    if (m3_storage_allocate(fixture->backend, byte_count, 16U, storage,
                            &error) != M3_STATUS_OK) {
        return false;
    }
    fixture->storages[fixture->storage_count++] = *storage;
    return true;
}

bool m3_op_test_tensor(m3_op_test_fixture *fixture, m3_tensor_view *view,
                       m3_dtype dtype, uint8_t rank,
                       const uint64_t *shape, const void *values)
{
    m3_tensor_metadata metadata;
    m3_storage *storage = NULL;
    m3_error error;

    m3_tensor_view_init(view);
    if (m3_tensor_metadata_init(&metadata, dtype, rank, shape, &error) !=
            M3_STATUS_OK ||
        !m3_op_test_storage(fixture, metadata.byte_count, &storage)) {
        return false;
    }
    if (metadata.byte_count != 0U &&
        (values == NULL ||
         m3_storage_write(storage, 0U, values, metadata.byte_count, &error) !=
             M3_STATUS_OK)) {
        return false;
    }
    return m3_tensor_view_contiguous(view, storage, dtype, rank, shape, 0U,
                                     &error) == M3_STATUS_OK;
}

m3_status m3_op_test_execute(m3_op_test_fixture *fixture,
                             const m3_command *commands,
                             size_t command_count, size_t *scratch_bytes,
                             m3_error *error)
{
    m3_scratch_arena arena;
    m3_scratch_arena *arena_pointer = NULL;
    size_t required = 0U;
    void *memory = NULL;
    m3_status status = m3_commands_scratch_bytes(
        fixture->backend, commands, command_count, &required, error);

    if (scratch_bytes != NULL) {
        *scratch_bytes = required;
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (required != 0U) {
        memory = malloc(required);
        if (memory == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "test scratch allocation failed");
        }
        status = m3_scratch_arena_init(&arena, memory, required, error);
        if (status != M3_STATUS_OK) {
            free(memory);
            return status;
        }
        arena_pointer = &arena;
    }
    status = m3_backend_execute(fixture->backend, commands, command_count,
                                arena_pointer, error);
    free(memory);
    return status;
}

float *m3_op_test_f32(m3_tensor_view *view)
{
    return (float *)((uint8_t *)m3_storage_data(view->storage) +
                     view->byte_offset);
}

int32_t *m3_op_test_i32(m3_tensor_view *view)
{
    return (int32_t *)((uint8_t *)m3_storage_data(view->storage) +
                       view->byte_offset);
}

uint16_t *m3_op_test_u16(m3_tensor_view *view)
{
    return (uint16_t *)((uint8_t *)m3_storage_data(view->storage) +
                        view->byte_offset);
}

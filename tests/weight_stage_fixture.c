/* SPDX-License-Identifier: GPL-2.0-only */

#include "weight_stage_fixture.h"

#include "m3_op_internal.h"

#include <stdlib.h>
#include <string.h>

bool m3_weight_stage_test_fixture_create(
    m3_weight_stage_test_fixture *fixture, m3_error *error)
{
    static const char a_header[] =
        "{\"gamma\":{\"dtype\":\"BF16\",\"shape\":[2],"
        "\"data_offsets\":[6,10]},"
        "\"beta\":{\"dtype\":\"F16\",\"shape\":[3],"
        "\"data_offsets\":[0,6]}}";
    static const char z_header[] =
        "{\"zeta\":{\"dtype\":\"F16\",\"shape\":[2],"
        "\"data_offsets\":[8,12]},"
        "\"alpha\":{\"dtype\":\"F32\",\"shape\":[2],"
        "\"data_offsets\":[0,8]}}";
    m3_weight_shard_source sources[2];

    if (fixture == NULL) {
        return false;
    }
    (void)memset(fixture, 0, sizeof(*fixture));
    m3_safetensors_metadata_init(&fixture->a_metadata);
    m3_safetensors_metadata_init(&fixture->z_metadata);
    m3_weight_table_init(&fixture->table);
    if (!m3_loader_test_make_root(fixture->root) ||
        !m3_loader_test_path(fixture->a_path, fixture->root,
                             "a-shard.safetensors") ||
        !m3_loader_test_path(fixture->z_path, fixture->root,
                             "z-shard.safetensors") ||
        !m3_loader_test_write_safetensors(fixture->a_path, a_header, 10U) ||
        !m3_loader_test_write_safetensors(fixture->z_path, z_header, 12U) ||
        m3_safetensors_inspect_file(fixture->a_path, &fixture->a_metadata,
                                    error) != M3_STATUS_OK ||
        m3_safetensors_inspect_file(fixture->z_path, &fixture->z_metadata,
                                    error) != M3_STATUS_OK) {
        (void)m3_weight_stage_test_fixture_dispose(fixture);
        return false;
    }
    sources[0].path = fixture->z_path;
    sources[0].metadata = &fixture->z_metadata;
    sources[1].path = fixture->a_path;
    sources[1].metadata = &fixture->a_metadata;
    if (m3_weight_table_build(&fixture->table, sources, 2U, error) !=
        M3_STATUS_OK) {
        (void)m3_weight_stage_test_fixture_dispose(fixture);
        return false;
    }
    return true;
}

bool m3_weight_stage_test_fixture_dispose(
    m3_weight_stage_test_fixture *fixture)
{
    bool removed;

    if (fixture == NULL) {
        return false;
    }
    m3_weight_table_dispose(&fixture->table);
    m3_safetensors_metadata_dispose(&fixture->a_metadata);
    m3_safetensors_metadata_dispose(&fixture->z_metadata);
    removed = fixture->root[0] == '\0' ||
              m3_loader_test_remove_tree(fixture->root);
    (void)memset(fixture, 0, sizeof(*fixture));
    return removed;
}

static void m3_weight_stage_fake_destroy(void *context)
{
    free(context);
}

static m3_status m3_weight_stage_fake_allocate(
    void *context_pointer, size_t byte_count, size_t alignment,
    void **handle, void **data, m3_error *error)
{
    m3_weight_stage_fake_context *context = context_pointer;
    size_t allocation_size = byte_count;
    size_t remainder;
    void *memory;

    *handle = NULL;
    *data = NULL;
    ++context->allocation_calls;
    if (context->allocation_calls == context->fail_allocation_call) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "injected staged allocation failure");
    }
    if (byte_count == 0U) {
        return M3_STATUS_OK;
    }
    remainder = allocation_size & (alignment - 1U);
    if (remainder != 0U) {
        if (allocation_size > SIZE_MAX - (alignment - remainder)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "fake staged allocation size overflows");
        }
        allocation_size += alignment - remainder;
    }
    memory = aligned_alloc(alignment, allocation_size);
    if (memory == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate fake staged storage");
    }
    *handle = memory;
    *data = memory;
    return M3_STATUS_OK;
}

static void m3_weight_stage_fake_free(void *context_pointer, void *handle,
                                      void *data)
{
    m3_weight_stage_fake_context *context = context_pointer;

    (void)data;
    ++context->free_calls;
    free(handle);
}

static m3_status m3_weight_stage_fake_execute(
    void *context, const m3_command *commands, size_t command_count,
    m3_scratch_arena *scratch, m3_error *error)
{
    (void)context;
    (void)commands;
    (void)command_count;
    (void)scratch;
    return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                        "fake backend does not execute commands");
}

bool m3_weight_stage_test_fake_backend_create(
    uint64_t maximum_storage_bytes,
    uint64_t recommended_working_set_bytes,
    size_t fail_allocation_call, m3_backend **backend,
    m3_weight_stage_fake_context **context_output, m3_error *error)
{
    static const m3_backend_vtable vtable = {
        m3_weight_stage_fake_destroy,
        m3_weight_stage_fake_allocate,
        m3_weight_stage_fake_free,
        m3_weight_stage_fake_execute
    };
    m3_weight_stage_fake_context *context;
    m3_backend_info info;

    if (backend == NULL || context_output == NULL) {
        return false;
    }
    *backend = NULL;
    *context_output = NULL;
    context = calloc(1U, sizeof(*context));
    if (context == NULL) {
        return false;
    }
    context->fail_allocation_call = fail_allocation_call;
    (void)memset(&info, 0, sizeof(info));
    (void)memcpy(info.name, "Staged weight test backend",
                 sizeof("Staged weight test backend"));
    info.kind = M3_BACKEND_HOST;
    info.unified_memory = true;
    info.maximum_storage_bytes = maximum_storage_bytes;
    info.recommended_working_set_bytes = recommended_working_set_bytes;
    if (m3_backend_create_internal(&vtable, context, &info, backend, error) !=
        M3_STATUS_OK) {
        free(context);
        return false;
    }
    *context_output = context;
    return true;
}

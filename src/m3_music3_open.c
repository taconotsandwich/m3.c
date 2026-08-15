/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_internal.h"

#include "m3_model_inspect_internal.h"
#include "m3_music3_schema.h"

#include <stdlib.h>

static void m3_music3_engine_dispose(m3_music3_engine *engine)
{
    size_t index;

    if (engine == NULL) {
        return;
    }
    for (index = 0U; index < M3_MUSIC3_WEIGHT_COMPONENT_COUNT; ++index) {
        m3_weight_table_dispose(&engine->tables[index]);
    }
    m3_tokenizer_dispose(&engine->tokenizer);
    m3_backend_free(engine->backend);
    free(engine);
}

static m3_status m3_music3_validate_backend(m3_backend *backend,
                                             m3_error *error)
{
    m3_backend_info info;
    m3_status status = m3_backend_get_info(backend, &info, error);

    if (status == M3_STATUS_OK && info.kind != M3_BACKEND_METAL) {
        status = m3_error_set(error, M3_STATUS_UNSUPPORTED,
                              "Music3 requires one Metal backend");
    }
    return status;
}

static m3_status m3_music3_load_tables(
    m3_music3_engine *engine, const char *root, m3_error *error)
{
    size_t index;
    m3_status status = M3_STATUS_OK;

    for (index = 0U; index < M3_MUSIC3_WEIGHT_COMPONENT_COUNT &&
                    status == M3_STATUS_OK; ++index) {
        m3_music3_allocation_plan plan;
        m3_music3_schema_summary expected;

        status = engine->operations.inspect_weights(
            engine->operations.context, root, (m3_component_id)index,
            &engine->tables[index], error);
        if (status == M3_STATUS_OK) {
            status = m3_music3_schema_expected_summary(
                (m3_component_id)index, &expected, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_music3_table_plan(
                &engine->tables[index], &plan, error);
        }
        if (status == M3_STATUS_OK &&
            plan.added_bytes != expected.payload_bytes) {
            status = m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "Music3 component weight total is not official");
        }
        if (status == M3_STATUS_OK) {
            engine->component_payload_bytes[index] = plan.added_bytes;
            engine->component_largest_shard_bytes[index] =
                plan.largest_storage_bytes;
        }
    }
    return status;
}

static m3_status m3_music3_load_exact_tokenizer(
    m3_music3_engine *engine, const char *root, m3_error *error)
{
    char *directory = m3_model_inspect_path_join(
        root, m3_component_directory(M3_COMPONENT_TOKENIZER), error);
    char *path = NULL;
    m3_status status;

    if (directory == NULL) {
        return error == NULL ? M3_STATUS_OUT_OF_MEMORY : error->status;
    }
    path = m3_model_inspect_path_join(directory, "tokenizer.json", error);
    free(directory);
    if (path == NULL) {
        return error == NULL ? M3_STATUS_OUT_OF_MEMORY : error->status;
    }
    status = engine->operations.load_tokenizer(
        engine->operations.context, &engine->tokenizer, path, error);
    free(path);
    return status;
}

m3_status m3_music3_engine_open_core(
    m3_music3_engine **engine, const char *trusted_model_root,
    const m3_music3_operations *operations, m3_error *error)
{
    m3_music3_engine *built;
    m3_model_metadata metadata;
    size_t index;
    m3_status status;

    if (engine == NULL || *engine != NULL || trusted_model_root == NULL ||
        trusted_model_root[0] == '\0' ||
        !m3_music3_operations_valid(operations)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 open arguments are invalid");
    }
    built = calloc(1U, sizeof(*built));
    if (built == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate Music3 engine");
    }
    built->operations = *operations;
    m3_tokenizer_init(&built->tokenizer);
    for (index = 0U; index < M3_MUSIC3_WEIGHT_COMPONENT_COUNT; ++index) {
        m3_weight_table_init(&built->tables[index]);
    }
    status = built->operations.create_backend(
        built->operations.context, &built->backend, error);
    if (status == M3_STATUS_OK) {
        status = m3_music3_validate_backend(built->backend, error);
    }
    if (status == M3_STATUS_OK) {
        m3_model_metadata_init(&metadata);
        status = built->operations.inspect_model(
            built->operations.context, trusted_model_root, &metadata,
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_load_tables(built, trusted_model_root, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_load_exact_tokenizer(
            built, trusted_model_root, error);
    }
    if (status != M3_STATUS_OK) {
        m3_music3_engine_dispose(built);
        return status;
    }
    *engine = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_music3_engine_open(m3_music3_engine **engine,
                                const char *trusted_model_root,
                                m3_error *error)
{
    return m3_music3_engine_open_core(
        engine, trusted_model_root, m3_music3_production_operations(),
        error);
}

void m3_music3_engine_free(m3_music3_engine *engine)
{
    m3_music3_engine_dispose(engine);
}

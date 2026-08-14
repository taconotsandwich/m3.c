/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "model_loader_fixture.h"
#include "music3_sparse_fixture.h"
#include "m3_model.h"

#include <stdint.h>
#include <string.h>

void m3_test_music3_sparse_directory(m3_test_context *test)
{
    char root[M3_TEST_PATH_CAPACITY];
    char component[M3_TEST_PATH_CAPACITY];
    char path[M3_TEST_PATH_CAPACITY];
    uint64_t logical_bytes = 0U;
    uint64_t physical_bytes = 0U;
    m3_model_metadata metadata;
    m3_model_metadata unchanged;
    m3_weight_table table;
    const m3_weight_binding *embedding;
    m3_error error;
    bool ready = m3_test_music3_create_sparse_layout(
        root, &logical_bytes, &physical_bytes);

    M3_TEST_EXPECT(test, ready, "create sparse official Music3 snapshot");
    if (!ready) {
        return;
    }
    m3_weight_table_init(&table);
    M3_TEST_EXPECT(test,
                   logical_bytes > UINT64_C(28505979820) &&
                       physical_bytes < UINT64_C(4) * 1024U * 1024U,
                   "represent official payload sizes with only sparse holes");
    m3_model_metadata_init(&metadata);
    M3_TEST_EXPECT(test,
                   m3_model_inspect_directory(root, &metadata, &error) ==
                       M3_STATUS_OK,
                   "inspect a complete exact official Music3 snapshot");
    M3_TEST_EXPECT(
        test,
        metadata.present_component_count == M3_COMPONENT_COUNT &&
            metadata.file_count == 20U && metadata.tensor_count == 1012U &&
            metadata.tensor_bytes == (size_t)UINT64_C(28505979820),
        "commit exact snapshot aggregates after schema validation");
    M3_TEST_EXPECT(
        test,
        metadata.components[M3_COMPONENT_LANGUAGE_MODEL].tensor_count ==
                399U &&
            metadata.components[M3_COMPONENT_TRANSFORMER].tensor_count ==
                441U &&
            metadata.components[M3_COMPONENT_VOCODER].tensor_count == 121U,
        "retain exact per-component inventories");
    M3_TEST_EXPECT(
        test,
        m3_music3_inspect_weight_table(
            root, M3_COMPONENT_LANGUAGE_MODEL, &table, &error) ==
            M3_STATUS_OK,
        "produce an owned official language-model weight table");
    embedding = m3_weight_table_find(&table, "model.embed_tokens.weight");
    M3_TEST_EXPECT(
        test,
        table.shard_count == 4U && table.binding_count == 399U &&
            table.aggregate_payload_bytes == UINT64_C(17168951296) &&
            strstr(table.shards[0].path,
                   "model-00001-of-00004.safetensors") != NULL &&
            table.shards[0].snapshot.regular_file &&
            table.shards[0].snapshot.file_size ==
                table.shards[0].data_section_offset +
                    table.shards[0].payload_bytes,
        "retain exact canonical shard provenance from the sparse snapshot");
    M3_TEST_EXPECT(
        test,
        embedding != NULL && embedding->shard_index == 0U &&
            embedding->data_start == 0U &&
            embedding->data_end <= table.shards[0].payload_bytes,
        "retain index-reconciled payload-relative official tensor ranges");
    {
        m3_weight_shard_record *owned_shards = table.shards;
        m3_weight_binding *owned_bindings = table.bindings;

        M3_TEST_EXPECT(
            test,
            m3_music3_inspect_weight_table(
                root, M3_COMPONENT_TOKENIZER, &table, &error) ==
                    M3_STATUS_INVALID_ARGUMENT &&
                table.shards == owned_shards &&
                table.bindings == owned_bindings,
            "resource-only components reject without replacing the table");
        M3_TEST_EXPECT(
            test,
            m3_music3_inspect_weight_table(
                root, M3_COMPONENT_SCHEDULER, &table, NULL) ==
                    M3_STATUS_INVALID_ARGUMENT &&
                table.shards == owned_shards &&
                table.bindings == owned_bindings,
            "resource-only rejection returns exact status without diagnostics");
    }
    m3_weight_table_dispose(&table);
    unchanged = metadata;
    M3_TEST_EXPECT(
        test,
        m3_loader_test_path(component, root, "vocoder") &&
            m3_loader_test_path(path, component, "legacy.pth") &&
            m3_loader_test_write_file(path, (const uint8_t *)"legacy", 6U),
        "add an unsupported component-local legacy weight");
    M3_TEST_EXPECT(test,
                   m3_model_inspect_directory(root, &metadata, &error) ==
                       M3_STATUS_UNSUPPORTED,
                   "reject component-local compatibility resources");
    M3_TEST_EXPECT(test,
                   memcmp(&metadata, &unchanged, sizeof(metadata)) == 0,
                   "failed exact snapshot inspection leaves output atomic");
    M3_TEST_EXPECT(test, m3_loader_test_remove_tree(root),
                   "remove sparse official Music3 snapshot");
}

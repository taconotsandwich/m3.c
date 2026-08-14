/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "model_loader_fixture.h"
#include "m3_file.h"
#include "m3_safetensors.h"
#include "m3_weights.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static bool m3_test_write_inspect_shard(
    const char *root, const char *name, const char *header,
    size_t payload_bytes, char path[M3_TEST_PATH_CAPACITY],
    m3_safetensors_metadata *metadata, m3_error *error)
{
    return m3_loader_test_path(path, root, name) &&
           m3_loader_test_write_safetensors(path, header, payload_bytes) &&
           m3_safetensors_inspect_file(path, metadata, error) == M3_STATUS_OK;
}

void m3_test_weight_bindings(m3_test_context *test)
{
    static const char z_header[] =
        "{\"zeta.weight\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[16,20]},"
        "\"alpha.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],"
        "\"data_offsets\":[0,16]}}";
    static const char a_header[] =
        "{\"beta.weight\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[0,4]}}";
    char root[M3_TEST_PATH_CAPACITY] = {0};
    char z_path[M3_TEST_PATH_CAPACITY];
    char a_path[M3_TEST_PATH_CAPACITY];
    m3_safetensors_metadata z_metadata;
    m3_safetensors_metadata a_metadata;
    m3_weight_shard_source sources[2];
    m3_weight_requirement exact[3];
    m3_weight_requirement duplicate[2];
    m3_weight_requirement missing;
    m3_weight_table table;
    const m3_weight_binding *alpha;
    const m3_weight_binding *zeta;
    m3_error error;
    bool ready;

    m3_safetensors_metadata_init(&z_metadata);
    m3_safetensors_metadata_init(&a_metadata);
    m3_weight_table_init(&table);
    ready = m3_loader_test_make_root(root) &&
            m3_test_write_inspect_shard(
                root, "z-shard.safetensors", z_header, 20U, z_path,
                &z_metadata, &error) &&
            m3_test_write_inspect_shard(
                root, "a-shard.safetensors", a_header, 4U, a_path,
                &a_metadata, &error);
    M3_TEST_EXPECT(test, ready, "create two inspected weight shards");
    if (!ready) {
        m3_safetensors_metadata_dispose(&z_metadata);
        m3_safetensors_metadata_dispose(&a_metadata);
        (void)m3_loader_test_remove_tree(root);
        return;
    }
    sources[0].path = z_path;
    sources[0].metadata = &z_metadata;
    sources[1].path = a_path;
    sources[1].metadata = &a_metadata;
    M3_TEST_EXPECT(test,
                   m3_weight_table_build(&table, sources, 2U, &error) ==
                       M3_STATUS_OK,
                   "build an owned provenance table from unsorted shards");
    M3_TEST_EXPECT(
        test,
        table.shard_count == 2U && table.binding_count == 3U &&
            table.aggregate_payload_bytes == 24U &&
            strcmp(table.shards[0].path, a_path) == 0 &&
            strcmp(table.shards[1].path, z_path) == 0 &&
            table.shards[0].path != sources[1].path &&
            table.shards[1].path != sources[0].path,
        "own one canonical lexicographic path copy per shard");
    alpha = m3_weight_table_find(&table, "alpha.weight");
    zeta = m3_weight_table_find(&table, "zeta.weight");
    M3_TEST_EXPECT(
        test,
        alpha != NULL && zeta != NULL && alpha->shard_index == 1U &&
            alpha->data_start == 0U && alpha->data_end == 16U &&
            zeta->shard_index == 1U && zeta->data_start == 16U &&
            zeta->data_end == 20U,
        "retain payload-relative binding ranges and remapped shard indexes");
    M3_TEST_EXPECT(
        test,
        m3_file_snapshot_equal(&table.shards[0].snapshot,
                               &a_metadata.source_snapshot) &&
            table.shards[0].data_section_offset ==
                a_metadata.data_section_offset &&
            table.shards[0].payload_bytes == 4U,
        "retain exact inspected shard provenance without payload bytes");
    exact[0].name = "alpha.weight";
    exact[0].tensor = z_metadata.tensors[0].tensor;
    exact[1].name = "beta.weight";
    exact[1].tensor = a_metadata.tensors[0].tensor;
    exact[2].name = "zeta.weight";
    exact[2].tensor = z_metadata.tensors[1].tensor;
    M3_TEST_EXPECT(test,
                   m3_weight_table_validate_required(&table, exact, 3U,
                                                     &error) == M3_STATUS_OK &&
                       m3_weight_table_validate_no_extra(&table, exact, 3U,
                                                         &error) ==
                           M3_STATUS_OK,
                   "exact requirements accept the provenance table");
    missing = exact[0];
    missing.name = "missing.weight";
    M3_TEST_EXPECT(test,
                   m3_weight_table_validate_required(&table, &missing, 1U,
                                                     &error) ==
                           M3_STATUS_INVALID_FORMAT &&
                       strstr(m3_error_message(&error), "missing required") !=
                           NULL,
                   "missing required weights retain explicit diagnostics");
    M3_TEST_EXPECT(test,
                   m3_weight_table_validate_no_extra(&table, exact, 2U,
                                                     &error) ==
                           M3_STATUS_INVALID_FORMAT &&
                       strstr(m3_error_message(&error), "unexpected extra") !=
                           NULL,
                   "unexpected weights retain explicit diagnostics");
    duplicate[0] = exact[0];
    duplicate[1] = exact[0];
    M3_TEST_EXPECT(test,
                   m3_weight_table_validate_required(&table, duplicate, 2U,
                                                     &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "duplicate requirements are rejected");
    exact[0].tensor.dtype = M3_DTYPE_F16;
    M3_TEST_EXPECT(test,
                   m3_weight_table_validate_required(&table, exact, 3U,
                                                     &error) ==
                       M3_STATUS_INVALID_FORMAT,
                   "weight requirements preserve exact dtype and shape");
    m3_weight_table_dispose(&table);
    m3_safetensors_metadata_dispose(&z_metadata);
    m3_safetensors_metadata_dispose(&a_metadata);
    M3_TEST_EXPECT(test, m3_loader_test_remove_tree(root),
                   "remove two-shard provenance fixture");
}

static bool m3_test_table_preserved(const m3_weight_table *table,
                                    const char *path,
                                    const char *binding_name)
{
    return table->shard_count == 1U && table->binding_count == 1U &&
           table->shards[0].path == path &&
           table->bindings[0].name == binding_name;
}

static void m3_test_set_synthetic_snapshot(m3_file_snapshot *snapshot,
                                           uint64_t device, uint64_t inode,
                                           uint64_t file_size)
{
    (void)memset(snapshot, 0, sizeof(*snapshot));
    snapshot->device = device;
    snapshot->inode = inode;
    snapshot->file_size = file_size;
    snapshot->regular_file = true;
}

static bool m3_test_capture_path(const char *path, m3_file_snapshot *snapshot,
                                 m3_error *error)
{
    int descriptor = open(path, O_RDONLY | O_NONBLOCK);
    bool success;

    if (descriptor < 0) {
        return false;
    }
    success = m3_file_snapshot_descriptor(descriptor, snapshot, error) ==
              M3_STATUS_OK;
    if (close(descriptor) != 0) {
        success = false;
    }
    return success;
}

void m3_test_weight_binding_rejections(m3_test_context *test)
{
    static const char header[] =
        "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2],"
        "\"data_offsets\":[0,8]}}";
    char root[M3_TEST_PATH_CAPACITY] = {0};
    char path[M3_TEST_PATH_CAPACITY];
    char other_path[M3_TEST_PATH_CAPACITY];
    char alias_path[M3_TEST_PATH_CAPACITY];
    char replacement_path[M3_TEST_PATH_CAPACITY];
    m3_safetensors_metadata metadata;
    m3_safetensors_metadata other_metadata;
    m3_safetensors_metadata alias_metadata;
    m3_safetensors_metadata bad_metadata;
    m3_safetensors_tensor bad_tensor;
    m3_weight_shard_source sources[2];
    m3_weight_table table;
    char *owned_path;
    char *owned_name;
    m3_error error;
    bool ready;

    m3_safetensors_metadata_init(&metadata);
    m3_safetensors_metadata_init(&other_metadata);
    m3_safetensors_metadata_init(&alias_metadata);
    m3_weight_table_init(&table);
    ready = m3_loader_test_make_root(root) &&
            m3_test_write_inspect_shard(root, "first.safetensors", header,
                                       8U, path, &metadata, &error) &&
            m3_test_write_inspect_shard(root, "second.safetensors", header,
                                       8U, other_path, &other_metadata,
                                       &error) &&
            m3_loader_test_path(alias_path, root, "alias.safetensors") &&
            link(path, alias_path) == 0 &&
            m3_safetensors_inspect_file(alias_path, &alias_metadata, &error) ==
                M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "create rejection provenance fixtures");
    if (!ready) {
        m3_safetensors_metadata_dispose(&metadata);
        m3_safetensors_metadata_dispose(&other_metadata);
        m3_safetensors_metadata_dispose(&alias_metadata);
        (void)m3_loader_test_remove_tree(root);
        return;
    }
    sources[0].path = path;
    sources[0].metadata = &metadata;
    M3_TEST_EXPECT(test,
                   m3_weight_table_build(&table, sources, 1U, &error) ==
                       M3_STATUS_OK,
                   "establish a table for atomic failure checks");
    owned_path = table.shards[0].path;
    owned_name = table.bindings[0].name;
    sources[1] = sources[0];
    M3_TEST_EXPECT(
        test,
        m3_weight_table_build(&table, sources, 2U, &error) ==
                M3_STATUS_INVALID_FORMAT &&
            strstr(m3_error_message(&error), "duplicate weight shard path") !=
                NULL &&
            m3_test_table_preserved(&table, owned_path, owned_name),
        "duplicate shard paths fail without replacing the old table");
    sources[1].path = alias_path;
    sources[1].metadata = &alias_metadata;
    M3_TEST_EXPECT(
        test,
        m3_weight_table_build(&table, sources, 2U, &error) ==
                M3_STATUS_INVALID_FORMAT &&
            strstr(m3_error_message(&error), "alias the same file") != NULL &&
            m3_test_table_preserved(&table, owned_path, owned_name),
        "hardlink shard aliases are rejected atomically");
    sources[1].path = other_path;
    sources[1].metadata = &other_metadata;
    M3_TEST_EXPECT(
        test,
        m3_weight_table_build(&table, sources, 2U, &error) ==
                M3_STATUS_INVALID_FORMAT &&
            strstr(m3_error_message(&error), "duplicate weight name") != NULL &&
            m3_test_table_preserved(&table, owned_path, owned_name),
        "duplicate tensor names across distinct shards are rejected");
    bad_metadata = metadata;
    bad_metadata.source_snapshot.file_size += 1U;
    sources[0].metadata = &bad_metadata;
    M3_TEST_EXPECT(
        test,
        m3_weight_table_build(&table, sources, 1U, &error) ==
                M3_STATUS_INVALID_FORMAT &&
            strstr(m3_error_message(&error), "payload total") != NULL &&
            m3_test_table_preserved(&table, owned_path, owned_name),
        "snapshot and payload mismatch preserves the old table");
    bad_metadata = metadata;
    bad_tensor = metadata.tensors[0];
    bad_metadata.tensors = &bad_tensor;
    bad_tensor.data_start = 9U;
    bad_tensor.data_end = 8U;
    sources[0].metadata = &bad_metadata;
    M3_TEST_EXPECT(
        test,
        m3_weight_table_build(&table, sources, 1U, &error) ==
                M3_STATUS_INVALID_FORMAT &&
            strstr(m3_error_message(&error), "reversed") != NULL &&
            m3_test_table_preserved(&table, owned_path, owned_name),
        "reversed payload ranges are rejected atomically");
    bad_tensor.data_start = 4U;
    bad_tensor.data_end = 12U;
    M3_TEST_EXPECT(
        test,
        m3_weight_table_build(&table, sources, 1U, &error) ==
                M3_STATUS_INVALID_FORMAT &&
            strstr(m3_error_message(&error), "exceeds") != NULL &&
            m3_test_table_preserved(&table, owned_path, owned_name),
        "out-of-payload ranges are rejected atomically");
    bad_tensor.data_start = 0U;
    bad_tensor.data_end = 7U;
    M3_TEST_EXPECT(
        test,
        m3_weight_table_build(&table, sources, 1U, &error) ==
                M3_STATUS_INVALID_FORMAT &&
            strstr(m3_error_message(&error), "range disagrees") != NULL &&
            m3_test_table_preserved(&table, owned_path, owned_name),
        "range and tensor byte mismatch is rejected atomically");
    M3_TEST_EXPECT(test,
                   m3_weight_table_build(&table, sources, 1U, NULL) ==
                           M3_STATUS_INVALID_FORMAT &&
                       m3_test_table_preserved(&table, owned_path, owned_name),
                   "weight failures return exact status without diagnostics");
    bad_metadata = metadata;
    bad_metadata.tensors = NULL;
    bad_metadata.tensor_count = 0U;
    bad_metadata.tensor_bytes = 0U;
    sources[0].metadata = &bad_metadata;
    M3_TEST_EXPECT(
        test,
        m3_weight_table_build(&table, sources, 1U, &error) ==
                M3_STATUS_INVALID_FORMAT &&
            strstr(m3_error_message(&error), "empty") != NULL &&
            m3_test_table_preserved(&table, owned_path, owned_name),
        "empty shards are rejected without replacing the old table");
    {
        const uint64_t large_payload = UINT64_MAX - UINT64_C(8191);
        const uint64_t large_shape[] = {large_payload / 4U};
        const uint64_t small_shape[] = {4096U};
        m3_safetensors_tensor synthetic_tensors[2];
        m3_safetensors_metadata synthetic_metadata[2];

        (void)memset(synthetic_tensors, 0, sizeof(synthetic_tensors));
        (void)memset(synthetic_metadata, 0, sizeof(synthetic_metadata));
        synthetic_tensors[0].name = "large";
        (void)m3_tensor_metadata_init(&synthetic_tensors[0].tensor,
                                      M3_DTYPE_F32, 1U, large_shape, &error);
        synthetic_tensors[0].data_end = large_payload;
        synthetic_metadata[0].tensors = &synthetic_tensors[0];
        synthetic_metadata[0].tensor_count = 1U;
        synthetic_metadata[0].tensor_bytes = (size_t)large_payload;
        synthetic_metadata[0].data_section_offset = 16U;
        m3_test_set_synthetic_snapshot(
            &synthetic_metadata[0].source_snapshot, 10U, 20U,
            large_payload + 16U);
        synthetic_tensors[1].name = "small";
        (void)m3_tensor_metadata_init(&synthetic_tensors[1].tensor,
                                      M3_DTYPE_F32, 1U, small_shape, &error);
        synthetic_tensors[1].data_end = 16384U;
        synthetic_metadata[1].tensors = &synthetic_tensors[1];
        synthetic_metadata[1].tensor_count = 1U;
        synthetic_metadata[1].tensor_bytes = 16384U;
        synthetic_metadata[1].data_section_offset = 16U;
        m3_test_set_synthetic_snapshot(
            &synthetic_metadata[1].source_snapshot, 11U, 21U, 16400U);
        sources[0].path = "large.safetensors";
        sources[0].metadata = &synthetic_metadata[0];
        sources[1].path = "small.safetensors";
        sources[1].metadata = &synthetic_metadata[1];
        M3_TEST_EXPECT(
            test,
            m3_weight_table_build(&table, sources, 2U, &error) ==
                    M3_STATUS_OVERFLOW &&
                strstr(m3_error_message(&error), "aggregate") != NULL &&
                m3_test_table_preserved(&table, owned_path, owned_name),
            "payload aggregate overflow preserves the old table");
    }
    M3_TEST_EXPECT(
        test,
        m3_loader_test_path(replacement_path, root, "replacement") &&
            m3_loader_test_write_safetensors(replacement_path, header, 8U) &&
            rename(replacement_path, path) == 0,
        "replace an inspected path with the same logical file size");
    {
        m3_file_snapshot replaced;
        bool captured = m3_test_capture_path(path, &replaced, &error);

        M3_TEST_EXPECT(
            test,
            captured && !m3_file_snapshot_equal(&table.shards[0].snapshot,
                                                &replaced),
            "stored identity detects same-size path replacement later");
    }
    m3_weight_table_dispose(&table);
    m3_safetensors_metadata_dispose(&metadata);
    m3_safetensors_metadata_dispose(&other_metadata);
    m3_safetensors_metadata_dispose(&alias_metadata);
    M3_TEST_EXPECT(test, m3_loader_test_remove_tree(root),
                   "remove rejection provenance fixtures");
}

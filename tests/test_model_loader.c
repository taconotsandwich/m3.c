/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "model_loader_fixture.h"
#include "m3_manifest.h"
#include "m3_model.h"
#include "m3_model_index.h"
#include "m3_safetensors.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

const char m3_loader_test_manifest[] =
    "{\"_class_name\":\"MiniMaxMusic3ModularPipeline\","
    "\"_blocks_class_name\":\"MiniMaxMusic3Blocks\","
    "\"_diffusers_version\":\"0.36.0.dev0\","
    "\"language_model\":[\"transformers\",\"Qwen3ForCausalLM\",{}],"
    "\"rvq_depth_decoder\":[\"diffusers\","
    "\"MiniMaxMusic3RVQDepthDecoder\",{}],"
    "\"condition_encoder\":[\"diffusers\","
    "\"MiniMaxMusic3ConditionEncoder\",{}],"
    "\"transformer\":[\"diffusers\","
    "\"MiniMaxMusic3Transformer1DModel\",{}],"
    "\"vocoder\":[\"diffusers\",\"MiniMaxMusic3Vocoder\",{}],"
    "\"tokenizer\":[\"transformers\",\"Qwen2Tokenizer\",{}],"
    "\"scheduler\":[\"diffusers\","
    "\"FlowMatchEulerDiscreteScheduler\",{}]}";

bool m3_loader_test_path(char path[M3_TEST_PATH_CAPACITY],
                         const char *directory, const char *name)
{
    int count = snprintf(path, M3_TEST_PATH_CAPACITY, "%s/%s", directory,
                         name);

    return count >= 0 && (size_t)count < M3_TEST_PATH_CAPACITY;
}

static bool m3_loader_test_write_all(int descriptor, const uint8_t *data,
                                     size_t size)
{
    size_t total = 0U;

    while (total < size) {
        ssize_t count = write(descriptor, data + total, size - total);

        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        total += (size_t)count;
    }
    return true;
}

bool m3_loader_test_write_file(const char *path, const uint8_t *data,
                               size_t size)
{
    int descriptor = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    bool success;

    if (descriptor < 0) {
        return false;
    }
    success = size == 0U || m3_loader_test_write_all(descriptor, data, size);
    if (close(descriptor) != 0) {
        success = false;
    }
    return success;
}

bool m3_loader_test_write_json(const char *path, const char *json)
{
    return m3_loader_test_write_file(path, (const uint8_t *)json,
                                     strlen(json));
}

bool m3_loader_test_write_safetensors(const char *path, const char *header,
                                      size_t payload_size)
{
    uint8_t length_bytes[8] = {0U};
    uint8_t payload[32] = {0U};
    uint64_t header_length = (uint64_t)strlen(header);
    unsigned int index;
    int descriptor;
    bool success;

    if (payload_size > sizeof(payload)) {
        return false;
    }
    for (index = 0U; index < 8U; ++index) {
        length_bytes[index] =
            (uint8_t)((header_length >> (index * 8U)) & 0xffU);
    }
    for (index = 0U; index < (unsigned int)payload_size; ++index) {
        payload[index] = (uint8_t)(index + 1U);
    }
    descriptor = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (descriptor < 0) {
        return false;
    }
    success = m3_loader_test_write_all(descriptor, length_bytes,
                                       sizeof(length_bytes)) &&
              m3_loader_test_write_all(
                  descriptor, (const uint8_t *)header, (size_t)header_length) &&
              m3_loader_test_write_all(descriptor, payload, payload_size);
    if (close(descriptor) != 0) {
        success = false;
    }
    return success;
}

bool m3_loader_test_make_root(char root[M3_TEST_PATH_CAPACITY])
{
    static const char pattern[] = "/tmp/m3-model-XXXXXX";
    int descriptor;

    (void)memcpy(root, pattern, sizeof(pattern));
    descriptor = mkstemp(root);
    if (descriptor < 0) {
        return false;
    }
    if (close(descriptor) != 0 || unlink(root) != 0 ||
        mkdir(root, 0700) != 0) {
        return false;
    }
    return true;
}

bool m3_loader_test_remove_tree(const char *path)
{
    DIR *directory;
    struct dirent *entry;
    bool success = true;

    if (path == NULL || strncmp(path, "/tmp/m3-model-", 14U) != 0) {
        return false;
    }
    directory = opendir(path);
    if (directory == NULL) {
        return unlink(path) == 0;
    }
    while ((entry = readdir(directory)) != NULL) {
        char child[M3_TEST_PATH_CAPACITY];
        struct stat child_stat;

        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (!m3_loader_test_path(child, path, entry->d_name) ||
            lstat(child, &child_stat) != 0) {
            success = false;
            continue;
        }
        if (S_ISDIR(child_stat.st_mode)) {
            if (!m3_loader_test_remove_tree(child)) {
                success = false;
            }
        } else if (unlink(child) != 0) {
            success = false;
        }
    }
    if (closedir(directory) != 0) {
        success = false;
    }
    if (rmdir(path) != 0) {
        success = false;
    }
    return success;
}

bool m3_loader_test_create_layout(char root[M3_TEST_PATH_CAPACITY])
{
    static const char tensor_header[] =
        "{\"__metadata__\":{\"format\":\"pt\"},"
        "\"weight\":{\"dtype\":\"F32\",\"shape\":[2],"
        "\"data_offsets\":[0,8]}}";
    char path[M3_TEST_PATH_CAPACITY];
    size_t component_index;

    if (!m3_loader_test_make_root(root) ||
        !m3_loader_test_path(path, root, "modular_model_index.json") ||
        !m3_loader_test_write_json(path, m3_loader_test_manifest)) {
        return false;
    }
    for (component_index = 0U; component_index < M3_COMPONENT_COUNT;
         ++component_index) {
        m3_component_id id = (m3_component_id)component_index;
        char component[M3_TEST_PATH_CAPACITY];

        if (!m3_loader_test_path(component, root,
                                 m3_component_directory(id)) ||
            mkdir(component, 0700) != 0) {
            return false;
        }
        if (m3_component_contains_weights(id)) {
            if (!m3_loader_test_path(path, component, "config.json") ||
                !m3_loader_test_write_json(path, "{}") ||
                !m3_loader_test_path(path, component, "model.safetensors") ||
                !m3_loader_test_write_safetensors(path, tensor_header, 8U)) {
                return false;
            }
        } else if (id == M3_COMPONENT_TOKENIZER) {
            if (!m3_loader_test_path(path, component, "tokenizer.json") ||
                !m3_loader_test_write_json(path, "{}") ||
                !m3_loader_test_path(path, component,
                                     "tokenizer_config.json") ||
                !m3_loader_test_write_json(path, "{}")) {
                return false;
            }
        } else if (!m3_loader_test_path(path, component,
                                        "scheduler_config.json") ||
                   !m3_loader_test_write_json(path, "{}")) {
            return false;
        }
    }
    return true;
}

void m3_test_safetensors_header(m3_test_context *test)
{
    static const char header[] =
        "{\"__metadata__\":{\"format\":\"pt\"},"
        "\"f32\":{\"dtype\":\"F32\",\"shape\":[2],"
        "\"data_offsets\":[0,8]},"
        "\"f16\":{\"dtype\":\"F16\",\"shape\":[],"
        "\"data_offsets\":[8,10]},"
        "\"bf16\":{\"dtype\":\"BF16\",\"shape\":[1,2],"
        "\"data_offsets\":[10,14]}}   ";
    char root[M3_TEST_PATH_CAPACITY] = {0};
    char path[M3_TEST_PATH_CAPACITY];
    char invalid_path[M3_TEST_PATH_CAPACITY];
    m3_safetensors_metadata metadata;
    m3_file_snapshot snapshot;
    m3_safetensors_tensor *owned_tensors;
    m3_error error;
    bool ready = m3_loader_test_make_root(root) &&
                 m3_loader_test_path(path, root, "weights.safetensors") &&
                 m3_loader_test_write_safetensors(path, header, 14U);

    M3_TEST_EXPECT(test, ready, "create supported Safetensors fixture");
    if (!ready) {
        return;
    }
    m3_safetensors_metadata_init(&metadata);
    M3_TEST_EXPECT(test,
                   m3_safetensors_inspect_file(path, &metadata, &error) ==
                       M3_STATUS_OK,
                   "inspect header without reading tensor payloads");
    M3_TEST_EXPECT(test, metadata.has_metadata,
                   "recognize reserved metadata object");
    M3_TEST_EXPECT(test,
                   metadata.tensor_count == 3U &&
                       metadata.tensor_bytes == 14U,
                   "aggregate supported tensor dtypes");
    M3_TEST_EXPECT(test,
                   metadata.data_section_offset ==
                       8U + (uint64_t)strlen(header),
                   "retain absolute Safetensors data-section offset");
    M3_TEST_EXPECT(
        test,
        metadata.source_snapshot.regular_file &&
            metadata.source_snapshot.file_size ==
                metadata.data_section_offset + 14U,
        "retain normalized source identity and exact file size");
    M3_TEST_EXPECT(test,
                   metadata.tensors[0].tensor.dtype == M3_DTYPE_F32 &&
                       metadata.tensors[1].tensor.dtype == M3_DTYPE_F16 &&
                       metadata.tensors[2].tensor.dtype == M3_DTYPE_BF16,
                   "retain sorted tensor metadata");
    snapshot = metadata.source_snapshot;
    owned_tensors = metadata.tensors;
    M3_TEST_EXPECT(
        test,
        m3_loader_test_path(invalid_path, root, "invalid.safetensors") &&
            m3_loader_test_write_safetensors(
                invalid_path,
                "{\"x\":{\"dtype\":\"I64\",\"shape\":[1],"
                "\"data_offsets\":[0,8]}}",
                8U) &&
            m3_safetensors_inspect_file(invalid_path, &metadata, &error) ==
                M3_STATUS_UNSUPPORTED &&
            metadata.tensors == owned_tensors && metadata.tensor_count == 3U &&
            m3_file_snapshot_equal(&metadata.source_snapshot, &snapshot),
        "failed inspection preserves prior owned metadata atomically");
    m3_safetensors_metadata_dispose(&metadata);
    M3_TEST_EXPECT(test, m3_loader_test_remove_tree(root),
                   "remove Safetensors fixture");
}

static void m3_loader_test_expect_invalid_header(m3_test_context *test,
                                                 const char *root,
                                                 const char *file_name,
                                                 const char *header,
                                                 size_t payload_size)
{
    char path[M3_TEST_PATH_CAPACITY];
    m3_safetensors_metadata metadata;
    m3_error error;
    bool written = m3_loader_test_path(path, root, file_name) &&
                   m3_loader_test_write_safetensors(path, header,
                                                    payload_size);

    M3_TEST_EXPECT(test, written, "write invalid Safetensors fixture");
    if (!written) {
        return;
    }
    m3_safetensors_metadata_init(&metadata);
    M3_TEST_EXPECT(test,
                   m3_safetensors_inspect_file(path, &metadata, &error) !=
                       M3_STATUS_OK,
                   "reject invalid Safetensors fixture");
    M3_TEST_EXPECT(test,
                   metadata.tensor_count == 0U && metadata.tensors == NULL,
                   "invalid header leaves inventory empty");
    m3_safetensors_metadata_dispose(&metadata);
}

void m3_test_safetensors_rejections(m3_test_context *test)
{
    static const char duplicate[] =
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[0,4]},\"x\":{\"dtype\":\"F32\","
        "\"shape\":[1],\"data_offsets\":[0,4]}}";
    static const char bad_dtype[] =
        "{\"x\":{\"dtype\":\"I64\",\"shape\":[1],"
        "\"data_offsets\":[0,8]}}";
    static const char bad_rank[] =
        "{\"x\":{\"dtype\":\"F16\","
        "\"shape\":[1,1,1,1,1,1,1,1,1],"
        "\"data_offsets\":[0,2]}}";
    static const char bad_size[] =
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],"
        "\"data_offsets\":[0,4]}}";
    static const char out_of_bounds[] =
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[0,8]}}";
    static const char bad_metadata[] = "{\"__metadata__\":{\"format\":1}}";
    static const char gap[] =
        "{\"x\":{\"dtype\":\"F16\",\"shape\":[1],"
        "\"data_offsets\":[1,3]}}";
    static const char leading_space[] =
        " {\"x\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[0,4]}}";
    static const char overflow_shape[] =
        "{\"x\":{\"dtype\":\"F32\","
        "\"shape\":[18446744073709551615,2],"
        "\"data_offsets\":[0,0]}}";
    uint8_t excessive_length[8] = {0xffU, 0xffU, 0xffU, 0xffU,
                                   0xffU, 0xffU, 0xffU, 0xffU};
    char root[M3_TEST_PATH_CAPACITY];
    char path[M3_TEST_PATH_CAPACITY];
    char fifo_path[M3_TEST_PATH_CAPACITY];
    char fifo_link[M3_TEST_PATH_CAPACITY];
    m3_safetensors_metadata metadata;
    m3_error error;
    bool ready = m3_loader_test_make_root(root);

    M3_TEST_EXPECT(test, ready, "create invalid-header fixture directory");
    if (!ready) {
        return;
    }
    m3_loader_test_expect_invalid_header(test, root, "duplicate.safetensors",
                                         duplicate, 4U);
    m3_loader_test_expect_invalid_header(test, root, "dtype.safetensors",
                                         bad_dtype, 8U);
    m3_loader_test_expect_invalid_header(test, root, "rank.safetensors",
                                         bad_rank, 2U);
    m3_loader_test_expect_invalid_header(test, root, "size.safetensors",
                                         bad_size, 4U);
    m3_loader_test_expect_invalid_header(test, root, "bounds.safetensors",
                                         out_of_bounds, 4U);
    m3_loader_test_expect_invalid_header(test, root, "metadata.safetensors",
                                         bad_metadata, 0U);
    m3_loader_test_expect_invalid_header(test, root, "gap.safetensors", gap,
                                         3U);
    m3_loader_test_expect_invalid_header(test, root, "leading.safetensors",
                                         leading_space, 4U);
    ready = m3_loader_test_path(path, root, "overflow.safetensors") &&
            m3_loader_test_write_safetensors(path, overflow_shape, 0U);
    M3_TEST_EXPECT(test, ready, "write overflowing shape fixture");
    m3_safetensors_metadata_init(&metadata);
    M3_TEST_EXPECT(test,
                   ready && m3_safetensors_inspect_file(
                                path, &metadata, NULL) == M3_STATUS_OVERFLOW,
                   "preserve overflow status without diagnostic output");
    m3_safetensors_metadata_dispose(&metadata);
    ready = m3_loader_test_path(path, root, "length.safetensors") &&
            m3_loader_test_write_file(path, excessive_length,
                                      sizeof(excessive_length));
    M3_TEST_EXPECT(test, ready, "write excessive header length fixture");
    m3_safetensors_metadata_init(&metadata);
    M3_TEST_EXPECT(test,
                   ready && m3_safetensors_inspect_file(
                                path, &metadata, &error) ==
                                M3_STATUS_INVALID_FORMAT,
                   "reject excessive little-endian header length");
    m3_safetensors_metadata_dispose(&metadata);
    ready = m3_loader_test_path(fifo_path, root, "weights.fifo") &&
            mkfifo(fifo_path, 0600) == 0 &&
            m3_loader_test_path(fifo_link, root, "fifo.safetensors") &&
            symlink(fifo_path, fifo_link) == 0;
    M3_TEST_EXPECT(test, ready, "create symlink-to-FIFO weight fixture");
    m3_safetensors_metadata_init(&metadata);
    M3_TEST_EXPECT(test,
                   ready && m3_safetensors_inspect_file(
                                fifo_link, &metadata, &error) ==
                                M3_STATUS_INVALID_FORMAT,
                   "reject symlink-to-FIFO without blocking");
    m3_safetensors_metadata_dispose(&metadata);
    M3_TEST_EXPECT(test, m3_loader_test_remove_tree(root),
                   "remove invalid-header fixtures");
}

void m3_test_modular_model_directory(m3_test_context *test)
{
    char root[M3_TEST_PATH_CAPACITY];
    char component[M3_TEST_PATH_CAPACITY];
    char path[M3_TEST_PATH_CAPACITY];
    char fifo_path[M3_TEST_PATH_CAPACITY];
    m3_model_metadata metadata;
    m3_model_metadata empty;
    m3_error error;
    bool ready = m3_loader_test_create_layout(root);

    M3_TEST_EXPECT(test, ready, "create legacy tiny modular fixture");
    if (!ready) {
        return;
    }
    m3_model_metadata_init(&metadata);
    empty = metadata;
    M3_TEST_EXPECT(test,
                   m3_model_inspect_directory(root, &metadata, &error) ==
                       M3_STATUS_INVALID_FORMAT,
                   "reject arbitrary configs and tiny tensor inventories");
    M3_TEST_EXPECT(test, memcmp(&metadata, &empty, sizeof(metadata)) == 0,
                   "schema rejection leaves output atomic");
    M3_TEST_EXPECT(test,
                   m3_model_inspect_directory("/missing/m3-model", &metadata,
                                              &error) == M3_STATUS_IO,
                   "reject missing model directory");
    M3_TEST_EXPECT(test,
                   memcmp(&metadata, &empty, sizeof(metadata)) == 0,
                   "failed inspection leaves output atomic");
    M3_TEST_EXPECT(test,
                   m3_loader_test_path(component, root, "language_model") &&
                       m3_loader_test_path(path, component, "config.json") &&
                       unlink(path) == 0 &&
                       m3_loader_test_path(fifo_path, component,
                                           "config.fifo") &&
                       mkfifo(fifo_path, 0600) == 0 &&
                       symlink(fifo_path, path) == 0,
                   "replace config with a symlink to FIFO");
    M3_TEST_EXPECT(test,
                   m3_model_inspect_directory(root, &metadata, &error) ==
                       M3_STATUS_INVALID_FORMAT,
                   "reject FIFO config without blocking");
    M3_TEST_EXPECT(test, memcmp(&metadata, &empty, sizeof(metadata)) == 0,
                   "FIFO rejection leaves output atomic");
    M3_TEST_EXPECT(test, m3_loader_test_remove_tree(root),
                   "remove modular model fixture");
}

void m3_test_modular_manifest(m3_test_context *test)
{
    static const char missing_components[] =
        "{\"_class_name\":\"MiniMaxMusic3ModularPipeline\","
        "\"_blocks_class_name\":\"MiniMaxMusic3Blocks\"}";
    static const char wrong_class[] =
        "{\"_class_name\":\"OtherPipeline\"}";
    static const char duplicate_class[] =
        "{\"_class_name\":\"MiniMaxMusic3ModularPipeline\","
        "\"_class_name\":\"MiniMaxMusic3ModularPipeline\"}";
    static const char wrong_component_type[] =
        "{\"_class_name\":\"MiniMaxMusic3ModularPipeline\","
        "\"_blocks_class_name\":\"MiniMaxMusic3Blocks\","
        "\"language_model\":{}}";
    char wrong_identity[sizeof(m3_loader_test_manifest)];
    char root[M3_TEST_PATH_CAPACITY];
    char path[M3_TEST_PATH_CAPACITY];
    char *identity_class;
    m3_error error;
    bool ready = m3_loader_test_make_root(root) &&
                 m3_loader_test_path(path, root,
                                     "modular_model_index.json");

    M3_TEST_EXPECT(test, ready, "create modular manifest fixture");
    if (!ready) {
        return;
    }
    (void)memcpy(wrong_identity, m3_loader_test_manifest,
                 sizeof(wrong_identity));
    identity_class = strstr(wrong_identity, "Qwen3ForCausalLM");
    M3_TEST_EXPECT(test, identity_class != NULL,
                   "locate language model identity fixture");
    if (identity_class == NULL) {
        (void)m3_loader_test_remove_tree(root);
        return;
    }
    identity_class[0] = 'X';
    M3_TEST_EXPECT(test,
                   m3_loader_test_write_json(path, m3_loader_test_manifest) &&
                       m3_manifest_validate_file(path, &error) == M3_STATUS_OK,
                   "accept official pipeline classes and components");
    M3_TEST_EXPECT(test,
                   m3_loader_test_write_json(path, missing_components) &&
                       m3_manifest_validate_file(path, &error) ==
                           M3_STATUS_INVALID_FORMAT,
                   "reject missing component declarations");
    M3_TEST_EXPECT(test,
                   m3_loader_test_write_json(path, wrong_class) &&
                       m3_manifest_validate_file(path, &error) ==
                           M3_STATUS_INVALID_FORMAT,
                   "reject wrong pipeline class");
    M3_TEST_EXPECT(test,
                   m3_loader_test_write_json(path, duplicate_class) &&
                       m3_manifest_validate_file(path, &error) ==
                           M3_STATUS_INVALID_FORMAT,
                   "reject duplicate required manifest field");
    M3_TEST_EXPECT(test,
                   m3_loader_test_write_json(path, wrong_component_type) &&
                       m3_manifest_validate_file(path, &error) ==
                           M3_STATUS_INVALID_FORMAT,
                   "reject wrong component declaration type");
    M3_TEST_EXPECT(test,
                   m3_loader_test_write_json(path, wrong_identity) &&
                       m3_manifest_validate_file(path, &error) ==
                           M3_STATUS_INVALID_FORMAT,
                   "reject wrong component library or class identity");
    M3_TEST_EXPECT(test, m3_loader_test_remove_tree(root),
                   "remove modular manifest fixture");
}

void m3_test_modular_model_index(m3_test_context *test)
{
    static const char valid_index[] =
        "{\"metadata\":{\"total_size\":6},\"weight_map\":{"
        "\"a\":\"model-00001-of-00002.safetensors\","
        "\"b\":\"model-00002-of-00002.safetensors\"}}";
    static const char invalid_index[] =
        "{\"metadata\":{\"total_size\":6},\"weight_map\":{"
        "\"a\":\"model-00001-of-00002.safetensors\","
        "\"a\":\"model-00002-of-00002.safetensors\"}}";
    char root[M3_TEST_PATH_CAPACITY];
    char index_path[M3_TEST_PATH_CAPACITY];
    m3_model_index index;
    m3_error error;
    bool ready = m3_loader_test_make_root(root) &&
                 m3_loader_test_path(index_path, root, "model.index.json") &&
                 m3_loader_test_write_json(index_path, valid_index);

    M3_TEST_EXPECT(test, ready, "create model index fixture");
    if (!ready) {
        return;
    }
    m3_model_index_init(&index);
    M3_TEST_EXPECT(test,
                   m3_model_index_read(index_path, &index, &error) ==
                       M3_STATUS_OK,
                   "parse a bounded sharded weight map");
    M3_TEST_EXPECT(test,
                   index.entry_count == 2U && index.has_total_size &&
                       index.total_size == 6U &&
                       strcmp(m3_model_index_shard(&index, "b"),
                              "model-00002-of-00002.safetensors") == 0,
                   "retain exact tensor-to-shard mappings");
    M3_TEST_EXPECT(test, m3_loader_test_write_json(index_path, invalid_index),
                   "replace index with a duplicate key");
    M3_TEST_EXPECT(test,
                   m3_model_index_read(index_path, &index, &error) ==
                       M3_STATUS_INVALID_FORMAT,
                   "reject duplicate tensor keys in model index");
    M3_TEST_EXPECT(test, index.entry_count == 2U,
                   "failed index replacement leaves output atomic");
    m3_model_index_dispose(&index);
    M3_TEST_EXPECT(test, m3_loader_test_remove_tree(root),
                   "remove indexed shard fixture");
}

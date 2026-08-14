/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "model_loader_fixture.h"
#include "m3_model.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static bool m3_manifest_test_write_shard(const char *directory,
                                         const char *file_name,
                                         const char *tensor_name)
{
    char header[256];
    char path[M3_TEST_PATH_CAPACITY];
    int count = snprintf(
        header, sizeof(header),
        "{\"%s\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[0,4]}}",
        tensor_name);

    return count >= 0 && (size_t)count < sizeof(header) &&
           m3_loader_test_path(path, directory, file_name) &&
           m3_loader_test_write_safetensors(path, header, 4U);
}

static bool m3_manifest_test_replace_weight(const char *root,
                                            const char *component_name,
                                            const char *new_name)
{
    char component[M3_TEST_PATH_CAPACITY];
    char old_path[M3_TEST_PATH_CAPACITY];
    char new_path[M3_TEST_PATH_CAPACITY];

    return m3_loader_test_path(component, root, component_name) &&
           m3_loader_test_path(old_path, component, "model.safetensors") &&
           m3_loader_test_path(new_path, component, new_name) &&
           rename(old_path, new_path) == 0;
}

static bool m3_manifest_test_add_root_resources(const char *root)
{
    static const char *const files[] = {
        ".gitattributes", "LICENSE", "README.md", "config.json",
        "dav.pth", "flowmatching_vae.pth"
    };
    static const char *const directories[] = {
        "assets", "figures", "scripts", "qwen_7B"
    };
    static const uint8_t contents[] = {'x'};
    char path[M3_TEST_PATH_CAPACITY];
    size_t index;

    for (index = 0U; index < sizeof(files) / sizeof(files[0]); ++index) {
        if (!m3_loader_test_path(path, root, files[index]) ||
            !m3_loader_test_write_file(path, contents, sizeof(contents))) {
            return false;
        }
    }
    for (index = 0U;
         index < sizeof(directories) / sizeof(directories[0]); ++index) {
        if (!m3_loader_test_path(path, root, directories[index]) ||
            mkdir(path, 0700) != 0) {
            return false;
        }
    }
    return true;
}

static bool m3_manifest_test_prepare_language_model(const char *root)
{
    static const char index_json[] =
        "{\"metadata\":{\"total_size\":16},\"weight_map\":{"
        "\"lm0\":\"model-00001-of-00004.safetensors\","
        "\"lm1\":\"model-00002-of-00004.safetensors\","
        "\"lm2\":\"model-00003-of-00004.safetensors\","
        "\"lm3\":\"model-00004-of-00004.safetensors\"}}";
    static const char *const shard_names[] = {
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors"
    };
    static const char *const tensor_names[] = {"lm0", "lm1", "lm2", "lm3"};
    char component[M3_TEST_PATH_CAPACITY];
    char path[M3_TEST_PATH_CAPACITY];
    size_t index;

    if (!m3_loader_test_path(component, root, "language_model") ||
        !m3_loader_test_path(path, component, "model.safetensors") ||
        unlink(path) != 0 ||
        !m3_loader_test_path(path, component, "generation_config.json") ||
        !m3_loader_test_write_json(path, "{}")) {
        return false;
    }
    for (index = 0U; index < 4U; ++index) {
        if (!m3_manifest_test_write_shard(component, shard_names[index],
                                          tensor_names[index])) {
            return false;
        }
    }
    return m3_loader_test_path(path, component,
                               "model.safetensors.index.json") &&
           m3_loader_test_write_json(path, index_json);
}

static bool m3_manifest_test_prepare_transformer(const char *root)
{
    static const char index_json[] =
        "{\"metadata\":{\"total_size\":8},\"weight_map\":{"
        "\"dit0\":\"diffusion_pytorch_model-00001-of-00002.safetensors\","
        "\"dit1\":\"diffusion_pytorch_model-00002-of-00002.safetensors\"}}";
    char component[M3_TEST_PATH_CAPACITY];
    char path[M3_TEST_PATH_CAPACITY];

    return m3_loader_test_path(component, root, "transformer") &&
           m3_loader_test_path(path, component, "model.safetensors") &&
           unlink(path) == 0 &&
           m3_manifest_test_write_shard(
               component,
               "diffusion_pytorch_model-00001-of-00002.safetensors",
               "dit0") &&
           m3_manifest_test_write_shard(
               component,
               "diffusion_pytorch_model-00002-of-00002.safetensors",
               "dit1") &&
           m3_loader_test_path(
               path, component,
               "diffusion_pytorch_model.safetensors.index.json") &&
           m3_loader_test_write_json(path, index_json);
}

void m3_test_official_snapshot_manifest(m3_test_context *test)
{
    char root[M3_TEST_PATH_CAPACITY];
    char tokenizer[M3_TEST_PATH_CAPACITY];
    char template_path[M3_TEST_PATH_CAPACITY];
    m3_model_metadata metadata;
    m3_error error;
    bool ready = m3_loader_test_create_layout(root) &&
                 m3_manifest_test_prepare_language_model(root) &&
                 m3_manifest_test_prepare_transformer(root) &&
                 m3_manifest_test_replace_weight(
                     root, "rvq_depth_decoder",
                     "diffusion_pytorch_model.safetensors") &&
                 m3_manifest_test_replace_weight(
                     root, "condition_encoder",
                     "diffusion_pytorch_model.safetensors") &&
                 m3_manifest_test_replace_weight(
                     root, "vocoder",
                     "diffusion_pytorch_model.safetensors") &&
                 m3_loader_test_path(tokenizer, root, "tokenizer") &&
                 m3_loader_test_path(template_path, tokenizer,
                                     "chat_template.jinja") &&
                 m3_loader_test_write_file(
                     template_path, (const uint8_t *)"{% generation %}",
                     strlen("{% generation %}")) &&
                 m3_manifest_test_add_root_resources(root);

    M3_TEST_EXPECT(test, ready, "create current official manifest fixture");
    if (!ready) {
        return;
    }
    m3_model_metadata_init(&metadata);
    M3_TEST_EXPECT(test,
                   m3_model_inspect_directory(root, &metadata, &error) ==
                       M3_STATUS_OK,
                   "inspect only the modular projection of official snapshot");
    M3_TEST_EXPECT(test,
                   metadata.present_component_count == M3_COMPONENT_COUNT &&
                       metadata.file_count == 21U &&
                       metadata.tensor_count == 9U &&
                       metadata.tensor_bytes == 48U,
                   "match current official modular filename inventory");
    M3_TEST_EXPECT(test,
                   metadata.components[M3_COMPONENT_LANGUAGE_MODEL]
                               .file_count == 7U &&
                       metadata.components[M3_COMPONENT_TRANSFORMER]
                               .file_count == 4U &&
                       metadata.components[M3_COMPONENT_TOKENIZER]
                               .file_count == 3U,
                   "exclude ignored root assets and legacy packages");
    M3_TEST_EXPECT(test, m3_loader_test_remove_tree(root),
                   "remove official manifest fixture");
}

/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "model_loader_fixture.h"
#include "m3_music3_config.h"

#include <stdio.h>
#include <string.h>

#define M3_CONFIG_FIXTURE_CAPACITY 8192U

static bool m3_config_append(char *json, size_t capacity, size_t *length,
                             const char *text)
{
    size_t text_length = strlen(text);

    if (text_length >= capacity - *length) {
        return false;
    }
    (void)memcpy(json + *length, text, text_length + 1U);
    *length += text_length;
    return true;
}

static bool m3_config_build_lm(char json[M3_CONFIG_FIXTURE_CAPACITY],
                               bool equivalent_decimals)
{
    const char *dropout = equivalent_decimals ? "0" : "0.0";
    const char *initializer = equivalent_decimals ? "2e-2" : "0.02";
    const char *epsilon = equivalent_decimals ? "0.000001" : "1e-06";
    size_t length = 0U;
    unsigned int layer;
    int count = snprintf(
        json, M3_CONFIG_FIXTURE_CAPACITY,
        "{\"architectures\":[\"Qwen3ForCausalLM\"],"
        "\"attention_bias\":false,\"attention_dropout\":%s,"
        "\"bos_token_id\":null,\"dtype\":\"bfloat16\","
        "\"eos_token_id\":null,\"head_dim\":128,"
        "\"hidden_act\":\"silu\",\"hidden_size\":4096,"
        "\"initializer_range\":%s,\"intermediate_size\":12288,"
        "\"layer_types\":[",
        dropout, initializer);

    if (count < 0 || (size_t)count >= M3_CONFIG_FIXTURE_CAPACITY) {
        return false;
    }
    length = (size_t)count;
    for (layer = 0U; layer < 36U; ++layer) {
        if (!m3_config_append(json, M3_CONFIG_FIXTURE_CAPACITY, &length,
                              layer == 0U ? "\"full_attention\""
                                          : ",\"full_attention\"")) {
            return false;
        }
    }
    if (!m3_config_append(
            json, M3_CONFIG_FIXTURE_CAPACITY, &length,
            "],\"max_position_embeddings\":10240,"
            "\"max_window_layers\":28,\"model_type\":\"qwen3\","
            "\"num_attention_heads\":32,\"num_hidden_layers\":36,"
            "\"num_key_value_heads\":8,\"pad_token_id\":null,"
            "\"rms_norm_eps\":")) {
        return false;
    }
    if (!m3_config_append(json, M3_CONFIG_FIXTURE_CAPACITY, &length,
                          epsilon) ||
        !m3_config_append(
            json, M3_CONFIG_FIXTURE_CAPACITY, &length,
            ",\"rope_parameters\":{\"rope_theta\":1000000,"
            "\"rope_type\":\"default\"},\"sliding_window\":null,"
            "\"tie_word_embeddings\":false,"
            "\"transformers_version\":\"5.13.0.dev0\","
            "\"use_cache\":true,\"use_sliding_window\":false,"
            "\"vocab_size\":200000}")) {
        return false;
    }
    return true;
}

static const char *m3_config_fixture_for(m3_component_id id)
{
    static const char rvq[] =
        "{\"_class_name\":\"MiniMaxMusic3RVQDepthDecoder\","
        "\"_diffusers_version\":\"0.40.0.dev0\","
        "\"audio_vocab_size\":1024,\"hidden_size\":4096,"
        "\"intermediate_size\":6144,\"max_position_embeddings\":16,"
        "\"num_attention_heads\":16,\"num_codebooks\":8,"
        "\"num_layers\":4}";
    static const char condition[] =
        "{\"_class_name\":\"MiniMaxMusic3ConditionEncoder\","
        "\"_diffusers_version\":\"0.40.0.dev0\","
        "\"condition_hidden_dim\":4096,\"input_hop_length\":960,"
        "\"input_sampling_rate\":24000,\"num_condition_layers\":8,"
        "\"out_dim\":2048,\"output_hop_length\":512,"
        "\"output_sampling_rate\":44100}";
    static const char flow[] =
        "{\"_class_name\":\"MiniMaxMusic3Transformer1DModel\","
        "\"_diffusers_version\":\"0.40.0.dev0\","
        "\"attention_head_dim\":64,\"condition_dim\":2048,"
        "\"ff_inner_dim\":8192,\"fourier_embedding_dim\":256,"
        "\"in_channels\":128,\"num_attention_heads\":32,"
        "\"num_layers\":36,\"rotary_dim\":32}";
    static const char vocoder[] =
        "{\"_class_name\":\"MiniMaxMusic3Vocoder\","
        "\"_diffusers_version\":\"0.40.0.dev0\","
        "\"decoder_hidden_dim\":1536,\"decoder_input_dim\":1024,"
        "\"latent_channels\":128,\"sampling_rate\":44100,"
        "\"upsampling_ratios\":[8,8,4,2]}";

    switch (id) {
    case M3_COMPONENT_RVQ_DEPTH_DECODER:
        return rvq;
    case M3_COMPONENT_CONDITION_ENCODER:
        return condition;
    case M3_COMPONENT_TRANSFORMER:
        return flow;
    case M3_COMPONENT_VOCODER:
        return vocoder;
    default:
        return NULL;
    }
}

bool m3_test_music3_write_config(const char *path, m3_component_id id)
{
    char lm[M3_CONFIG_FIXTURE_CAPACITY];
    const char *json = m3_config_fixture_for(id);

    if (id == M3_COMPONENT_LANGUAGE_MODEL) {
        if (!m3_config_build_lm(lm, false)) {
            return false;
        }
        json = lm;
    }
    return json != NULL && m3_loader_test_write_json(path, json);
}

static bool m3_config_fixture_rejected(const char *path, m3_component_id id,
                                       const char *json)
{
    m3_music3_component_config config;
    m3_error error;

    m3_music3_component_config_init(&config);
    return m3_loader_test_write_json(path, json) &&
           m3_music3_config_read_file(path, id, &config, &error) ==
               M3_STATUS_INVALID_FORMAT;
}

void m3_test_music3_configs(m3_test_context *test)
{
    char root[M3_TEST_PATH_CAPACITY];
    char path[M3_TEST_PATH_CAPACITY];
    char json[M3_CONFIG_FIXTURE_CAPACITY];
    m3_music3_component_config configs[M3_COMPONENT_COUNT];
    m3_error error;
    m3_component_id id;
    bool ready = m3_loader_test_make_root(root) &&
                 m3_loader_test_path(path, root, "config.json");

    M3_TEST_EXPECT(test, ready, "create Music3 config fixture");
    if (!ready) {
        return;
    }
    for (id = M3_COMPONENT_LANGUAGE_MODEL; id < M3_COMPONENT_COUNT;
         id = (m3_component_id)((int)id + 1)) {
        m3_music3_component_config_init(&configs[(size_t)id]);
    }
    for (id = M3_COMPONENT_LANGUAGE_MODEL; id <= M3_COMPONENT_VOCODER;
         id = (m3_component_id)((int)id + 1)) {
        M3_TEST_EXPECT(
            test,
            m3_test_music3_write_config(path, id) &&
                m3_music3_config_read_file(
                    path, id, &configs[(size_t)id], &error) == M3_STATUS_OK,
            "accept exact official component config");
    }
    M3_TEST_EXPECT(test,
                   m3_music3_config_validate_cross(configs, &error) ==
                       M3_STATUS_OK,
                   "validate all cross-component dimensions");
    M3_TEST_EXPECT(
        test,
        m3_config_build_lm(json, true) &&
            m3_loader_test_write_json(path, json) &&
            m3_music3_config_read_file(
                path, M3_COMPONENT_LANGUAGE_MODEL,
                &configs[M3_COMPONENT_LANGUAGE_MODEL], &error) ==
                M3_STATUS_OK,
        "accept locale-independent equivalent JSON decimals");
    if (m3_config_build_lm(json, false)) {
        char *field = strstr(json, "\"hidden_size\":4096");
        m3_music3_component_config unchanged =
            configs[M3_COMPONENT_LANGUAGE_MODEL];

        if (field != NULL) {
            field[strlen("\"hidden_size\":409")] = '5';
        }
        M3_TEST_EXPECT(
            test,
            field != NULL && m3_loader_test_write_json(path, json) &&
                m3_music3_config_read_file(
                    path, M3_COMPONENT_LANGUAGE_MODEL,
                    &configs[M3_COMPONENT_LANGUAGE_MODEL], &error) ==
                    M3_STATUS_INVALID_FORMAT,
            "reject changed language-model dimensions");
        M3_TEST_EXPECT(
            test,
            memcmp(&unchanged, &configs[M3_COMPONENT_LANGUAGE_MODEL],
                   sizeof(unchanged)) == 0,
            "failed config validation leaves output atomic");
    }
    if (m3_config_build_lm(json, false)) {
        size_t length = strlen(json);

        json[length - 1U] = '\0';
        ready = m3_config_append(
            json, sizeof(json), &length,
            ",\"torch_dtype\":\"bfloat16\"}");
        M3_TEST_EXPECT(
            test,
            ready && m3_loader_test_write_json(path, json) &&
                m3_music3_config_read_file(
                    path, M3_COMPONENT_LANGUAGE_MODEL,
                    &configs[M3_COMPONENT_LANGUAGE_MODEL], &error) ==
                    M3_STATUS_INVALID_FORMAT,
            "reject legacy top-level dtype compatibility");
    }
    if (m3_config_build_lm(json, false)) {
        char *field = strstr(json, "1e-06");

        if (field != NULL) {
            (void)memcpy(field, "1e--6", 5U);
        }
        M3_TEST_EXPECT(test,
                       field != NULL && m3_config_fixture_rejected(
                                            path,
                                            M3_COMPONENT_LANGUAGE_MODEL,
                                            json),
                       "reject malformed architectural decimals");
    }
    if (m3_config_build_lm(json, false)) {
        char *field = strstr(json, "\"rope_type\":\"default\"");

        if (field != NULL) {
            field[strlen("\"rope_type\":\"")] = 'x';
        }
        M3_TEST_EXPECT(test,
                       field != NULL && m3_config_fixture_rejected(
                                            path,
                                            M3_COMPONENT_LANGUAGE_MODEL,
                                            json),
                       "reject alternate RoPE semantics");
    }
    {
        const char *rvq = m3_config_fixture_for(
            M3_COMPONENT_RVQ_DEPTH_DECODER);
        char *field;

        (void)snprintf(json, sizeof(json), "%s", rvq);
        field = strstr(json, "MiniMaxMusic3RVQDepthDecoder");
        if (field != NULL) {
            field[0] = 'X';
        }
        M3_TEST_EXPECT(test,
                       field != NULL && m3_config_fixture_rejected(
                                            path,
                                            M3_COMPONENT_RVQ_DEPTH_DECODER,
                                            json),
                       "reject a wrong component class identity");
    }
    {
        const char *condition = m3_config_fixture_for(
            M3_COMPONENT_CONDITION_ENCODER);
        char *field;

        (void)snprintf(json, sizeof(json), "%s", condition);
        field = strstr(json, "\"condition_hidden_dim\":4096");
        if (field != NULL) {
            (void)memcpy(field + strlen("\"condition_hidden_dim\":"),
                         "null", 4U);
        }
        M3_TEST_EXPECT(test,
                       field != NULL && m3_config_fixture_rejected(
                                            path,
                                            M3_COMPONENT_CONDITION_ENCODER,
                                            json),
                       "reject a wrong architectural value type");
    }
    {
        const char *vocoder = m3_config_fixture_for(M3_COMPONENT_VOCODER);
        char *field;

        (void)snprintf(json, sizeof(json), "%s", vocoder);
        field = strstr(json, "[8,8,4,2]");
        if (field != NULL) {
            field[strlen("[8,8,4,")] = '3';
        }
        M3_TEST_EXPECT(test,
                       field != NULL && m3_config_fixture_rejected(
                                            path, M3_COMPONENT_VOCODER, json),
                       "reject changed vocoder ratio arrays");
        (void)snprintf(json, sizeof(json), "%s", vocoder);
        {
            size_t length = strlen(json);

            json[length - 1U] = '\0';
            ready = m3_config_append(
                json, sizeof(json), &length,
                ",\"sampling_rate\":44100}");
        }
        M3_TEST_EXPECT(test,
                       ready && m3_config_fixture_rejected(
                                    path, M3_COMPONENT_VOCODER, json),
                       "reject duplicate architectural fields");
    }
    configs[M3_COMPONENT_CONDITION_ENCODER].output_hop_length += 1U;
    M3_TEST_EXPECT(test,
                   m3_music3_config_validate_cross(configs, &error) ==
                       M3_STATUS_INVALID_FORMAT,
                   "reject inconsistent cross-component dimensions");
    M3_TEST_EXPECT(test, m3_loader_test_remove_tree(root),
                   "remove Music3 config fixture");
}

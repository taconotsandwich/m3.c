/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_config.h"

#include "m3_file.h"
#include "m3_json.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#define M3_MUSIC3_CONFIG_MAX_BYTES (4U * 1024U * 1024U)

typedef enum {
    M3_CONFIG_UINT = 0,
    M3_CONFIG_DECIMAL,
    M3_CONFIG_STRING,
    M3_CONFIG_NONEMPTY_STRING,
    M3_CONFIG_BOOL,
    M3_CONFIG_NULL,
    M3_CONFIG_REPEAT_STRING_ARRAY,
    M3_CONFIG_UINT_ARRAY,
    M3_CONFIG_ROPE
} m3_config_kind;

typedef struct {
    const char *name;
    m3_config_kind kind;
    uint64_t number;
    int exponent;
    const char *text;
    const uint16_t *array;
    size_t count;
    bool optional;
} m3_config_field;

static const uint16_t m3_vocoder_ratios[] = {8U, 8U, 4U, 2U};

static const m3_config_field m3_lm_fields[] = {
    {"architectures", M3_CONFIG_REPEAT_STRING_ARRAY, 0U, 0,
     "Qwen3ForCausalLM", NULL, 1U, false},
    {"attention_bias", M3_CONFIG_BOOL, 0U, 0, NULL, NULL, 0U, false},
    {"attention_dropout", M3_CONFIG_DECIMAL, 0U, 0, NULL, NULL, 0U, false},
    {"bos_token_id", M3_CONFIG_NULL, 0U, 0, NULL, NULL, 0U, false},
    {"dtype", M3_CONFIG_STRING, 0U, 0, "bfloat16", NULL, 0U, false},
    {"eos_token_id", M3_CONFIG_NULL, 0U, 0, NULL, NULL, 0U, false},
    {"head_dim", M3_CONFIG_UINT, 128U, 0, NULL, NULL, 0U, false},
    {"hidden_act", M3_CONFIG_STRING, 0U, 0, "silu", NULL, 0U, false},
    {"hidden_size", M3_CONFIG_UINT, 4096U, 0, NULL, NULL, 0U, false},
    {"initializer_range", M3_CONFIG_DECIMAL, 2U, -2, NULL, NULL, 0U,
     false},
    {"intermediate_size", M3_CONFIG_UINT, 12288U, 0, NULL, NULL, 0U,
     false},
    {"layer_types", M3_CONFIG_REPEAT_STRING_ARRAY, 0U, 0,
     "full_attention", NULL, 36U, false},
    {"max_position_embeddings", M3_CONFIG_UINT, 10240U, 0, NULL, NULL, 0U,
     false},
    {"max_window_layers", M3_CONFIG_UINT, 28U, 0, NULL, NULL, 0U, false},
    {"model_type", M3_CONFIG_STRING, 0U, 0, "qwen3", NULL, 0U, false},
    {"num_attention_heads", M3_CONFIG_UINT, 32U, 0, NULL, NULL, 0U, false},
    {"num_hidden_layers", M3_CONFIG_UINT, 36U, 0, NULL, NULL, 0U, false},
    {"num_key_value_heads", M3_CONFIG_UINT, 8U, 0, NULL, NULL, 0U, false},
    {"pad_token_id", M3_CONFIG_NULL, 0U, 0, NULL, NULL, 0U, false},
    {"rms_norm_eps", M3_CONFIG_DECIMAL, 1U, -6, NULL, NULL, 0U, false},
    {"rope_parameters", M3_CONFIG_ROPE, 0U, 0, NULL, NULL, 0U, false},
    {"sliding_window", M3_CONFIG_NULL, 0U, 0, NULL, NULL, 0U, false},
    {"tie_word_embeddings", M3_CONFIG_BOOL, 0U, 0, NULL, NULL, 0U, false},
    {"transformers_version", M3_CONFIG_NONEMPTY_STRING, 0U, 0, NULL, NULL,
     0U, true},
    {"use_cache", M3_CONFIG_BOOL, 1U, 0, NULL, NULL, 0U, false},
    {"use_sliding_window", M3_CONFIG_BOOL, 0U, 0, NULL, NULL, 0U, false},
    {"vocab_size", M3_CONFIG_UINT, 200000U, 0, NULL, NULL, 0U, false}
};

static const m3_config_field m3_rvq_fields[] = {
    {"_class_name", M3_CONFIG_STRING, 0U, 0,
     "MiniMaxMusic3RVQDepthDecoder", NULL, 0U, false},
    {"_diffusers_version", M3_CONFIG_NONEMPTY_STRING, 0U, 0, NULL, NULL, 0U,
     true},
    {"audio_vocab_size", M3_CONFIG_UINT, 1024U, 0, NULL, NULL, 0U, false},
    {"hidden_size", M3_CONFIG_UINT, 4096U, 0, NULL, NULL, 0U, false},
    {"intermediate_size", M3_CONFIG_UINT, 6144U, 0, NULL, NULL, 0U, false},
    {"max_position_embeddings", M3_CONFIG_UINT, 16U, 0, NULL, NULL, 0U,
     false},
    {"num_attention_heads", M3_CONFIG_UINT, 16U, 0, NULL, NULL, 0U, false},
    {"num_codebooks", M3_CONFIG_UINT, 8U, 0, NULL, NULL, 0U, false},
    {"num_layers", M3_CONFIG_UINT, 4U, 0, NULL, NULL, 0U, false}
};

static const m3_config_field m3_condition_fields[] = {
    {"_class_name", M3_CONFIG_STRING, 0U, 0,
     "MiniMaxMusic3ConditionEncoder", NULL, 0U, false},
    {"_diffusers_version", M3_CONFIG_NONEMPTY_STRING, 0U, 0, NULL, NULL, 0U,
     true},
    {"condition_hidden_dim", M3_CONFIG_UINT, 4096U, 0, NULL, NULL, 0U,
     false},
    {"input_hop_length", M3_CONFIG_UINT, 960U, 0, NULL, NULL, 0U, false},
    {"input_sampling_rate", M3_CONFIG_UINT, 24000U, 0, NULL, NULL, 0U,
     false},
    {"num_condition_layers", M3_CONFIG_UINT, 8U, 0, NULL, NULL, 0U, false},
    {"out_dim", M3_CONFIG_UINT, 2048U, 0, NULL, NULL, 0U, false},
    {"output_hop_length", M3_CONFIG_UINT, 512U, 0, NULL, NULL, 0U, false},
    {"output_sampling_rate", M3_CONFIG_UINT, 44100U, 0, NULL, NULL, 0U,
     false}
};

static const m3_config_field m3_flow_fields[] = {
    {"_class_name", M3_CONFIG_STRING, 0U, 0,
     "MiniMaxMusic3Transformer1DModel", NULL, 0U, false},
    {"_diffusers_version", M3_CONFIG_NONEMPTY_STRING, 0U, 0, NULL, NULL, 0U,
     true},
    {"attention_head_dim", M3_CONFIG_UINT, 64U, 0, NULL, NULL, 0U, false},
    {"condition_dim", M3_CONFIG_UINT, 2048U, 0, NULL, NULL, 0U, false},
    {"ff_inner_dim", M3_CONFIG_UINT, 8192U, 0, NULL, NULL, 0U, false},
    {"fourier_embedding_dim", M3_CONFIG_UINT, 256U, 0, NULL, NULL, 0U,
     false},
    {"in_channels", M3_CONFIG_UINT, 128U, 0, NULL, NULL, 0U, false},
    {"num_attention_heads", M3_CONFIG_UINT, 32U, 0, NULL, NULL, 0U, false},
    {"num_layers", M3_CONFIG_UINT, 36U, 0, NULL, NULL, 0U, false},
    {"rotary_dim", M3_CONFIG_UINT, 32U, 0, NULL, NULL, 0U, false}
};

static const m3_config_field m3_vocoder_fields[] = {
    {"_class_name", M3_CONFIG_STRING, 0U, 0, "MiniMaxMusic3Vocoder", NULL,
     0U, false},
    {"_diffusers_version", M3_CONFIG_NONEMPTY_STRING, 0U, 0, NULL, NULL, 0U,
     true},
    {"decoder_hidden_dim", M3_CONFIG_UINT, 1536U, 0, NULL, NULL, 0U, false},
    {"decoder_input_dim", M3_CONFIG_UINT, 1024U, 0, NULL, NULL, 0U, false},
    {"latent_channels", M3_CONFIG_UINT, 128U, 0, NULL, NULL, 0U, false},
    {"sampling_rate", M3_CONFIG_UINT, 44100U, 0, NULL, NULL, 0U, false},
    {"upsampling_ratios", M3_CONFIG_UINT_ARRAY, 0U, 0, NULL,
     m3_vocoder_ratios, 4U, false}
};

static m3_status m3_config_invalid(m3_error *error, const char *field)
{
    return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                        "Music3 config field '%s' has an unexpected value",
                        field);
}

static m3_status m3_config_read_literal(m3_json_reader *reader,
                                        const char *literal,
                                        m3_error *error)
{
    size_t length = strlen(literal);

    (void)error;
    (void)m3_json_next_is(reader, (uint8_t)literal[0]);
    if (reader->size - reader->position < length ||
        memcmp(reader->data + reader->position, literal, length) != 0) {
        return M3_STATUS_INVALID_FORMAT;
    }
    reader->position += length;
    return M3_STATUS_OK;
}

static m3_status m3_config_read_decimal(m3_json_reader *reader,
                                        uint64_t expected_mantissa,
                                        int expected_exponent,
                                        m3_error *error)
{
    size_t start;
    size_t end;
    size_t index;
    size_t decimal_digits = 0U;
    size_t trailing_zeros = 0U;
    uint64_t mantissa = 0U;
    int64_t exponent = 0;
    bool negative = false;
    bool after_decimal = false;
    bool significant = false;
    m3_status status;

    (void)m3_json_next_is(reader, (uint8_t)'0');
    start = reader->position;
    if (start >= reader->size ||
        (reader->data[start] != (uint8_t)'-' &&
         (reader->data[start] < (uint8_t)'0' ||
          reader->data[start] > (uint8_t)'9'))) {
        return M3_STATUS_INVALID_FORMAT;
    }
    status = m3_json_skip_value(reader, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    end = reader->position;
    index = start;
    if (index < end && reader->data[index] == (uint8_t)'-') {
        negative = true;
        index += 1U;
    }
    while (index < end && reader->data[index] != (uint8_t)'e' &&
           reader->data[index] != (uint8_t)'E') {
        uint8_t byte = reader->data[index++];

        if (byte == (uint8_t)'.') {
            after_decimal = true;
            continue;
        }
        if (after_decimal) {
            decimal_digits += 1U;
        }
        if (byte != (uint8_t)'0' || significant) {
            uint64_t digit = (uint64_t)(byte - (uint8_t)'0');

            significant = true;
            if (mantissa > (UINT64_MAX - digit) / 10U) {
                return M3_STATUS_INVALID_FORMAT;
            }
            mantissa = mantissa * 10U + digit;
            trailing_zeros = byte == (uint8_t)'0' ? trailing_zeros + 1U : 0U;
        }
    }
    if (index < end) {
        bool exponent_negative = false;
        int parsed_exponent = 0;

        index += 1U;
        if (index < end && (reader->data[index] == (uint8_t)'+' ||
                            reader->data[index] == (uint8_t)'-')) {
            exponent_negative = reader->data[index] == (uint8_t)'-';
            index += 1U;
        }
        while (index < end) {
            unsigned int digit =
                (unsigned int)(reader->data[index++] - (uint8_t)'0');

            if (parsed_exponent > (INT_MAX - (int)digit) / 10) {
                return M3_STATUS_INVALID_FORMAT;
            }
            parsed_exponent = parsed_exponent * 10 + (int)digit;
        }
        exponent = exponent_negative ? -(int64_t)parsed_exponent
                                     : (int64_t)parsed_exponent;
    }
    if (mantissa == 0U) {
        exponent = 0;
        negative = false;
    } else {
        if (decimal_digits > (size_t)INT_MAX ||
            trailing_zeros > (size_t)INT_MAX) {
            return M3_STATUS_INVALID_FORMAT;
        }
        exponent -= (int64_t)decimal_digits;
        exponent += (int64_t)trailing_zeros;
        while (trailing_zeros != 0U) {
            mantissa /= 10U;
            trailing_zeros -= 1U;
        }
    }
    return !negative && mantissa == expected_mantissa &&
                   exponent == (int64_t)expected_exponent
               ? M3_STATUS_OK
               : M3_STATUS_INVALID_FORMAT;
}

static m3_status m3_config_read_string_value(m3_json_reader *reader,
                                              const char *expected,
                                              bool nonempty,
                                              m3_error *error)
{
    char *value = NULL;
    m3_status status = m3_json_read_string(reader, &value, error);

    if (status == M3_STATUS_OK &&
        ((expected != NULL && strcmp(value, expected) != 0) ||
         (nonempty && value[0] == '\0'))) {
        status = M3_STATUS_INVALID_FORMAT;
    }
    free(value);
    return status;
}

static m3_status m3_config_read_repeat_array(m3_json_reader *reader,
                                              const m3_config_field *field,
                                              m3_error *error)
{
    size_t index;
    m3_status status = m3_json_expect(reader, (uint8_t)'[', error);

    for (index = 0U; index < field->count && status == M3_STATUS_OK; ++index) {
        if (index != 0U) {
            status = m3_json_expect(reader, (uint8_t)',', error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_config_read_string_value(reader, field->text, false,
                                                  error);
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_expect(reader, (uint8_t)']', error);
    }
    return status;
}

static m3_status m3_config_read_uint_array(m3_json_reader *reader,
                                            const m3_config_field *field,
                                            m3_error *error)
{
    size_t index;
    m3_status status = m3_json_expect(reader, (uint8_t)'[', error);

    for (index = 0U; index < field->count && status == M3_STATUS_OK; ++index) {
        uint64_t value = 0U;

        if (index != 0U) {
            status = m3_json_expect(reader, (uint8_t)',', error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_json_read_uint64(reader, &value, error);
        }
        if (status == M3_STATUS_OK && value != field->array[index]) {
            status = M3_STATUS_INVALID_FORMAT;
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_expect(reader, (uint8_t)']', error);
    }
    return status;
}

static m3_status m3_config_read_rope(m3_json_reader *reader,
                                      m3_error *error)
{
    unsigned int seen = 0U;
    bool first = true;
    m3_status status = m3_json_expect(reader, (uint8_t)'{', error);

    while (status == M3_STATUS_OK &&
           !m3_json_next_is(reader, (uint8_t)'}')) {
        char *name = NULL;

        if (!first) {
            status = m3_json_expect(reader, (uint8_t)',', error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_json_read_string(reader, &name, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_json_expect(reader, (uint8_t)':', error);
        }
        if (status == M3_STATUS_OK && strcmp(name, "rope_type") == 0 &&
            (seen & 1U) == 0U) {
            status = m3_config_read_string_value(reader, "default", false,
                                                  error);
            seen |= 1U;
        } else if (status == M3_STATUS_OK &&
                   strcmp(name, "rope_theta") == 0 && (seen & 2U) == 0U) {
            uint64_t value = 0U;

            status = m3_json_read_uint64(reader, &value, error);
            if (status == M3_STATUS_OK && value != 1000000U) {
                status = M3_STATUS_INVALID_FORMAT;
            }
            seen |= 2U;
        } else if (status == M3_STATUS_OK) {
            status = M3_STATUS_INVALID_FORMAT;
        }
        free(name);
        first = false;
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_expect(reader, (uint8_t)'}', error);
    }
    return status == M3_STATUS_OK && seen == 3U ? M3_STATUS_OK
                                                : M3_STATUS_INVALID_FORMAT;
}

static m3_status m3_config_read_field(m3_json_reader *reader,
                                       const m3_config_field *field,
                                       m3_error *error)
{
    m3_status status;
    if (reader == NULL || field == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 config field argument is null");
    }
    switch (field->kind) {
    case M3_CONFIG_UINT: {
        uint64_t value = 0U;

        status = m3_json_read_uint64(reader, &value, error);
        return status == M3_STATUS_OK && value == field->number
                   ? M3_STATUS_OK
                   : m3_config_invalid(error, field->name);
    }
    case M3_CONFIG_DECIMAL:
        status = m3_config_read_decimal(reader, field->number,
                                        field->exponent, error);
        break;
    case M3_CONFIG_STRING:
        status = m3_config_read_string_value(reader, field->text, false,
                                             error);
        break;
    case M3_CONFIG_NONEMPTY_STRING:
        status = m3_config_read_string_value(reader, NULL, true, error);
        break;
    case M3_CONFIG_BOOL:
        status = m3_config_read_literal(reader,
                                        field->number != 0U ? "true" : "false",
                                        error);
        break;
    case M3_CONFIG_NULL:
        status = m3_config_read_literal(reader, "null", error);
        break;
    case M3_CONFIG_REPEAT_STRING_ARRAY:
        status = m3_config_read_repeat_array(reader, field, error);
        break;
    case M3_CONFIG_UINT_ARRAY:
        status = m3_config_read_uint_array(reader, field, error);
        break;
    case M3_CONFIG_ROPE:
        status = m3_config_read_rope(reader, error);
        break;
    default:
        status = M3_STATUS_INVALID_FORMAT;
        break;
    }
    if (status != M3_STATUS_OK) {
        return m3_config_invalid(error, field->name);
    }
    return M3_STATUS_OK;
}

static m3_status m3_config_specs(m3_component_id id,
                                  const m3_config_field **fields,
                                  size_t *field_count, m3_error *error)
{
    switch (id) {
    case M3_COMPONENT_LANGUAGE_MODEL:
        *fields = m3_lm_fields;
        *field_count = sizeof(m3_lm_fields) / sizeof(m3_lm_fields[0]);
        break;
    case M3_COMPONENT_RVQ_DEPTH_DECODER:
        *fields = m3_rvq_fields;
        *field_count = sizeof(m3_rvq_fields) / sizeof(m3_rvq_fields[0]);
        break;
    case M3_COMPONENT_CONDITION_ENCODER:
        *fields = m3_condition_fields;
        *field_count = sizeof(m3_condition_fields) /
                       sizeof(m3_condition_fields[0]);
        break;
    case M3_COMPONENT_TRANSFORMER:
        *fields = m3_flow_fields;
        *field_count = sizeof(m3_flow_fields) / sizeof(m3_flow_fields[0]);
        break;
    case M3_COMPONENT_VOCODER:
        *fields = m3_vocoder_fields;
        *field_count = sizeof(m3_vocoder_fields) /
                       sizeof(m3_vocoder_fields[0]);
        break;
    default:
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "component has no Music3 weight config");
    }
    return M3_STATUS_OK;
}

static m3_status m3_config_parse(const uint8_t *data, size_t size,
                                  m3_component_id id, m3_error *error)
{
    const m3_config_field *fields = NULL;
    size_t field_count = 0U;
    uint64_t seen = 0U;
    m3_json_reader reader;
    bool first = true;
    m3_status status = m3_config_specs(id, &fields, &field_count, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_json_reader_init(&reader, data, size);
    status = m3_json_expect(&reader, (uint8_t)'{', error);
    while (status == M3_STATUS_OK &&
           !m3_json_next_is(&reader, (uint8_t)'}')) {
        char *name = NULL;
        size_t index;

        if (!first) {
            status = m3_json_expect(&reader, (uint8_t)',', error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_json_read_string(&reader, &name, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_json_expect(&reader, (uint8_t)':', error);
        }
        for (index = 0U; index < field_count && status == M3_STATUS_OK;
             ++index) {
            if (strcmp(name, fields[index].name) == 0) {
                break;
            }
        }
        if (status == M3_STATUS_OK &&
            (index == field_count || (seen & (UINT64_C(1) << index)) != 0U)) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "unknown or duplicate Music3 config field '%s'",
                                  name);
        }
        if (status == M3_STATUS_OK) {
            status = m3_config_read_field(&reader, &fields[index], error);
            seen |= UINT64_C(1) << index;
        }
        free(name);
        first = false;
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_expect(&reader, (uint8_t)'}', error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_json_finish(&reader, error);
    }
    if (status == M3_STATUS_OK) {
        size_t index;

        for (index = 0U; index < field_count; ++index) {
            if (!fields[index].optional &&
                (seen & (UINT64_C(1) << index)) == 0U) {
                return m3_error_set(
                    error, M3_STATUS_INVALID_FORMAT,
                    "Music3 config lacks required field '%s'",
                    fields[index].name);
            }
        }
    }
    return status;
}

void m3_music3_component_config_init(m3_music3_component_config *config)
{
    if (config != NULL) {
        (void)memset(config, 0, sizeof(*config));
        config->id = M3_COMPONENT_COUNT;
    }
}

static void m3_config_fill(m3_component_id id,
                           m3_music3_component_config *config)
{
    m3_music3_component_config_init(config);
    config->id = id;
    config->valid = true;
    switch (id) {
    case M3_COMPONENT_LANGUAGE_MODEL:
        config->hidden_size = 4096U;
        break;
    case M3_COMPONENT_RVQ_DEPTH_DECODER:
        config->hidden_size = 4096U;
        config->codebooks = 8U;
        break;
    case M3_COMPONENT_CONDITION_ENCODER:
        config->hidden_size = 4096U;
        config->condition_layers = 8U;
        config->condition_dim = 2048U;
        config->output_hop_length = 512U;
        config->output_sampling_rate = 44100U;
        break;
    case M3_COMPONENT_TRANSFORMER:
        config->condition_dim = 2048U;
        config->input_channels = 128U;
        break;
    case M3_COMPONENT_VOCODER:
        config->latent_channels = 128U;
        config->upsampling_product = 512U;
        config->sampling_rate = 44100U;
        break;
    default:
        break;
    }
}

m3_status m3_music3_config_read_file(
    const char *path, m3_component_id id,
    m3_music3_component_config *config, m3_error *error)
{
    uint8_t *data = NULL;
    size_t size = 0U;
    m3_music3_component_config parsed;
    m3_status status;

    if (path == NULL || config == NULL || !m3_component_contains_weights(id)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 config argument is invalid");
    }
    m3_music3_component_config_init(&parsed);
    status = m3_file_read_bounded(path, M3_MUSIC3_CONFIG_MAX_BYTES, &data,
                                  &size, error);
    if (status == M3_STATUS_OK) {
        status = m3_config_parse(data, size, id, error);
    }
    free(data);
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_config_fill(id, &parsed);
    *config = parsed;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_music3_config_validate_cross(
    const m3_music3_component_config configs[M3_COMPONENT_COUNT],
    m3_error *error)
{
    const m3_music3_component_config *lm;
    const m3_music3_component_config *rvq;
    const m3_music3_component_config *condition;
    const m3_music3_component_config *flow;
    const m3_music3_component_config *vocoder;
    size_t index;

    if (configs == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 config set is null");
    }
    for (index = 0U; index <= (size_t)M3_COMPONENT_VOCODER; ++index) {
        if (!configs[index].valid || configs[index].id != (m3_component_id)index) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "Music3 config set is incomplete");
        }
    }
    lm = &configs[M3_COMPONENT_LANGUAGE_MODEL];
    rvq = &configs[M3_COMPONENT_RVQ_DEPTH_DECODER];
    condition = &configs[M3_COMPONENT_CONDITION_ENCODER];
    flow = &configs[M3_COMPONENT_TRANSFORMER];
    vocoder = &configs[M3_COMPONENT_VOCODER];
    if (lm->hidden_size != rvq->hidden_size ||
        lm->hidden_size != condition->hidden_size ||
        rvq->codebooks != condition->condition_layers ||
        condition->condition_dim != flow->condition_dim ||
        flow->input_channels != vocoder->latent_channels ||
        condition->output_hop_length != vocoder->upsampling_product ||
        condition->output_sampling_rate != vocoder->sampling_rate) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "Music3 component configs are inconsistent");
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

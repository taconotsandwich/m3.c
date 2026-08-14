/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_music3_schema.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    m3_safetensors_tensor *tensors;
    size_t count;
    size_t capacity;
    uint64_t elements;
    uint64_t bytes;
} m3_schema_fixture;

static void m3_schema_fixture_dispose(m3_schema_fixture *fixture)
{
    size_t index;

    for (index = 0U; index < fixture->count; ++index) {
        free(fixture->tensors[index].name);
    }
    free(fixture->tensors);
    (void)memset(fixture, 0, sizeof(*fixture));
}

static m3_status m3_schema_fixture_visit(
    const char *name, const m3_tensor_metadata *tensor, void *context,
    m3_error *error)
{
    m3_schema_fixture *fixture = context;
    size_t length = strlen(name);
    char *copy;

    if (fixture->count == fixture->capacity) {
        size_t capacity = fixture->capacity == 0U
                              ? 64U
                              : fixture->capacity * 2U;
        m3_safetensors_tensor *tensors;

        if (capacity < fixture->capacity ||
            capacity > SIZE_MAX / sizeof(*tensors)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "test schema fixture overflows");
        }
        tensors = realloc(fixture->tensors, capacity * sizeof(*tensors));
        if (tensors == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot grow test schema fixture");
        }
        fixture->tensors = tensors;
        fixture->capacity = capacity;
    }
    copy = malloc(length + 1U);
    if (copy == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot copy test schema name");
    }
    (void)memcpy(copy, name, length + 1U);
    fixture->tensors[fixture->count].name = copy;
    fixture->tensors[fixture->count].tensor = *tensor;
    fixture->tensors[fixture->count].data_start = fixture->bytes;
    fixture->bytes += (uint64_t)tensor->byte_count;
    fixture->tensors[fixture->count].data_end = fixture->bytes;
    fixture->elements += (uint64_t)tensor->element_count;
    fixture->count += 1U;
    return M3_STATUS_OK;
}

static const m3_safetensors_tensor *m3_schema_fixture_find(
    const m3_schema_fixture *fixture, const char *name)
{
    size_t index;

    for (index = 0U; index < fixture->count; ++index) {
        if (strcmp(fixture->tensors[index].name, name) == 0) {
            return &fixture->tensors[index];
        }
    }
    return NULL;
}

static bool m3_schema_fixture_unique(const m3_schema_fixture *fixture)
{
    size_t left;

    for (left = 0U; left < fixture->count; ++left) {
        size_t right;

        for (right = left + 1U; right < fixture->count; ++right) {
            if (strcmp(fixture->tensors[left].name,
                       fixture->tensors[right].name) == 0) {
                return false;
            }
        }
    }
    return true;
}

void m3_test_music3_schema_inventory(m3_test_context *test)
{
    m3_component_id id;

    for (id = M3_COMPONENT_LANGUAGE_MODEL; id <= M3_COMPONENT_VOCODER;
         id = (m3_component_id)((int)id + 1)) {
        m3_schema_fixture fixture = {NULL, 0U, 0U, 0U, 0U};
        m3_music3_schema_summary summary;
        m3_error error;
        m3_status visit_status;
        m3_status summary_status = m3_music3_schema_expected_summary(
            id, &summary, &error);

        (void)m3_error_set(&error, M3_STATUS_INVALID_FORMAT,
                           "stale schema error");
        visit_status = m3_music3_schema_visit(
            id, m3_schema_fixture_visit, &fixture, &error);

        M3_TEST_EXPECT(test, visit_status == M3_STATUS_OK,
                       "generate complete official component schema");
        M3_TEST_EXPECT(test,
                       error.status == M3_STATUS_OK &&
                           m3_error_message(&error)[0] == '\0',
                       "successful schema visits clear stale errors");
        M3_TEST_EXPECT(test, summary_status == M3_STATUS_OK,
                       "read official component schema summary");
        M3_TEST_EXPECT(test,
                       fixture.count == summary.tensor_count &&
                           fixture.elements == summary.element_count &&
                           fixture.bytes == summary.payload_bytes,
                       "match exact tensor count and aggregate sizes");
        M3_TEST_EXPECT(test, m3_schema_fixture_unique(&fixture),
                       "generate unique exact tensor names");
        M3_TEST_EXPECT(
            test,
            m3_music3_schema_validate_inventory(
                id, fixture.tensors, fixture.count, &error) == M3_STATUS_OK,
            "accept complete generated inventory through production validator");
        m3_schema_fixture_dispose(&fixture);
    }
}

void m3_test_music3_schema_names(m3_test_context *test)
{
    m3_schema_fixture lm = {NULL, 0U, 0U, 0U, 0U};
    m3_schema_fixture rvq = {NULL, 0U, 0U, 0U, 0U};
    m3_schema_fixture condition = {NULL, 0U, 0U, 0U, 0U};
    m3_schema_fixture flow = {NULL, 0U, 0U, 0U, 0U};
    m3_schema_fixture vocoder = {NULL, 0U, 0U, 0U, 0U};
    m3_error error;
    bool ready =
        m3_music3_schema_visit(M3_COMPONENT_LANGUAGE_MODEL,
                               m3_schema_fixture_visit, &lm, &error) ==
            M3_STATUS_OK &&
        m3_music3_schema_visit(M3_COMPONENT_RVQ_DEPTH_DECODER,
                               m3_schema_fixture_visit, &rvq, &error) ==
            M3_STATUS_OK &&
        m3_music3_schema_visit(M3_COMPONENT_CONDITION_ENCODER,
                               m3_schema_fixture_visit, &condition, &error) ==
            M3_STATUS_OK &&
        m3_music3_schema_visit(M3_COMPONENT_TRANSFORMER,
                               m3_schema_fixture_visit, &flow, &error) ==
            M3_STATUS_OK &&
        m3_music3_schema_visit(M3_COMPONENT_VOCODER,
                               m3_schema_fixture_visit, &vocoder, &error) ==
            M3_STATUS_OK;

    M3_TEST_EXPECT(test, ready, "generate naming-rule fixtures");
    M3_TEST_EXPECT(
        test,
        m3_schema_fixture_find(&lm, "model.embed_tokens.weight") != NULL &&
            m3_schema_fixture_find(&lm, "lm_head.weight") != NULL &&
            m3_schema_fixture_find(
                &lm, "model.layers.35.mlp.down_proj.weight") != NULL &&
            m3_schema_fixture_find(
                &lm, "model.layers.35.self_attn.q_norm.weight") != NULL,
        "preserve first, terminal, and exceptional LM tensor names");
    M3_TEST_EXPECT(
        test,
        m3_schema_fixture_find(&rvq, "layers.0.attn.to_out.weight") != NULL &&
            m3_schema_fixture_find(&rvq,
                                   "layers.0.attn.to_out.0.weight") == NULL &&
            m3_schema_fixture_find(&rvq,
                                   "layers.0.gate_proj.weight") != NULL &&
            m3_schema_fixture_find(&rvq,
                                   "layers.0.mlp.gate_proj.weight") == NULL,
        "preserve exact RVQ attention and projection names");
    M3_TEST_EXPECT(
        test,
        m3_schema_fixture_find(&condition, "layer_weight_logits") != NULL &&
            m3_schema_fixture_find(&condition, "layer_scale") != NULL &&
            m3_schema_fixture_find(&condition, "proj.weight") != NULL &&
            m3_schema_fixture_find(&condition, "proj.bias") != NULL,
        "preserve the complete four-tensor condition schema");
    M3_TEST_EXPECT(
        test,
        m3_schema_fixture_find(
            &flow, "transformer_blocks.0.attn.to_out.0.weight") != NULL &&
            m3_schema_fixture_find(
                &flow, "transformer_blocks.0.attn.to_out.weight") == NULL &&
            m3_schema_fixture_find(
                &flow, "transformer_blocks.35.ff_out.bias") != NULL &&
            m3_schema_fixture_find(&flow,
                                   "postprocess_conv.weight") != NULL,
        "preserve exact flow output and terminal names");
    M3_TEST_EXPECT(
        test,
        m3_schema_fixture_find(&vocoder,
                               "blocks.0.res_unit1.conv1.weight_g") != NULL &&
            m3_schema_fixture_find(&vocoder,
                                   "blocks.0.res_unit1.conv1.weight_v") !=
                NULL &&
            m3_schema_fixture_find(&vocoder,
                                   "blocks.0.res_unit1.conv1.weight") == NULL &&
            m3_schema_fixture_find(&vocoder, "snake_out.alpha") != NULL &&
            m3_schema_fixture_find(&vocoder,
                                   "conv_out.weight_v") != NULL,
        "preserve legacy and terminal vocoder names");
    if (flow.count != 0U) {
        m3_tensor_metadata saved = flow.tensors[0].tensor;
        char *saved_name = flow.tensors[0].name;
        char saved_initial = saved_name[0];

        flow.tensors[0].tensor.dtype = M3_DTYPE_BF16;
        M3_TEST_EXPECT(
            test,
            m3_music3_schema_validate_inventory(
                M3_COMPONENT_TRANSFORMER, flow.tensors, flow.count, &error) ==
                M3_STATUS_INVALID_FORMAT,
            "reject a wrong tensor dtype");
        flow.tensors[0].tensor = saved;
        flow.tensors[0].tensor.shape[0] += 1U;
        M3_TEST_EXPECT(
            test,
            m3_music3_schema_validate_inventory(
                M3_COMPONENT_TRANSFORMER, flow.tensors, flow.count, &error) ==
                M3_STATUS_INVALID_FORMAT,
            "reject a wrong tensor shape");
        flow.tensors[0].tensor = saved;
        flow.tensors[0].name[0] = 'X';
        M3_TEST_EXPECT(
            test,
            m3_music3_schema_validate_inventory(
                M3_COMPONENT_TRANSFORMER, flow.tensors, flow.count, &error) ==
                M3_STATUS_INVALID_FORMAT,
            "reject an unexpected tensor name");
        flow.tensors[0].name[0] = saved_initial;
        flow.tensors[0].name = flow.tensors[1].name;
        M3_TEST_EXPECT(
            test,
            m3_music3_schema_validate_inventory(
                M3_COMPONENT_TRANSFORMER, flow.tensors, flow.count, &error) ==
                M3_STATUS_INVALID_FORMAT,
            "reject a duplicate tensor name");
        flow.tensors[0].name = saved_name;
        M3_TEST_EXPECT(
            test,
            m3_music3_schema_validate_inventory(
                M3_COMPONENT_TRANSFORMER, flow.tensors, flow.count - 1U,
                &error) == M3_STATUS_INVALID_FORMAT,
            "reject a missing tensor");
    }
    m3_schema_fixture_dispose(&vocoder);
    m3_schema_fixture_dispose(&flow);
    m3_schema_fixture_dispose(&condition);
    m3_schema_fixture_dispose(&rvq);
    m3_schema_fixture_dispose(&lm);
}

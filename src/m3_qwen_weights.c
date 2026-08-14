/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_qwen_internal.h"

#include "m3_music3_schema.h"

#include <stddef.h>
#include <string.h>

static const m3_qwen_dimensions m3_qwen_official = {
    M3_QWEN_VOCAB_SIZE,
    M3_QWEN_HIDDEN_SIZE,
    M3_QWEN_LAYER_COUNT,
    M3_QWEN_QUERY_HEADS,
    M3_QWEN_KEY_VALUE_HEADS,
    M3_QWEN_HEAD_DIMENSION,
    M3_QWEN_INTERMEDIATE_SIZE,
    M3_QWEN_EOS_TOKEN_ID,
    M3_QWEN_SEMANTIC_TOKEN_START,
    M3_QWEN_SEMANTIC_TOKEN_COUNT,
    1.0e-6F,
    1000000.0F
};

const m3_qwen_dimensions *m3_qwen_official_dimensions(void)
{
    return &m3_qwen_official;
}

static m3_status m3_qwen_weight_plan_visit(
    const char *name, const m3_tensor_metadata *tensor, void *context,
    m3_error *error)
{
    m3_qwen_weight_plan *plan = context;
    size_t length;
    size_t index;

    if (plan->count >= M3_QWEN_WEIGHT_COUNT) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "Qwen schema has more than 399 weights");
    }
    length = strlen(name);
    if (length >= M3_QWEN_WEIGHT_NAME_CAPACITY) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Qwen weight name exceeds plan capacity");
    }
    index = plan->count;
    (void)memcpy(plan->names[index], name, length + 1U);
    plan->requirements[index].name = plan->names[index];
    plan->requirements[index].tensor = *tensor;
    ++plan->count;
    return M3_STATUS_OK;
}

m3_status m3_qwen_weight_plan_init(m3_qwen_weight_plan *plan,
                                   m3_error *error)
{
    m3_status status;

    if (plan == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen weight plan is null");
    }
    (void)memset(plan, 0, sizeof(*plan));
    status = m3_music3_schema_visit(M3_COMPONENT_LANGUAGE_MODEL,
                                    m3_qwen_weight_plan_visit, plan,
                                    error);
    if (status == M3_STATUS_OK && plan->count != M3_QWEN_WEIGHT_COUNT) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "Qwen schema has %zu weights instead of 399",
                              plan->count);
    }
    if (status != M3_STATUS_OK) {
        (void)memset(plan, 0, sizeof(*plan));
        return status;
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

static void m3_qwen_layer_bind(m3_qwen_layer_weights *layer,
                               const m3_tensor_view *const *views)
{
    layer->input_norm = views[0];
    layer->query_weight = views[1];
    layer->key_weight = views[2];
    layer->value_weight = views[3];
    layer->output_weight = views[4];
    layer->query_norm = views[5];
    layer->key_norm = views[6];
    layer->post_attention_norm = views[7];
    layer->gate_weight = views[8];
    layer->up_weight = views[9];
    layer->down_weight = views[10];
}

m3_status m3_qwen_weights_bind(m3_qwen_weights *weights,
                               const m3_weight_stage *stage,
                               m3_error *error)
{
    m3_qwen_weight_plan plan;
    const m3_tensor_view *views[M3_QWEN_WEIGHT_COUNT];
    m3_qwen_weights bound;
    size_t layer;
    m3_status status;

    if (weights == NULL || stage == NULL || stage->backend == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen weights and staged model are required");
    }
    status = m3_qwen_weight_plan_init(&plan, error);
    if (status == M3_STATUS_OK) {
        status = m3_weight_stage_resolve_required(
            stage, plan.requirements, plan.count, views, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    (void)memset(&bound, 0, sizeof(bound));
    bound.embedding = views[0];
    bound.final_norm = views[1];
    bound.head = views[2];
    for (layer = 0U; layer < M3_QWEN_LAYER_COUNT; ++layer) {
        m3_qwen_layer_bind(&bound.layers[layer],
                           &views[3U + layer * 11U]);
    }
    *weights = bound;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

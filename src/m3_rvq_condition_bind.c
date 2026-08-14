/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_rvq_condition_internal.h"

#include "m3_music3_schema.h"

#include <string.h>

typedef struct {
    char (*name_storage)[128];
    m3_weight_requirement *requirements;
    size_t capacity;
    size_t count;
} m3_requirement_builder;

static m3_status m3_requirement_visit(
    const char *name, const m3_tensor_metadata *tensor, void *context,
    m3_error *error)
{
    m3_requirement_builder *builder = context;
    size_t length;

    if (builder->count >= builder->capacity) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "runtime requirement count overflows");
    }
    length = strlen(name);
    if (length >= sizeof(builder->name_storage[builder->count])) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "runtime weight name overflows");
    }
    (void)memcpy(builder->name_storage[builder->count], name, length + 1U);
    builder->requirements[builder->count].name =
        builder->name_storage[builder->count];
    builder->requirements[builder->count].tensor = *tensor;
    ++builder->count;
    return M3_STATUS_OK;
}

static m3_status m3_requirement_build(
    m3_component_id component, char (*names)[128],
    m3_weight_requirement *requirements, size_t capacity, size_t *count,
    m3_error *error)
{
    m3_requirement_builder builder = {
        names, requirements, capacity, 0U
    };
    m3_music3_schema_summary summary;
    m3_status status = m3_music3_schema_expected_summary(
        component, &summary, error);

    if (status == M3_STATUS_OK && summary.tensor_count != capacity) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "runtime schema capacity mismatch");
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_visit(
            component, m3_requirement_visit, &builder, error);
    }
    if (status == M3_STATUS_OK && builder.count != capacity) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "runtime schema count mismatch");
    }
    if (status == M3_STATUS_OK) {
        *count = builder.count;
    }
    return status;
}

m3_status m3_rvq_requirements(m3_rvq_requirement_set *set,
                              m3_error *error)
{
    m3_rvq_requirement_set built = {0};
    m3_status status;

    if (set == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "RVQ requirement output is required");
    }
    status = m3_requirement_build(
        M3_COMPONENT_RVQ_DEPTH_DECODER, built.name_storage,
        built.requirements, 47U, &built.count, error);
    if (status == M3_STATUS_OK) {
        size_t index;

        *set = built;
        for (index = 0U; index < set->count; ++index) {
            set->requirements[index].name = set->name_storage[index];
        }
        m3_error_reset(error);
    }
    return status;
}

m3_status m3_condition_requirements(m3_condition_requirement_set *set,
                                    m3_error *error)
{
    m3_condition_requirement_set built = {0};
    m3_status status;

    if (set == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "condition requirement output is required");
    }
    status = m3_requirement_build(
        M3_COMPONENT_CONDITION_ENCODER, built.name_storage,
        built.requirements, 4U, &built.count, error);
    if (status == M3_STATUS_OK) {
        size_t index;

        *set = built;
        for (index = 0U; index < set->count; ++index) {
            set->requirements[index].name = set->name_storage[index];
        }
        m3_error_reset(error);
    }
    return status;
}

m3_status m3_rvq_weights_bind(const m3_weight_stage *stage,
                              m3_rvq_weights *weights, m3_error *error)
{
    m3_rvq_requirement_set requirements;
    const m3_tensor_view *views[47];
    m3_rvq_weights built = {0};
    size_t layer;
    size_t head;
    m3_status status;

    if (weights == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "RVQ weight output is required");
    }
    status = m3_rvq_requirements(&requirements, error);
    if (status == M3_STATUS_OK) {
        status = m3_weight_stage_resolve_required(
            stage, requirements.requirements, requirements.count, views,
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_weight_table_validate_no_extra(
            stage->table, requirements.requirements, requirements.count,
            error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    built.audio_embeddings = views[0];
    built.projection = views[1];
    built.position_embeddings = views[2];
    built.norm = views[3];
    for (layer = 0U; layer < M3_RVQ_LAYER_CAPACITY; ++layer) {
        size_t base = 4U + layer * 9U;
        m3_rvq_layer_weights *target = &built.layers[layer];

        target->input_norm = views[base];
        target->query = views[base + 1U];
        target->key = views[base + 2U];
        target->value = views[base + 3U];
        target->attention_out = views[base + 4U];
        target->post_attention_norm = views[base + 5U];
        target->gate = views[base + 6U];
        target->up = views[base + 7U];
        target->down = views[base + 8U];
    }
    for (head = 0U; head < M3_RVQ_RESIDUAL_COUNT; ++head) {
        built.heads[head] = views[40U + head];
    }
    *weights = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_condition_weights_bind(
    const m3_weight_stage *stage, m3_condition_weights *weights,
    m3_error *error)
{
    m3_condition_requirement_set requirements;
    const m3_tensor_view *views[4];
    m3_condition_weights built = {0};
    m3_status status;

    if (weights == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "condition weight output is required");
    }
    status = m3_condition_requirements(&requirements, error);
    if (status == M3_STATUS_OK) {
        status = m3_weight_stage_resolve_required(
            stage, requirements.requirements, requirements.count, views,
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_weight_table_validate_no_extra(
            stage->table, requirements.requirements, requirements.count,
            error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    built.layer_weight_logits = views[0];
    built.layer_scale = views[1];
    built.projection = views[2];
    built.bias = views[3];
    *weights = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

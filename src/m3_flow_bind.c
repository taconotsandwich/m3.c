/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_flow_internal.h"

#include "m3_music3_schema.h"

#include <string.h>

typedef struct {
    m3_flow_requirement_set *set;
} m3_flow_requirement_builder;

static m3_status m3_flow_requirement_visit(
    const char *name, const m3_tensor_metadata *tensor, void *context,
    m3_error *error)
{
    m3_flow_requirement_builder *builder = context;
    m3_flow_requirement_set *set = builder->set;
    size_t length;

    if (set->count >= M3_FLOW_WEIGHT_COUNT) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "flow requirement count overflows");
    }
    length = strlen(name);
    if (length >= sizeof(set->names[set->count])) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "flow weight name overflows");
    }
    (void)memcpy(set->names[set->count], name, length + 1U);
    set->requirements[set->count].name = set->names[set->count];
    set->requirements[set->count].tensor = *tensor;
    ++set->count;
    return M3_STATUS_OK;
}

m3_status m3_flow_requirements(m3_flow_requirement_set *requirements,
                               m3_error *error)
{
    m3_flow_requirement_set built = {0};
    m3_flow_requirement_builder builder = {&built};
    m3_music3_schema_summary summary;
    size_t index;
    m3_status status;

    if (requirements == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow requirement output is required");
    }
    status = m3_music3_schema_expected_summary(
        M3_COMPONENT_TRANSFORMER, &summary, error);
    if (status == M3_STATUS_OK &&
        summary.tensor_count != M3_FLOW_WEIGHT_COUNT) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "flow schema capacity mismatch");
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_schema_visit(
            M3_COMPONENT_TRANSFORMER, m3_flow_requirement_visit,
            &builder, error);
    }
    if (status == M3_STATUS_OK && built.count != M3_FLOW_WEIGHT_COUNT) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "flow schema count mismatch");
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    *requirements = built;
    for (index = 0U; index < requirements->count; ++index) {
        requirements->requirements[index].name =
            requirements->names[index];
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

static void m3_flow_bind_layer(m3_flow_layer_weights *layer,
                               const m3_tensor_view *const *views)
{
    layer->norm1_scale = views[0];
    layer->norm1_bias = views[1];
    layer->norm2_scale = views[2];
    layer->norm2_bias = views[3];
    layer->query = views[4];
    layer->key = views[5];
    layer->value = views[6];
    layer->attention_out = views[7];
    layer->feed_forward_in = views[8];
    layer->feed_forward_in_bias = views[9];
    layer->feed_forward_out = views[10];
    layer->feed_forward_out_bias = views[11];
}

m3_status m3_flow_weights_bind(const m3_weight_stage *stage,
                               m3_flow_weights *weights,
                               m3_error *error)
{
    m3_flow_requirement_set requirements;
    const m3_tensor_view *views[M3_FLOW_WEIGHT_COUNT];
    m3_flow_weights built = {0};
    size_t layer;
    m3_status status;

    if (weights == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow weight output is required");
    }
    status = m3_flow_requirements(&requirements, error);
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
    built.time_projection = views[0];
    built.time_linear_in = views[1];
    built.time_linear_in_bias = views[2];
    built.time_linear_out = views[3];
    built.time_linear_out_bias = views[4];
    built.preprocess_convolution = views[5];
    built.input_projection = views[6];
    built.output_projection = views[7];
    built.postprocess_convolution = views[8];
    for (layer = 0U; layer < M3_FLOW_LAYER_COUNT; ++layer) {
        m3_flow_bind_layer(&built.layers[layer],
                           &views[9U + layer * 12U]);
    }
    *weights = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

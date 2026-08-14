/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_rvq_condition_internal.h"

#include <stdlib.h>
#include <string.h>

static int m3_binding_compare(const void *left_pointer,
                              const void *right_pointer)
{
    const m3_weight_binding *left = left_pointer;
    const m3_weight_binding *right = right_pointer;

    return strcmp(left->name, right->name);
}

static void m3_binding_stage(
    const m3_weight_requirement *requirements, size_t count,
    m3_weight_binding *bindings, m3_tensor_view *views,
    m3_weight_table *table, m3_weight_stage *stage)
{
    size_t index;

    (void)memset(bindings, 0, count * sizeof(*bindings));
    (void)memset(views, 0, count * sizeof(*views));
    (void)memset(table, 0, sizeof(*table));
    m3_weight_stage_init(stage);
    for (index = 0U; index < count; ++index) {
        bindings[index].name = (char *)requirements[index].name;
        bindings[index].tensor = requirements[index].tensor;
    }
    qsort(bindings, count, sizeof(*bindings), m3_binding_compare);
    for (index = 0U; index < count; ++index) {
        views[index].metadata = bindings[index].tensor;
    }
    table->bindings = bindings;
    table->binding_count = count;
    stage->table = table;
    stage->views = views;
    stage->view_count = count;
}

void m3_test_rvq_binding(m3_test_context *test)
{
    m3_rvq_requirement_set requirements;
    m3_weight_binding bindings[47];
    m3_tensor_view views[47];
    m3_weight_table table;
    m3_weight_stage stage;
    m3_rvq_weights weights = {0};
    m3_rvq_weights preserved;
    const m3_weight_binding *binding;
    m3_error error;
    bool ready = m3_rvq_requirements(&requirements, &error) ==
                 M3_STATUS_OK;

    M3_TEST_EXPECT(test, ready, "generate exact published RVQ requirements");
    if (!ready) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        requirements.count == 47U &&
            strcmp(requirements.requirements[0].name,
                   "audio_embeddings.weight") == 0 &&
            strcmp(requirements.requirements[1].name,
                   "projection.weight") == 0 &&
            strcmp(requirements.requirements[2].name,
                   "pos_embedding.weight") == 0 &&
            strcmp(requirements.requirements[3].name, "norm.weight") == 0 &&
            strcmp(requirements.requirements[4].name,
                   "layers.0.input_layernorm.weight") == 0 &&
            strcmp(requirements.requirements[39].name,
                   "layers.3.down_proj.weight") == 0 &&
            strcmp(requirements.requirements[40].name,
                   "audio_heads.0.weight") == 0 &&
            strcmp(requirements.requirements[46].name,
                   "audio_heads.6.weight") == 0,
        "RVQ requirements preserve published tensor order");
    M3_TEST_EXPECT(
        test,
        requirements.requirements[0].tensor.dtype == M3_DTYPE_BF16 &&
            requirements.requirements[0].tensor.rank == 2U &&
            requirements.requirements[0].tensor.shape[0] == 7168U &&
            requirements.requirements[0].tensor.shape[1] == 4096U &&
            requirements.requirements[2].tensor.shape[0] == 16U &&
            requirements.requirements[6].tensor.shape[0] == 4096U &&
            requirements.requirements[10].tensor.shape[0] == 6144U &&
            requirements.requirements[46].tensor.shape[0] == 1024U,
        "RVQ requirements retain exact published dtypes and shapes");
    m3_binding_stage(requirements.requirements, requirements.count,
                     bindings, views, &table, &stage);
    M3_TEST_EXPECT(
        test,
        m3_rvq_weights_bind(&stage, &weights, &error) ==
            M3_STATUS_OK,
        "bind all 47 exact staged RVQ tensors");
    binding = m3_weight_table_find(&table, "layers.2.attn.to_v.weight");
    M3_TEST_EXPECT(
        test,
        binding != NULL &&
            weights.layers[2].value ==
                &views[(size_t)(binding - table.bindings)] &&
            weights.heads[6] ==
                m3_weight_stage_find_view(&stage, "audio_heads.6.weight"),
        "RVQ binder maps schema order onto typed runtime slots");
    preserved = weights;
    binding = m3_weight_table_find(&table, "projection.weight");
    if (binding != NULL) {
        views[(size_t)(binding - table.bindings)].metadata.shape[0] += 1U;
    }
    M3_TEST_EXPECT(
        test,
        binding != NULL &&
            m3_rvq_weights_bind(&stage, &weights, NULL) ==
                M3_STATUS_INVALID_FORMAT &&
            memcmp(&weights, &preserved, sizeof(weights)) == 0,
        "RVQ bind failure preserves the complete published weight set");
}

void m3_test_condition_binding(m3_test_context *test)
{
    m3_condition_requirement_set requirements;
    m3_weight_binding bindings[4];
    m3_tensor_view views[4];
    m3_weight_table table;
    m3_weight_stage stage;
    m3_condition_weights weights = {0};
    m3_condition_weights preserved;
    const m3_weight_binding *binding;
    m3_error error;
    bool ready = m3_condition_requirements(&requirements, &error) ==
                 M3_STATUS_OK;

    M3_TEST_EXPECT(test, ready,
                   "generate exact published condition requirements");
    if (!ready) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        requirements.count == 4U &&
            strcmp(requirements.requirements[0].name,
                   "layer_weight_logits") == 0 &&
            strcmp(requirements.requirements[1].name,
                   "layer_scale") == 0 &&
            strcmp(requirements.requirements[2].name, "proj.weight") == 0 &&
            strcmp(requirements.requirements[3].name, "proj.bias") == 0 &&
            requirements.requirements[0].tensor.dtype == M3_DTYPE_F32 &&
            requirements.requirements[0].tensor.shape[0] == 8U &&
            requirements.requirements[2].tensor.rank == 3U &&
            requirements.requirements[2].tensor.shape[0] == 2048U &&
            requirements.requirements[2].tensor.shape[1] == 4096U &&
            requirements.requirements[2].tensor.shape[2] == 3U,
        "condition requirements retain published order and shapes");
    m3_binding_stage(requirements.requirements, requirements.count,
                     bindings, views, &table, &stage);
    M3_TEST_EXPECT(
        test,
        m3_condition_weights_bind(&stage, &weights, &error) ==
                M3_STATUS_OK &&
            weights.layer_weight_logits ==
                m3_weight_stage_find_view(&stage, "layer_weight_logits") &&
            weights.projection ==
                m3_weight_stage_find_view(&stage, "proj.weight"),
        "bind all four exact staged condition tensors");
    preserved = weights;
    binding = m3_weight_table_find(&table, "proj.bias");
    stage.view_count = 3U;
    M3_TEST_EXPECT(
        test,
        binding != NULL &&
            m3_condition_weights_bind(&stage, &weights, NULL) ==
                M3_STATUS_INVALID_FORMAT &&
            memcmp(&weights, &preserved, sizeof(weights)) == 0,
        "condition bind failure is atomic without an error sink");
}

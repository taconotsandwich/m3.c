/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_flow_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    uint8_t rank;
    uint64_t shape[3];
} m3_flow_expected_tensor;

static bool m3_flow_requirement_order_exact(
    const m3_flow_requirement_set *requirements)
{
    static const m3_flow_expected_tensor roots[] = {
        {"time_proj.weight", 2U, {128U, 1U, 0U}},
        {"time_embed.linear_1.weight", 2U, {2048U, 256U, 0U}},
        {"time_embed.linear_1.bias", 1U, {2048U, 0U, 0U}},
        {"time_embed.linear_2.weight", 2U, {2048U, 2048U, 0U}},
        {"time_embed.linear_2.bias", 1U, {2048U, 0U, 0U}},
        {"preprocess_conv.weight", 3U, {2304U, 2304U, 1U}},
        {"proj_in.weight", 2U, {2048U, 2304U, 0U}},
        {"proj_out.weight", 2U, {128U, 2048U, 0U}},
        {"postprocess_conv.weight", 3U, {128U, 128U, 1U}}
    };
    static const m3_flow_expected_tensor layer[] = {
        {"norm1.weight", 1U, {2048U, 0U, 0U}},
        {"norm1.bias", 1U, {2048U, 0U, 0U}},
        {"norm2.weight", 1U, {2048U, 0U, 0U}},
        {"norm2.bias", 1U, {2048U, 0U, 0U}},
        {"attn.to_q.weight", 2U, {2048U, 2048U, 0U}},
        {"attn.to_k.weight", 2U, {2048U, 2048U, 0U}},
        {"attn.to_v.weight", 2U, {2048U, 2048U, 0U}},
        {"attn.to_out.0.weight", 2U, {2048U, 2048U, 0U}},
        {"ff_in.weight", 2U, {16384U, 2048U, 0U}},
        {"ff_in.bias", 1U, {16384U, 0U, 0U}},
        {"ff_out.weight", 2U, {2048U, 8192U, 0U}},
        {"ff_out.bias", 1U, {2048U, 0U, 0U}}
    };
    size_t index;

    if (requirements->count != M3_FLOW_WEIGHT_COUNT) {
        return false;
    }
    for (index = 0U; index < sizeof(roots) / sizeof(roots[0]); ++index) {
        const m3_tensor_metadata *tensor =
            &requirements->requirements[index].tensor;
        uint8_t axis;

        if (strcmp(requirements->requirements[index].name,
                   roots[index].name) != 0 ||
            tensor->dtype != M3_DTYPE_F32 ||
            tensor->rank != roots[index].rank) {
            return false;
        }
        for (axis = 0U; axis < tensor->rank; ++axis) {
            if (tensor->shape[axis] != roots[index].shape[axis]) {
                return false;
            }
        }
    }
    for (index = 0U; index < 36U * 12U; ++index) {
        const size_t requirement_index = index + 9U;
        const m3_flow_expected_tensor *expected = &layer[index % 12U];
        const m3_tensor_metadata *tensor =
            &requirements->requirements[requirement_index].tensor;
        char name[128];
        uint8_t axis;

        (void)snprintf(name, sizeof(name), "transformer_blocks.%zu.%s",
                       index / 12U, expected->name);
        if (strcmp(requirements->requirements[requirement_index].name,
                   name) != 0 ||
            tensor->dtype != M3_DTYPE_F32 ||
            tensor->rank != expected->rank) {
            return false;
        }
        for (axis = 0U; axis < tensor->rank; ++axis) {
            if (tensor->shape[axis] != expected->shape[axis]) {
                return false;
            }
        }
    }
    return true;
}

static int m3_flow_binding_compare(const void *left_pointer,
                                   const void *right_pointer)
{
    const m3_weight_binding *left = left_pointer;
    const m3_weight_binding *right = right_pointer;

    return strcmp(left->name, right->name);
}

static void m3_flow_binding_stage(
    const m3_flow_requirement_set *requirements,
    m3_weight_binding bindings[M3_FLOW_WEIGHT_COUNT],
    m3_tensor_view views[M3_FLOW_WEIGHT_COUNT], m3_weight_table *table,
    m3_weight_stage *stage)
{
    size_t index;

    (void)memset(bindings, 0,
                 M3_FLOW_WEIGHT_COUNT * sizeof(*bindings));
    (void)memset(views, 0, M3_FLOW_WEIGHT_COUNT * sizeof(*views));
    (void)memset(table, 0, sizeof(*table));
    m3_weight_stage_init(stage);
    for (index = 0U; index < M3_FLOW_WEIGHT_COUNT; ++index) {
        bindings[index].name =
            (char *)requirements->requirements[index].name;
        bindings[index].tensor =
            requirements->requirements[index].tensor;
    }
    qsort(bindings, M3_FLOW_WEIGHT_COUNT, sizeof(*bindings),
          m3_flow_binding_compare);
    for (index = 0U; index < M3_FLOW_WEIGHT_COUNT; ++index) {
        views[index].metadata = bindings[index].tensor;
    }
    table->bindings = bindings;
    table->binding_count = M3_FLOW_WEIGHT_COUNT;
    stage->table = table;
    stage->views = views;
    stage->view_count = M3_FLOW_WEIGHT_COUNT;
}

void m3_test_flow_binding(m3_test_context *test)
{
    m3_flow_requirement_set requirements;
    m3_weight_binding bindings[M3_FLOW_WEIGHT_COUNT];
    m3_tensor_view views[M3_FLOW_WEIGHT_COUNT];
    m3_weight_table table;
    m3_weight_stage stage;
    m3_flow_weights weights = {0};
    m3_flow_weights preserved;
    const m3_weight_binding *binding;
    m3_error error;
    bool ready = m3_flow_requirements(&requirements, &error) ==
                 M3_STATUS_OK;

    M3_TEST_EXPECT(test, ready,
                   "generate all exact published flow requirements");
    if (!ready) {
        return;
    }
    M3_TEST_EXPECT(
        test, m3_flow_requirement_order_exact(&requirements),
        "all 441 flow names, dtypes, dimensions, and rows are exact");
    m3_flow_binding_stage(
        &requirements, bindings, views, &table, &stage);
    M3_TEST_EXPECT(
        test,
        m3_flow_weights_bind(&stage, &weights, &error) == M3_STATUS_OK,
        "atomically bind all 441 exact flow tensors");
    binding = m3_weight_table_find(
        &table, "transformer_blocks.35.attn.to_out.0.weight");
    M3_TEST_EXPECT(
        test,
        binding != NULL &&
            weights.layers[35].attention_out ==
                &views[(size_t)(binding - table.bindings)] &&
            weights.time_projection ==
                m3_weight_stage_find_view(&stage, "time_proj.weight"),
        "flow binder maps terminal and root tensors to typed slots");
    preserved = weights;
    binding = m3_weight_table_find(&table, "proj_out.weight");
    if (binding != NULL) {
        views[(size_t)(binding - table.bindings)].metadata.shape[0] += 1U;
    }
    M3_TEST_EXPECT(
        test,
        binding != NULL &&
            m3_flow_weights_bind(&stage, &weights, NULL) ==
                M3_STATUS_INVALID_FORMAT &&
            memcmp(&weights, &preserved, sizeof(weights)) == 0,
        "flow bind failure preserves every previously bound pointer");
}

static uint32_t m3_flow_f32_bits(float value)
{
    uint32_t bits;

    (void)memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void m3_test_flow_schedule_contract(m3_test_context *test)
{
    static const uint32_t timestep_bits[M3_FLOW_INFERENCE_STEPS] = {
        UINT32_C(0x00000000), UINT32_C(0x3d088890),
        UINT32_C(0x3d888888), UINT32_C(0x3dccccd0),
        UINT32_C(0x3e088888), UINT32_C(0x3e2aaaac),
        UINT32_C(0x3e4ccccc), UINT32_C(0x3e6eeef0),
        UINT32_C(0x3e888888), UINT32_C(0x3e99999a),
        UINT32_C(0x3eaaaaaa), UINT32_C(0x3ebbbbbc),
        UINT32_C(0x3ecccccc), UINT32_C(0x3eddddde),
        UINT32_C(0x3eeeeeee), UINT32_C(0x3f000000),
        UINT32_C(0x3f088888), UINT32_C(0x3f111111),
        UINT32_C(0x3f19999a), UINT32_C(0x3f222222),
        UINT32_C(0x3f2aaaaa), UINT32_C(0x3f333333),
        UINT32_C(0x3f3bbbbc), UINT32_C(0x3f444444),
        UINT32_C(0x3f4ccccd), UINT32_C(0x3f555555),
        UINT32_C(0x3f5dddde), UINT32_C(0x3f666666),
        UINT32_C(0x3f6eeeef), UINT32_C(0x3f777777)
    };
    m3_flow_config config;
    uint32_t step;
    float noise = 0.0F;
    float previous = 0.0F;
    bool exact = true;

    m3_flow_config_init(&config);
    for (step = 0U; step < M3_FLOW_INFERENCE_STEPS; ++step) {
        exact = exact &&
                m3_flow_f32_bits(m3_flow_timestep(
                    step, M3_FLOW_INFERENCE_STEPS)) ==
                    timestep_bits[step];
    }
    M3_TEST_EXPECT(
        test, exact,
        "30-step inverted FlowMatch schedule matches every oracle F32 bit");
    M3_TEST_EXPECT(
        test,
        m3_flow_f32_bits(m3_flow_timestep_delta(0U, 30U)) ==
                UINT32_C(0x3d088890) &&
            m3_flow_f32_bits(m3_flow_timestep_delta(1U, 30U)) ==
                UINT32_C(0x3d088880) &&
            m3_flow_f32_bits(m3_flow_timestep_delta(17U, 30U)) ==
                UINT32_C(0x3d088890) &&
            m3_flow_f32_bits(m3_flow_timestep_delta(29U, 30U)) ==
                UINT32_C(0x3d088890),
        "Euler deltas preserve the oracle's alternating rounded steps");
    m3_flow_blend_coefficients(
        m3_flow_timestep(29U, 30U), &noise, &previous);
    M3_TEST_EXPECT(
        test,
        m3_flow_f32_bits(noise) == UINT32_C(0x3d088990) &&
            m3_flow_f32_bits(previous) == UINT32_C(0x3f777777),
        "overlap coefficients prohibit contraction and retain 1e-6 bias");
    M3_TEST_EXPECT(
        test,
        config.latent_channels == M3_FLOW_LATENT_CHANNELS &&
            config.condition_dimension == M3_FLOW_CONDITION_DIMENSION &&
            config.layer_count == M3_FLOW_LAYER_COUNT &&
            config.attention_heads == M3_FLOW_ATTENTION_HEADS &&
            config.head_dimension == M3_FLOW_HEAD_DIMENSION &&
            config.feed_forward_dimension ==
                M3_FLOW_FEED_FORWARD_DIMENSION &&
            config.rotary_dimension == M3_FLOW_ROTARY_DIMENSION &&
            config.fourier_dimension == M3_FLOW_FOURIER_DIMENSION &&
            config.chunk_frames == 200U && config.chunk_hop == 100U &&
            config.carry_length == 172U &&
            config.inference_steps == 30U &&
            config.maximum_frames == 9000U &&
            config.guidance_scale == 1.7F,
        "public flow configuration is fixed to the released checkpoint");
}

void m3_test_flow_chunk_contract(m3_test_context *test)
{
    static const uint64_t frames[] = {
        1U, 100U, 101U, 200U, 201U, 300U, 301U
    };
    static const size_t counts[] = {1U, 1U, 1U, 1U, 2U, 2U, 3U};
    m3_flow_config config;
    m3_error error;
    size_t index;
    bool exact = true;
    uint64_t start = 0U;
    uint64_t length = 0U;
    uint64_t latent = 0U;
    uint64_t carry_start = 0U;
    uint64_t carry = 0U;

    m3_flow_config_init(&config);
    for (index = 0U; index < sizeof(frames) / sizeof(frames[0]); ++index) {
        size_t count = 0U;

        exact = exact &&
                m3_flow_chunk_count(
                    &config, frames[index], &count, &error) ==
                    M3_STATUS_OK &&
                count == counts[index];
    }
    M3_TEST_EXPECT(
        test, exact,
        "chunk starts lock boundaries 1,100,101,200,201,300,301");
    exact = m3_flow_chunk_window(
                &config, 301U, 0U, &start, &length, &error) ==
                M3_STATUS_OK &&
            start == 0U && length == 200U &&
            m3_flow_chunk_window(
                &config, 301U, 1U, &start, &length, &error) ==
                M3_STATUS_OK &&
            start == 100U && length == 200U &&
            m3_flow_chunk_window(
                &config, 301U, 2U, &start, &length, &error) ==
                M3_STATUS_OK &&
            start == 200U && length == 101U;
    M3_TEST_EXPECT(
        test, exact,
        "301 frames use exact 200-frame windows at 100-frame hops");
    M3_TEST_EXPECT(
        test,
        m3_condition_output_length(1U, 441U, 128U, &latent, &error) ==
                M3_STATUS_OK &&
            latent == 3U &&
            m3_condition_output_length(
                100U, 441U, 128U, &latent, &error) == M3_STATUS_OK &&
            latent == 344U &&
            m3_condition_output_length(
                101U, 441U, 128U, &latent, &error) == M3_STATUS_OK &&
            latent == 347U &&
            m3_condition_output_length(
                200U, 441U, 128U, &latent, &error) == M3_STATUS_OK &&
            latent == 689U,
        "condition lengths use exact floor F*441/128");
    M3_TEST_EXPECT(
        test,
        m3_flow_carry_window(
            &config, 689U, &carry_start, &carry, &error) ==
                M3_STATUS_OK &&
            carry_start == 345U && carry == 172U &&
            m3_flow_carry_window(
                &config, 344U, &carry_start, &carry, &error) ==
                M3_STATUS_OK &&
            carry_start == 0U && carry == 172U &&
            m3_flow_carry_window(
                &config, 172U, &carry_start, &carry, &error) ==
                M3_STATUS_OK &&
            carry_start == 0U && carry == 0U,
        "carry retains [L-344,L-172) with exact short-window clamps");
    M3_TEST_EXPECT(
        test,
        m3_flow_chunk_count(&config, 0U, &index, NULL) ==
            M3_STATUS_INVALID_ARGUMENT,
        "flow rejects an empty autoregressive frame sequence");
}

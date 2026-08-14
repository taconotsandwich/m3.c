/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"
#include "m3_test.h"
#include "qwen_runtime_fixture.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    m3_tensor_view views[13];
    m3_qwen_layer_workspace layer;
} m3_qwen_layer_test_workspace;

static bool m3_qwen_layer_test_tensor(
    m3_qwen_test_fixture *fixture, m3_tensor_view *view, uint8_t rank,
    const uint64_t *shape)
{
    uint16_t zeros[16] = {0};

    return m3_op_test_tensor(&fixture->tensors, view, M3_DTYPE_BF16, rank,
                             shape, zeros);
}

static bool m3_qwen_layer_test_workspace_build(
    m3_qwen_test_fixture *fixture, uint64_t sequence,
    const float *hidden, m3_qwen_layer_test_workspace *workspace)
{
    const uint64_t hidden_shape[] = {2U, sequence, 4U};
    const uint64_t key_shape[] = {2U, sequence, 2U};
    const uint64_t query_heads[] = {2U, 2U, sequence, 2U};
    const uint64_t key_heads[] = {2U, 1U, sequence, 2U};
    const uint64_t reordered[] = {2U, sequence, 2U, 2U};
    size_t index;

    (void)memset(workspace, 0, sizeof(*workspace));
    if (!m3_qwen_layer_test_tensor(fixture, &workspace->views[0], 3U,
                                   hidden_shape) ||
        !m3_qwen_layer_test_tensor(fixture, &workspace->views[1], 3U,
                                   hidden_shape) ||
        !m3_qwen_layer_test_tensor(fixture, &workspace->views[2], 3U,
                                   hidden_shape) ||
        !m3_qwen_layer_test_tensor(fixture, &workspace->views[3], 3U,
                                   key_shape) ||
        !m3_qwen_layer_test_tensor(fixture, &workspace->views[4], 3U,
                                   key_shape) ||
        !m3_qwen_layer_test_tensor(fixture, &workspace->views[5], 4U,
                                   query_heads) ||
        !m3_qwen_layer_test_tensor(fixture, &workspace->views[6], 4U,
                                   key_heads) ||
        !m3_qwen_layer_test_tensor(fixture, &workspace->views[7], 4U,
                                   query_heads) ||
        !m3_qwen_layer_test_tensor(fixture, &workspace->views[8], 4U,
                                   reordered) ||
        !m3_qwen_layer_test_tensor(fixture, &workspace->views[9], 3U,
                                   hidden_shape) ||
        !m3_qwen_layer_test_tensor(fixture, &workspace->views[10], 3U,
                                   hidden_shape) ||
        !m3_qwen_layer_test_tensor(fixture, &workspace->views[11], 3U,
                                   hidden_shape) ||
        !m3_qwen_layer_test_tensor(fixture, &workspace->views[12], 3U,
                                   hidden_shape)) {
        return false;
    }
    for (index = 0U; index < workspace->views[0].metadata.element_count;
         ++index) {
        m3_op_store_float(
            &workspace->views[0],
            m3_op_element_offset(&workspace->views[0], index),
            hidden[index]);
    }
    workspace->layer.hidden = &workspace->views[0];
    workspace->layer.normalized = &workspace->views[1];
    workspace->layer.query = &workspace->views[2];
    workspace->layer.key = &workspace->views[3];
    workspace->layer.value = &workspace->views[4];
    workspace->layer.rotated_query = &workspace->views[5];
    workspace->layer.rotated_key = &workspace->views[6];
    workspace->layer.attention = &workspace->views[7];
    workspace->layer.attention_reordered = &workspace->views[8];
    workspace->layer.projection = &workspace->views[9];
    workspace->layer.gate = &workspace->views[10];
    workspace->layer.up = &workspace->views[11];
    workspace->layer.down = &workspace->views[12];
    return true;
}

static void m3_qwen_layer_test_cache_zero(m3_qwen_test_fixture *fixture)
{
    size_t index;

    for (index = 0U; index < fixture->cache.workspace.count; ++index) {
        (void)memset(m3_storage_data(fixture->cache.workspace.storages[index]),
                     0, m3_storage_size(
                            fixture->cache.workspace.storages[index]));
    }
}

static bool m3_qwen_layer_test_run(
    m3_qwen_test_fixture *fixture,
    m3_qwen_layer_test_workspace *workspace, uint64_t position)
{
    m3_error error;

    return m3_qwen_layer_execute(
               &fixture->executor, &fixture->dimensions,
               &fixture->weights.layers[0], &workspace->layer,
               &fixture->cache.workspace.views[0],
               &fixture->cache.workspace.views[1], &fixture->rope.views[0],
               &fixture->rope.views[1], position, &error) == M3_STATUS_OK;
}

void m3_test_qwen_prefill_causal_offset(m3_test_context *test)
{
    static const float first[] = {
        1.0F, 0.5F, -0.5F, 2.0F,
        -1.0F, 1.0F, 0.25F, 0.5F,
        0.75F, -0.25F, 1.5F, -1.0F,
        0.5F, 1.0F, -1.5F, 0.25F
    };
    static const float changed_future[] = {
        1.0F, 0.5F, -0.5F, 2.0F,
        2.0F, -2.0F, 1.0F, -1.0F,
        0.75F, -0.25F, 1.5F, -1.0F,
        -1.0F, -0.5F, 2.0F, 1.5F
    };
    m3_qwen_test_fixture fixture;
    m3_qwen_layer_test_workspace left;
    m3_qwen_layer_test_workspace right;
    size_t channel;
    bool past_invariant = true;
    bool future_changed = false;

    if (!m3_qwen_test_fixture_init(&fixture, 4U) ||
        !m3_qwen_layer_test_workspace_build(&fixture, 2U, first, &left) ||
        !m3_qwen_layer_test_workspace_build(
            &fixture, 2U, changed_future, &right)) {
        M3_TEST_EXPECT(test, false, "initialize Qwen causal layer fixtures");
        m3_qwen_test_fixture_dispose(&fixture);
        return;
    }
    m3_qwen_layer_test_cache_zero(&fixture);
    if (!m3_qwen_layer_test_run(&fixture, &left, 0U)) {
        M3_TEST_EXPECT(test, false, "execute first causal prefill layer");
    }
    m3_qwen_layer_test_cache_zero(&fixture);
    if (!m3_qwen_layer_test_run(&fixture, &right, 0U)) {
        M3_TEST_EXPECT(test, false, "execute changed-future prefill layer");
    }
    for (channel = 0U; channel < 4U; ++channel) {
        past_invariant = past_invariant &&
            m3_qwen_test_bf16_at(left.layer.hidden, channel) ==
                m3_qwen_test_bf16_at(right.layer.hidden, channel) &&
            m3_qwen_test_bf16_at(left.layer.hidden, 8U + channel) ==
                m3_qwen_test_bf16_at(right.layer.hidden, 8U + channel);
        future_changed = future_changed ||
            m3_qwen_test_bf16_at(left.layer.hidden, 4U + channel) !=
                m3_qwen_test_bf16_at(right.layer.hidden, 4U + channel) ||
            m3_qwen_test_bf16_at(left.layer.hidden, 12U + channel) !=
                m3_qwen_test_bf16_at(right.layer.hidden, 12U + channel);
    }
    M3_TEST_EXPECT(test, past_invariant && future_changed,
                   "use causal offset zero so future tokens cannot alter past rows");
    m3_qwen_test_fixture_dispose(&fixture);
}

static void m3_qwen_layer_test_prior_value(
    m3_qwen_test_fixture *fixture, bool changed)
{
    m3_tensor_view *value = &fixture->cache.workspace.views[1];
    static const float replacement[] = {4.0F, -4.0F, -3.0F, 3.0F};
    size_t index;

    m3_qwen_layer_test_cache_zero(fixture);
    if (!changed) {
        return;
    }
    for (index = 0U; index < 4U; ++index) {
        m3_op_store_float(value, m3_op_element_offset(value, 4U + index),
                          replacement[index]);
    }
}

void m3_test_qwen_step_causal_offset(m3_test_context *test)
{
    static const float current[] = {
        1.25F, 0.5F, 0.75F, -0.5F,
        1.25F, 0.5F, 0.75F, -0.5F
    };
    m3_qwen_test_fixture fixture;
    m3_qwen_layer_test_workspace left;
    m3_qwen_layer_test_workspace right;
    size_t index;
    bool prior_visible = false;
    bool current_written = false;
    bool unused_untouched = true;

    if (!m3_qwen_test_fixture_init(&fixture, 4U) ||
        !m3_qwen_layer_test_workspace_build(&fixture, 1U, current, &left) ||
        !m3_qwen_layer_test_workspace_build(&fixture, 1U, current, &right)) {
        M3_TEST_EXPECT(test, false, "initialize Qwen step causal fixtures");
        m3_qwen_test_fixture_dispose(&fixture);
        return;
    }
    m3_qwen_layer_test_prior_value(&fixture, false);
    if (!m3_qwen_layer_test_run(&fixture, &left, 2U)) {
        M3_TEST_EXPECT(test, false, "execute step with original prior values");
    }
    m3_qwen_layer_test_prior_value(&fixture, true);
    if (!m3_qwen_layer_test_run(&fixture, &right, 2U)) {
        M3_TEST_EXPECT(test, false, "execute step with changed prior values");
    }
    for (index = 0U; index < 8U; ++index) {
        prior_visible = prior_visible ||
            m3_qwen_test_bf16_at(left.layer.hidden, index) !=
                m3_qwen_test_bf16_at(right.layer.hidden, index);
    }
    for (index = 8U; index < 12U; ++index) {
        current_written = current_written ||
            m3_qwen_test_bf16_at(&fixture.cache.workspace.views[0], index) !=
                0U;
        unused_untouched = unused_untouched &&
            m3_qwen_test_bf16_at(&fixture.cache.workspace.views[0],
                                  index + 4U) == 0U &&
            m3_qwen_test_bf16_at(&fixture.cache.workspace.views[1],
                                  index + 4U) == 0U;
    }
    M3_TEST_EXPECT(test, prior_visible,
                   "use published position two as causal offset for one-token step");
    M3_TEST_EXPECT(test, current_written && unused_untouched,
                   "write only current token-major cache slot at step position");
    m3_qwen_test_fixture_dispose(&fixture);
}

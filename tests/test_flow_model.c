/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "flow_runtime_test.h"
#include "rvq_condition_test.h"

#include <stdint.h>
#include <string.h>

static uint32_t m3_flow_model_bits(float value)
{
    uint32_t bits;

    (void)memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void m3_flow_model_identity(const m3_tensor_view *view)
{
    m3_tensor_view *mutable_view = (m3_tensor_view *)view;
    size_t rows = (size_t)view->metadata.shape[0];
    size_t columns = (size_t)view->metadata.shape[1];
    size_t row;
    size_t column;

    for (row = 0U; row < rows; ++row) {
        for (column = 0U; column < columns; ++column) {
            m3_rc_set(mutable_view, row * columns + column,
                      row == column ? 1.0F : 0.0F);
        }
    }
}

static void m3_flow_model_zero(const m3_tensor_view *view)
{
    m3_tensor_view *mutable_view = (m3_tensor_view *)view;
    size_t index;

    for (index = 0U; index < view->metadata.element_count; ++index) {
        m3_rc_set(mutable_view, index, 0.0F);
    }
}

void m3_test_flow_fourier_rotary(m3_test_context *test)
{
    m3_flow_test_fixture fixture;
    m3_flow_run run;
    const uint64_t fourier_shape[] = {2U, 4U};
    const uint64_t rotary_shape[] = {3U, 1U};
    m3_tensor_view fourier;
    m3_tensor_view cosines;
    m3_tensor_view sines;
    m3_error error;
    float timestep = m3_flow_timestep(1U, 30U);
    bool ready = m3_flow_test_fixture_init(&fixture, 2U);

    M3_TEST_EXPECT(test, ready,
                   "create reduced flow Fourier fixture");
    if (!ready) {
        return;
    }
    m3_rc_set((m3_tensor_view *)fixture.weights.time_projection, 0U,
              0.25F);
    m3_rc_set((m3_tensor_view *)fixture.weights.time_projection, 1U,
              -0.5F);
    ready = m3_flow_test_run_init(&fixture, 2U, &run);
    M3_TEST_EXPECT(test, ready,
                   "allocate reduced production flow workspace");
    if (!ready) {
        m3_flow_test_fixture_dispose(&fixture);
        return;
    }
    ready = m3_flow_prepare_tables(&run, 2U, timestep, &error) ==
                M3_STATUS_OK &&
            m3_flow_view(&run, M3_FLOW_WS_FOURIER, M3_DTYPE_F32, 2U,
                         fourier_shape, &fourier, &error) ==
                M3_STATUS_OK &&
            m3_flow_view(&run, M3_FLOW_WS_COSINES, M3_DTYPE_F32, 2U,
                         rotary_shape, &cosines, &error) ==
                M3_STATUS_OK &&
            m3_flow_view(&run, M3_FLOW_WS_SINES, M3_DTYPE_F32, 2U,
                         rotary_shape, &sines, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready,
                   "build learned Fourier and partial-RoPE tables");
    if (ready) {
        M3_TEST_EXPECT(
            test,
            m3_flow_model_bits(m3_rc_get(&fourier, 0U)) ==
                    UINT32_C(0x3f7fa62f) &&
                m3_flow_model_bits(m3_rc_get(&fourier, 1U)) ==
                    UINT32_C(0x3f7e98fc) &&
                m3_flow_model_bits(m3_rc_get(&fourier, 2U)) ==
                    UINT32_C(0x3d565e47) &&
                m3_flow_model_bits(m3_rc_get(&fourier, 3U)) ==
                    UINT32_C(0xbdd61311) &&
                memcmp(m3_storage_const_data(fourier.storage),
                       (const unsigned char *)m3_storage_const_data(
                           fourier.storage) +
                           4U * sizeof(float),
                       4U * sizeof(float)) == 0,
            "Fourier order is cos-half then sin-half for both CFG rows");
        M3_TEST_EXPECT(
            test,
            m3_flow_model_bits(m3_rc_get(&cosines, 0U)) ==
                    UINT32_C(0x3f800000) &&
                m3_flow_model_bits(m3_rc_get(&sines, 0U)) ==
                    UINT32_C(0x00000000) &&
                m3_flow_model_bits(m3_rc_get(&cosines, 1U)) ==
                    UINT32_C(0x3f0a5140) &&
                m3_flow_model_bits(m3_rc_get(&sines, 1U)) ==
                    UINT32_C(0x3f576aa4) &&
                m3_flow_model_bits(m3_rc_get(&cosines, 2U)) ==
                    UINT32_C(0xbed51133) &&
                m3_flow_model_bits(m3_rc_get(&sines, 2U)) ==
                    UINT32_C(0x3f68c7b7),
            "RoPE positions include the prepended time token at position zero");
    }
    m3_flow_test_run_dispose(&run);
    m3_flow_test_fixture_dispose(&fixture);
}

void m3_test_flow_layer_noncausal(m3_test_context *test)
{
    static const float hidden_values[12] = {
        1.0F, 2.0F, 3.0F, 4.0F,
        2.0F, -1.0F, 0.5F, 3.0F,
        -2.0F, 1.0F, 4.0F, -1.0F
    };
    static const float expected[12] = {
        -0.14428973F, 1.5383742F, 3.485278F, 5.120638F,
        2.2728846F, -2.189964F, 0.31555206F, 4.101527F,
        -2.9921727F, 1.0243497F, 5.2648945F, -1.2970712F
    };
    const uint64_t hidden_shape[] = {2U, 3U, 4U};
    const uint64_t heads_shape[] = {2U, 1U, 3U, 4U};
    m3_flow_test_fixture fixture;
    m3_flow_run run;
    m3_tensor_view hidden;
    m3_tensor_view query_rotary;
    m3_error error;
    size_t index;
    bool ready = m3_flow_test_fixture_init(&fixture, 2U);

    M3_TEST_EXPECT(test, ready,
                   "create reduced noncausal flow layer fixture");
    if (!ready) {
        return;
    }
    m3_flow_model_identity(fixture.weights.layers[0].query);
    m3_flow_model_identity(fixture.weights.layers[0].key);
    m3_flow_model_identity(fixture.weights.layers[0].value);
    m3_flow_model_identity(fixture.weights.layers[0].attention_out);
    ready = m3_flow_test_run_init(&fixture, 2U, &run) &&
            m3_flow_view(&run, M3_FLOW_WS_HIDDEN, M3_DTYPE_F32, 3U,
                         hidden_shape, &hidden, &error) == M3_STATUS_OK;
    for (index = 0U; ready && index < 12U; ++index) {
        m3_rc_set(&hidden, index, hidden_values[index]);
        m3_rc_set(&hidden, 12U + index, hidden_values[index]);
    }
    ready = ready &&
            m3_flow_prepare_tables(&run, 2U, 0.0F, &error) ==
                M3_STATUS_OK &&
            m3_flow_transformer_layer(
                &run, &fixture.weights.layers[0], 2U, &error) ==
                M3_STATUS_OK &&
            m3_flow_view(&run, M3_FLOW_WS_QUERY_ROTARY, M3_DTYPE_F32,
                         4U, heads_shape, &query_rotary, &error) ==
                M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready,
                   "execute the production flow transformer layer core");
    if (ready) {
        for (index = 0U; index < 12U; ++index) {
            M3_TEST_EXPECT_F32(
                test, m3_rc_get(&hidden, index), expected[index],
                3.0e-6F, 3.0e-6F,
                "flow layer matches affine-LN, attention, and residual oracle");
        }
        M3_TEST_EXPECT_F32(
            test, m3_rc_get(&query_rotary, 4U), 1.4917974F,
            2.0e-6F, 2.0e-6F,
            "partial RoPE rotates the first head pair");
        M3_TEST_EXPECT_F32(
            test, m3_rc_get(&query_rotary, 6U), -0.41239214F,
            2.0e-6F, 2.0e-6F,
            "partial RoPE leaves trailing head dimensions untouched");
        M3_TEST_EXPECT(
            test, m3_rc_get(&hidden, 0U) < 0.0F,
            "first token attends to later tokens without a causal mask");
    }
    m3_flow_test_run_dispose(&run);
    m3_flow_test_fixture_dispose(&fixture);
}

void m3_test_flow_layer_feed_forward(m3_test_context *test)
{
    const uint64_t hidden_shape[] = {2U, 2U, 4U};
    m3_flow_test_fixture fixture;
    m3_flow_run run;
    m3_tensor_view hidden;
    m3_error error;
    size_t token;
    bool ready = m3_flow_test_fixture_init(&fixture, 1U);

    M3_TEST_EXPECT(test, ready,
                   "create reduced flow feed-forward fixture");
    if (!ready) {
        return;
    }
    m3_rc_set((m3_tensor_view *)
                  fixture.weights.layers[0].feed_forward_in_bias,
              0U, 1.0F);
    m3_rc_set((m3_tensor_view *)
                  fixture.weights.layers[0].feed_forward_in_bias,
              1U, 2.0F);
    m3_rc_set((m3_tensor_view *)
                  fixture.weights.layers[0].feed_forward_in_bias,
              2U, 3.0F);
    m3_rc_set((m3_tensor_view *)
                  fixture.weights.layers[0].feed_forward_in_bias,
              3U, 100.0F);
    m3_rc_set((m3_tensor_view *)
                  fixture.weights.layers[0].feed_forward_in_bias,
              4U, 100.0F);
    m3_rc_set((m3_tensor_view *)
                  fixture.weights.layers[0].feed_forward_in_bias,
              5U, 100.0F);
    m3_rc_set((m3_tensor_view *)
                  fixture.weights.layers[0].feed_forward_out,
              0U, 1.0F);
    m3_rc_set((m3_tensor_view *)
                  fixture.weights.layers[0].feed_forward_out,
              4U, 1.0F);
    m3_rc_set((m3_tensor_view *)
                  fixture.weights.layers[0].feed_forward_out,
              8U, 1.0F);
    m3_rc_set((m3_tensor_view *)
                  fixture.weights.layers[0].feed_forward_out_bias,
              0U, 1.0F);
    m3_rc_set((m3_tensor_view *)
                  fixture.weights.layers[0].feed_forward_out_bias,
              1U, 2.0F);
    m3_rc_set((m3_tensor_view *)
                  fixture.weights.layers[0].feed_forward_out_bias,
              2U, 3.0F);
    m3_rc_set((m3_tensor_view *)
                  fixture.weights.layers[0].feed_forward_out_bias,
              3U, 4.0F);
    ready = m3_flow_test_run_init(&fixture, 1U, &run) &&
            m3_flow_view(&run, M3_FLOW_WS_HIDDEN, M3_DTYPE_F32, 3U,
                         hidden_shape, &hidden, &error) == M3_STATUS_OK;
    for (token = 0U; ready && token < 4U; ++token) {
        m3_rc_set(&hidden, token * 4U, 10.0F);
        m3_rc_set(&hidden, token * 4U + 1U, 20.0F);
        m3_rc_set(&hidden, token * 4U + 2U, 30.0F);
        m3_rc_set(&hidden, token * 4U + 3U, 40.0F);
    }
    ready = ready &&
            m3_flow_prepare_tables(&run, 1U, 0.0F, &error) ==
                M3_STATUS_OK &&
            m3_flow_transformer_layer(
                &run, &fixture.weights.layers[0], 1U, &error) ==
                M3_STATUS_OK;
    for (token = 0U; token < 4U; ++token) {
        M3_TEST_EXPECT(
            test,
            ready && m3_rc_get(&hidden, token * 4U) == 111.0F &&
                m3_rc_get(&hidden, token * 4U + 1U) == 222.0F &&
                m3_rc_get(&hidden, token * 4U + 2U) == 333.0F &&
                m3_rc_get(&hidden, token * 4U + 3U) == 44.0F,
            "flow block splits first states and second gates before FF residual");
    }
    m3_flow_test_run_dispose(&run);
    m3_flow_test_fixture_dispose(&fixture);
}

void m3_test_flow_cfg_euler_blend(m3_test_context *test)
{
    const uint64_t pair_shape[] = {2U, 1U, 2U};
    const uint64_t single_shape[] = {1U, 1U, 2U};
    m3_flow_test_fixture fixture;
    m3_flow_run run;
    m3_tensor_view velocity;
    m3_tensor_view latent;
    m3_tensor_view noise;
    m3_tensor_view previous;
    m3_error error;
    bool ready = m3_flow_test_fixture_init(&fixture, 1U) &&
                 m3_flow_test_run_init(&fixture, 1U, &run);

    M3_TEST_EXPECT(test, ready,
                   "create reduced CFG and Euler workspace");
    if (!ready) {
        m3_flow_test_fixture_dispose(&fixture);
        return;
    }
    ready = m3_flow_view(&run, M3_FLOW_WS_VELOCITY, M3_DTYPE_F32, 3U,
                         pair_shape, &velocity, &error) == M3_STATUS_OK;
    if (ready) {
        m3_rc_set(&velocity, 0U, 5432.875F);
        m3_rc_set(&velocity, 1U, 1.0F);
        m3_rc_set(&velocity, 2U, -12345.625F);
        m3_rc_set(&velocity, 3U, -1.0F);
        ready = m3_flow_guided_velocity(
                    &run, &velocity, 1U, &error) == M3_STATUS_OK;
    }
    M3_TEST_EXPECT(
        test,
        ready && m3_flow_model_bits(m3_rc_get(&velocity, 0U)) ==
                     UINT32_C(0x468baba7),
        "CFG uses row0 conditional and row1 unconditional with stored rounding");
    ready = ready &&
            m3_flow_view(&run, M3_FLOW_WS_LATENT, M3_DTYPE_F32, 3U,
                         single_shape, &latent, &error) == M3_STATUS_OK;
    if (ready) {
        m3_rc_set(&velocity, 0U, -0.1259765625F);
        m3_rc_set(&velocity, 1U, 0.0F);
        m3_rc_set(&latent, 0U, 123456.75F);
        m3_rc_set(&latent, 1U, 0.0F);
        ready = m3_flow_euler_step(
                    &run, &latent, &velocity, 1U,
                    m3_flow_timestep_delta(0U, 30U), &error) ==
                M3_STATUS_OK;
    }
    M3_TEST_EXPECT(
        test,
        ready && m3_flow_model_bits(m3_rc_get(&latent, 0U)) ==
                     UINT32_C(0x47f1205f),
        "Euler update stores velocity*delta before adding the sample");
    ready = ready &&
            m3_flow_view(&run, M3_FLOW_WS_NOISE_PROMPT, M3_DTYPE_F32,
                         3U, single_shape, &noise, &error) ==
                M3_STATUS_OK &&
            m3_flow_view(&run, M3_FLOW_WS_PREVIOUS_LATENT, M3_DTYPE_F32,
                         3U, single_shape, &previous, &error) ==
                M3_STATUS_OK;
    if (ready) {
        m3_rc_set(&noise, 0U, 12345.25F);
        m3_rc_set(&noise, 1U, 0.0F);
        m3_rc_set(&previous, 0U, -2345.75F);
        m3_rc_set(&previous, 1U, 0.0F);
        ready = m3_flow_blend_overlap(
                    &run, &latent, 1U, 1U,
                    m3_flow_timestep(29U, 30U), &error) ==
                M3_STATUS_OK;
    }
    M3_TEST_EXPECT(
        test,
        ready && m3_flow_model_bits(m3_rc_get(&latent, 0U)) ==
                     UINT32_C(0xc4e80136),
        "overlap blend keeps separate rounded products and addition");
    m3_flow_test_run_dispose(&run);
    m3_flow_test_fixture_dispose(&fixture);
}

void m3_test_flow_model_rows_and_carry(m3_test_context *test)
{
    const uint64_t shape[] = {1U, 2U, 2U};
    const uint64_t pair_shape[] = {2U, 2U, 2U};
    const uint64_t carry_source_shape[] = {1U, 4U, 2U};
    const uint64_t carry_shape[] = {1U, 1U, 2U};
    m3_flow_test_fixture fixture;
    m3_flow_run run;
    m3_tensor_view latent;
    m3_tensor_view condition;
    m3_tensor_view velocity;
    m3_tensor_view carry_latent;
    m3_tensor_view carry_condition;
    m3_tensor_view previous_latent;
    m3_tensor_view previous_condition;
    uint64_t carry_length = 0U;
    m3_error error;
    bool ready = m3_flow_test_fixture_init(&fixture, 2U);

    M3_TEST_EXPECT(test, ready,
                   "create reduced flow row-order fixture");
    if (!ready) {
        return;
    }
    m3_flow_model_zero(fixture.weights.input_projection);
    m3_flow_model_zero(fixture.weights.output_projection);
    m3_rc_set((m3_tensor_view *)fixture.weights.input_projection, 0U,
              1.0F);
    m3_rc_set((m3_tensor_view *)fixture.weights.input_projection, 7U,
              1.0F);
    m3_rc_set((m3_tensor_view *)fixture.weights.input_projection, 16U,
              1.0F);
    m3_rc_set((m3_tensor_view *)fixture.weights.input_projection, 23U,
              1.0F);
    m3_rc_set((m3_tensor_view *)fixture.weights.output_projection, 0U,
              1.0F);
    m3_rc_set((m3_tensor_view *)fixture.weights.output_projection, 6U,
              1.0F);
    ready = m3_flow_test_run_init(&fixture, 4U, &run) &&
            m3_rc_tensor(&fixture.fixture, &latent, M3_DTYPE_F32, 3U,
                         shape, 0.0F) &&
            m3_rc_tensor(&fixture.fixture, &condition, M3_DTYPE_F32, 3U,
                         shape, 0.0F);
    if (ready) {
        m3_rc_set(&latent, 0U, 1.0F);
        m3_rc_set(&latent, 1U, 10.0F);
        m3_rc_set(&latent, 2U, 2.0F);
        m3_rc_set(&latent, 3U, 20.0F);
        m3_rc_set(&condition, 0U, 3.0F);
        m3_rc_set(&condition, 1U, 30.0F);
        m3_rc_set(&condition, 2U, 4.0F);
        m3_rc_set(&condition, 3U, 40.0F);
        ready = m3_flow_forward(
                    &run, &latent, &condition, 2U, 0.0F, &velocity,
                    &error) == M3_STATUS_OK &&
                velocity.metadata.shape[0] == pair_shape[0];
    }
    M3_TEST_EXPECT(
        test,
        ready && m3_rc_get(&velocity, 0U) == 1.0F &&
            m3_rc_get(&velocity, 1U) == 3.0F &&
            m3_rc_get(&velocity, 4U) == 1.0F &&
            m3_rc_get(&velocity, 5U) == 0.0F,
        "model concat keeps conditional row0, unconditional row1, and zero branch");
    if (ready) {
        ready = m3_flow_guided_velocity(
                    &run, &velocity, 2U, &error) == M3_STATUS_OK;
    }
    M3_TEST_EXPECT_F32(
        test, ready ? m3_rc_get(&velocity, 1U) : 0.0F, 5.1F,
        1.0e-6F, 1.0e-6F,
        "guided model output uses uncond+1.7*(cond-uncond)");
    ready = ready &&
            m3_rc_tensor(&fixture.fixture, &carry_latent, M3_DTYPE_F32,
                         3U, carry_source_shape, 0.0F) &&
            m3_rc_tensor(&fixture.fixture, &carry_condition,
                         M3_DTYPE_F32, 3U, carry_source_shape, 0.0F);
    if (ready) {
        m3_rc_set(&carry_latent, 4U, 7.0F);
        m3_rc_set(&carry_latent, 5U, 8.0F);
        m3_rc_set(&carry_condition, 4U, 9.0F);
        m3_rc_set(&carry_condition, 5U, 10.0F);
        ready = m3_flow_preserve_carry(
                    &run, &carry_latent, &carry_condition, 4U,
                    &carry_length, &error) == M3_STATUS_OK &&
                m3_flow_view(&run, M3_FLOW_WS_PREVIOUS_LATENT,
                             M3_DTYPE_F32, 3U, carry_shape,
                             &previous_latent, &error) == M3_STATUS_OK &&
                m3_flow_view(&run, M3_FLOW_WS_PREVIOUS_CONDITION,
                             M3_DTYPE_F32, 3U, carry_shape,
                             &previous_condition, &error) == M3_STATUS_OK;
    }
    if (ready) {
        m3_rc_set(&carry_latent, 4U, 99.0F);
        m3_rc_set(&carry_condition, 4U, 99.0F);
    }
    M3_TEST_EXPECT(
        test,
        ready && carry_length == 1U &&
            m3_rc_get(&previous_latent, 0U) == 7.0F &&
            m3_rc_get(&previous_condition, 0U) == 9.0F,
        "latent and condition carry are copied into flow-owned storage");
    m3_flow_test_run_dispose(&run);
    m3_flow_test_fixture_dispose(&fixture);
}

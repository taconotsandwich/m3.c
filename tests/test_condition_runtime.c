/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_rvq_condition_internal.h"
#include "rvq_condition_test.h"
#include "weight_stage_fixture.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static void m3_condition_expected(const m3_condition_test_fixture *fixture,
                                  float expected[4], float mixed[3])
{
    float scaled[3];
    float convolution[3];
    size_t frame;

    for (frame = 0U; frame < 3U; ++frame) {
        mixed[frame] = m3_condition_test_mixed(
            &fixture->frames, frame);
        scaled[frame] = mixed[frame] *
                        m3_rc_get(fixture->weights.layer_scale, 0U);
    }
    for (frame = 0U; frame < 3U; ++frame) {
        float sum = m3_rc_get(fixture->weights.bias, 0U);

        if (frame > 0U) {
            sum = sum +
                  scaled[frame - 1U] *
                      m3_rc_get(fixture->weights.projection, 0U);
        }
        sum = sum + scaled[frame] *
                        m3_rc_get(fixture->weights.projection, 1U);
        if (frame + 1U < 3U) {
            sum = sum +
                  scaled[frame + 1U] *
                      m3_rc_get(fixture->weights.projection, 2U);
        }
        convolution[frame] = sum;
    }
    expected[0] = convolution[0];
    expected[1] = convolution[0];
    expected[2] = convolution[1];
    expected[3] = convolution[2];
}

void m3_test_condition_numeric_resize(m3_test_context *test)
{
    m3_condition_test_fixture fixture;
    m3_runtime_progress_log progress = {0};
    m3_condition_output output;
    m3_condition_output short_output;
    m3_tensor_view one_frame;
    float expected[4];
    float mixed[3];
    float preserved[4];
    m3_runtime_tensor_spec rvq_specs[M3_RVQ_WORKSPACE_COUNT];
    m3_runtime_workspace rvq_workspace;
    const m3_rvq_config rvq_config = {
        2U, 1U, 1U, 2U, 16U, M3_DTYPE_BF16
    };
    uint64_t length = 0U;
    const uint64_t one_shape[] = {1U, 1U, 8U, 1U};
    m3_condition_config short_config;
    m3_error error;
    m3_status encode_status;
    size_t index;
    bool ready = m3_condition_test_fixture_init(&fixture);

    M3_TEST_EXPECT(test, ready,
                   "create strided BF16 condition runtime fixture");
    if (!ready) {
        return;
    }
    progress.cancel_at = UINT64_MAX;
    m3_condition_output_init(&output);
    m3_condition_output_init(&short_output);
    m3_runtime_workspace_init(&rvq_workspace);
    m3_condition_expected(&fixture, expected, mixed);
    encode_status = m3_condition_encode_core(
        fixture.fixture.backend, &fixture.config, &fixture.weights,
        &fixture.frames, m3_rc_progress, &progress, &output, &error);
    M3_TEST_EXPECT(
        test, encode_status == M3_STATUS_OK,
        "execute BF16-to-F32 mix, scale, convolution, and resize schedule");
    if (encode_status != M3_STATUS_OK) {
        m3_condition_output_dispose(&output);
        m3_condition_test_fixture_dispose(&fixture);
        return;
    }
    M3_TEST_EXPECT(
        test,
        output.storage != NULL && output.tensor.metadata.dtype == M3_DTYPE_F32 &&
            output.tensor.metadata.rank == 3U &&
            output.tensor.metadata.shape[0] == 1U &&
            output.tensor.metadata.shape[1] == 4U &&
            output.tensor.metadata.shape[2] == 1U,
        "condition encoder publishes owned contiguous [1,4,1] F32 output");
    for (index = 0U; index < 4U; ++index) {
        preserved[index] = m3_rc_get(&output.tensor, index);
        M3_TEST_EXPECT_F32(
            test, preserved[index], expected[index], 1.0e-6F, 1.0e-6F,
            "condition numeric oracle matches ordered mix, conv, and nearest resize");
    }
    {
        float paired =
            m3_rc_get(&fixture.frames, 0U) * 0.125F +
            m3_rc_get(&fixture.frames, 4U) * 0.125F;

        for (index = 1U; index < 8U; ++index) {
            if (index != 4U) {
                paired = paired +
                         m3_rc_get(&fixture.frames, index) * 0.125F;
            }
        }
        M3_TEST_EXPECT(
            test,
            mixed[0] == 0.375F && paired == 0.75F &&
                mixed[0] != paired &&
                m3_rc_get(&fixture.frames, 0U) != 1.0e8F,
            "eight-term oracle locks ascending F32 accumulation after BF16 cast");
    }
    M3_TEST_EXPECT(
        test,
        progress.call_count == 11U && progress.completed[0] == 0U &&
            progress.completed[10] == 10U && progress.total[0] == 10U &&
            progress.total[10] == 10U,
        "condition progress covers eight mixes, projection, and resize");
    m3_rvq_workspace_specs(&rvq_config, rvq_specs);
    ready = m3_runtime_workspace_build(
                &rvq_workspace, fixture.fixture.backend, rvq_specs,
                M3_RVQ_WORKSPACE_COUNT, &error) == M3_STATUS_OK;
    for (index = 0U; index < M3_RVQ_WORKSPACE_COUNT && ready; ++index) {
        const uint64_t clobber = UINT64_MAX;

        ready = m3_storage_write(
                    rvq_workspace.storages[index], 0U, &clobber,
                    sizeof(clobber), &error) == M3_STATUS_OK;
    }
    m3_runtime_workspace_dispose(&rvq_workspace);
    M3_TEST_EXPECT(
        test, ready && m3_rc_get(&output.tensor, 0U) == preserved[0],
        "owned condition output survives transient RVQ workspace reuse");
    M3_TEST_EXPECT(
        test,
        m3_condition_output_length(1U, 441U, 128U, &length, &error) ==
                M3_STATUS_OK &&
            length == 3U &&
            m3_condition_output_length(127U, 441U, 128U, &length,
                                       &error) == M3_STATUS_OK &&
            length == 437U &&
            m3_condition_output_length(128U, 441U, 128U, &length,
                                       &error) == M3_STATUS_OK &&
            length == 441U,
        "published floor resize uses the exact reduced ratio 441/128");
    ready = m3_rc_tensor(&fixture.fixture, &one_frame, M3_DTYPE_BF16, 4U,
                         one_shape, 1.0F);
    M3_TEST_EXPECT(test, ready, "create one-frame condition boundary input");
    short_config = fixture.config;
    short_config.resize_numerator = 1U;
    short_config.resize_denominator = 2U;
    if (ready) {
        M3_TEST_EXPECT(
            test,
            m3_condition_encode_core(
                fixture.fixture.backend, &short_config, &fixture.weights,
                &one_frame, NULL, NULL, &short_output, &error) ==
                    M3_STATUS_OK &&
                short_output.tensor.metadata.shape[1] == 1U,
            "F=1 schedule clamps a sub-unit floor resize to one output");
    }
    m3_condition_output_dispose(&short_output);
    m3_condition_output_dispose(&output);
    m3_runtime_workspace_dispose(&rvq_workspace);
    m3_condition_test_fixture_dispose(&fixture);
}

void m3_test_condition_atomic_validation(m3_test_context *test)
{
    m3_condition_test_fixture fixture;
    m3_runtime_progress_log cancel = {0};
    m3_condition_output output;
    m3_condition_weights bad_weights;
    m3_condition_config bad_config;
    m3_tensor_view wrong_dtype;
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    m3_storage *preserved_storage;
    float preserved_value;
    const uint64_t frame_shape[] = {1U, 3U, 8U, 1U};
    const size_t frame_strides[] = {192U, 64U, 8U, 4U};
    m3_error error;
    bool ready = m3_condition_test_fixture_init(&fixture);

    M3_TEST_EXPECT(test, ready,
                   "create condition atomic-validation fixture");
    if (!ready) {
        return;
    }
    m3_condition_output_init(&output);
    ready = m3_condition_encode_core(
                fixture.fixture.backend, &fixture.config, &fixture.weights,
                &fixture.frames, NULL, NULL, &output, &error) ==
            M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "establish published condition output");
    if (!ready) {
        m3_condition_test_fixture_dispose(&fixture);
        return;
    }
    preserved_storage = output.storage;
    preserved_value = m3_rc_get(&output.tensor, 0U);
    (void)m3_backend_get_allocation_stats(
        fixture.fixture.backend, &before, &error);
    cancel.cancel_at = 4U;
    M3_TEST_EXPECT(
        test,
        m3_condition_encode_core(
            fixture.fixture.backend, &fixture.config, &fixture.weights,
            &fixture.frames, m3_rc_progress, &cancel, &output, &error) ==
                M3_STATUS_CANCELLED &&
            cancel.call_count == 5U && output.storage == preserved_storage &&
            m3_rc_get(&output.tensor, 0U) == preserved_value &&
            m3_backend_get_allocation_stats(
                fixture.fixture.backend, &after, &error) == M3_STATUS_OK &&
            after.live_allocated_bytes == before.live_allocated_bytes &&
            after.live_storage_count == before.live_storage_count,
        "condition cancellation cleans new state and preserves old output");
    bad_config = fixture.config;
    bad_config.resize_denominator = 0U;
    M3_TEST_EXPECT(
        test,
        m3_condition_encode_core(
            fixture.fixture.backend, &bad_config, &fixture.weights,
            &fixture.frames, NULL, NULL, &output, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            output.storage == preserved_storage,
        "invalid resize configuration is atomic without an error sink");
    bad_weights = fixture.weights;
    bad_weights.bias = fixture.weights.layer_weight_logits;
    m3_tensor_view_init(&wrong_dtype);
    ready = m3_tensor_view_strided(
                &wrong_dtype, fixture.frames.storage, M3_DTYPE_F32, 4U,
                frame_shape, frame_strides, 0U, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "create wrong-dtype condition input");
    M3_TEST_EXPECT(
        test,
        m3_condition_encode_core(
            fixture.fixture.backend, &fixture.config, &bad_weights,
            &fixture.frames, NULL, NULL, &output, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            output.storage == preserved_storage &&
            m3_condition_encode_core(
                fixture.fixture.backend, &fixture.config, &fixture.weights,
                fixture.weights.layer_weight_logits, NULL, NULL, &output,
                NULL) == M3_STATUS_INVALID_ARGUMENT &&
            output.storage == preserved_storage && ready &&
            m3_condition_encode_core(
                fixture.fixture.backend, &fixture.config, &fixture.weights,
                &wrong_dtype, NULL, NULL, &output, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            output.storage == preserved_storage,
        "invalid condition weights, shape, and dtype preserve state");
    M3_TEST_EXPECT(
        test,
        m3_condition_output_length(
            UINT64_MAX, 2U, 1U, &(uint64_t){9U}, NULL) ==
            M3_STATUS_OVERFLOW,
        "condition length arithmetic rejects overflow without an error sink");
    m3_condition_output_dispose(&output);
    m3_condition_test_fixture_dispose(&fixture);
}

void m3_test_rvq_condition_preflight_limits(m3_test_context *test)
{
    m3_backend *backend = NULL;
    m3_weight_stage_fake_context *context = NULL;
    m3_storage *live = NULL;
    m3_runtime_tensor_spec spec = {
        M3_DTYPE_F32, 1U, {10U}, 16U
    };
    m3_error error;
    bool ready = m3_weight_stage_test_fake_backend_create(
        128U, 100U, 0U, &backend, &context, &error);

    M3_TEST_EXPECT(test, ready, "create bounded runtime preflight backend");
    if (!ready) {
        return;
    }
    ready = m3_storage_allocate(
                backend, 32U, 16U, &live, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "establish live runtime allocation");
    if (!ready) {
        m3_backend_free(backend);
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_rvq_condition_preflight(
            backend, 40U, &spec, 1U, &error) ==
                M3_STATUS_OUT_OF_MEMORY &&
            context->allocation_calls == 1U,
        "preflight counts live plus owned output plus complete workspace");
    spec.shape[0] = 33U;
    M3_TEST_EXPECT(
        test,
        m3_rvq_condition_preflight(
            backend, 16U, &spec, 1U, &error) ==
                M3_STATUS_OUT_OF_RANGE &&
            m3_rvq_condition_preflight(
                backend, 129U, NULL, 0U, NULL) ==
                M3_STATUS_OUT_OF_RANGE &&
            context->allocation_calls == 1U,
        "preflight rejects every over-limit storage before allocation");
    spec.shape[0] = 10U;
    M3_TEST_EXPECT(
        test,
        m3_rvq_condition_preflight(
            backend, 16U, &spec, 1U, &error) == M3_STATUS_OK &&
            context->allocation_calls == 1U,
        "preflight accepts an aggregate within both backend limits");
    m3_storage_free(live);
    m3_backend_free(backend);
}

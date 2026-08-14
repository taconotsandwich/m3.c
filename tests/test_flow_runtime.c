/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "flow_runtime_test.h"
#include "rvq_condition_test.h"
#include "weight_stage_fixture.h"

#include <stdint.h>
#include <string.h>

static bool m3_flow_rng_equal(const m3_rng *left, const m3_rng *right)
{
    return left->state == right->state &&
           left->increment == right->increment &&
           left->spare_normal == right->spare_normal &&
           left->has_spare_normal == right->has_spare_normal;
}

static bool m3_flow_runtime_layout(const m3_flow_output *output)
{
    static const uint64_t lengths[] = {4U, 4U, 3U};
    size_t index;

    if (output == NULL || output->chunk_count != 3U ||
        output->storages == NULL || output->chunks == NULL) {
        return false;
    }
    for (index = 0U; index < 3U; ++index) {
        const m3_tensor_view *chunk = &output->chunks[index];

        if (output->storages[index] == NULL ||
            chunk->storage != output->storages[index] ||
            chunk->metadata.dtype != M3_DTYPE_F32 ||
            chunk->metadata.rank != 3U ||
            chunk->metadata.shape[0] != 1U ||
            chunk->metadata.shape[1] != 2U ||
            chunk->metadata.shape[2] != lengths[index] ||
            !m3_tensor_is_contiguous(chunk)) {
            return false;
        }
    }
    return true;
}

static bool m3_flow_runtime_values(const m3_flow_output *output,
                                   const float normals[22])
{
    const m3_tensor_view *first;
    const m3_tensor_view *middle;
    const m3_tensor_view *last;
    size_t index;

    if (!m3_flow_runtime_layout(output)) {
        return false;
    }
    first = &output->chunks[0];
    middle = &output->chunks[1];
    last = &output->chunks[2];
    for (index = 0U; index < 8U; ++index) {
        if (m3_rc_get(first, index) != normals[index]) {
            return false;
        }
    }
    if (m3_rc_get(middle, 0U) != m3_rc_get(first, 2U) ||
        m3_rc_get(middle, 4U) != m3_rc_get(first, 6U) ||
        m3_rc_get(last, 0U) != m3_rc_get(middle, 2U) ||
        m3_rc_get(last, 3U) != m3_rc_get(middle, 6U)) {
        return false;
    }
    for (index = 1U; index < 4U; ++index) {
        if (m3_rc_get(middle, index) != normals[8U + index] ||
            m3_rc_get(middle, 4U + index) != normals[12U + index]) {
            return false;
        }
    }
    return m3_rc_get(last, 1U) == normals[17U] &&
           m3_rc_get(last, 2U) == normals[18U] &&
           m3_rc_get(last, 4U) == normals[20U] &&
           m3_rc_get(last, 5U) == normals[21U];
}

static bool m3_flow_reference_normals(m3_rng *rng, float values[22])
{
    m3_error error;

    return m3_rng_normal_f32_fill(rng, values, 8U, &error) ==
               M3_STATUS_OK &&
           m3_rng_normal_f32_fill(rng, values + 8U, 8U, &error) ==
               M3_STATUS_OK &&
           m3_rng_normal_f32_fill(rng, values + 16U, 6U, &error) ==
               M3_STATUS_OK;
}

void m3_test_flow_runtime_chunks(m3_test_context *test)
{
    m3_flow_test_fixture fixture;
    m3_runtime_progress_log progress = {0};
    m3_tensor_view strided_frames;
    m3_flow_output output;
    m3_flow_run clobber;
    m3_rng rng = {0};
    m3_rng reference = {0};
    float normals[22];
    float preserved = 0.0F;
    m3_error error;
    size_t index;
    bool ready = m3_flow_test_fixture_init(&fixture, 7U);

    M3_TEST_EXPECT(test, ready, "create reduced flow synthesis fixture");
    if (!ready) {
        return;
    }
    ready = m3_flow_test_strided_frames(&fixture, 7U, &strided_frames) &&
            m3_rng_seed(&rng, UINT64_C(7123), UINT64_C(91), &error) ==
                M3_STATUS_OK;
    reference = rng;
    ready = ready && m3_flow_reference_normals(&reference, normals);
    M3_TEST_EXPECT(
        test, ready,
        "create strided BF16 frames and exact chunk-local RNG oracle");
    if (!ready) {
        m3_flow_test_fixture_dispose(&fixture);
        return;
    }
    progress.cancel_at = UINT64_MAX;
    m3_flow_output_init(&output);
    ready = m3_flow_synthesize_core(
                fixture.fixture.backend, &fixture.config,
                &fixture.weights, &fixture.condition_weights,
                &strided_frames, &rng, m3_rc_progress, &progress,
                &output, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(
        test, ready,
        "run the production flow path at reduced fixture dimensions");
    M3_TEST_EXPECT(
        test,
        ready && progress.call_count == 7U &&
            progress.completed[0] == 0U &&
            progress.completed[6] == 6U &&
            progress.total[0] == 6U && progress.total[6] == 6U,
        "flow progress publishes validation then every Euler step");
    M3_TEST_EXPECT(
        test, ready && m3_flow_runtime_values(&output, normals),
        "first, carried middle, and final chunks match RNG draw order");
    M3_TEST_EXPECT(
        test, ready && m3_flow_rng_equal(&rng, &reference),
        "successful synthesis publishes the exactly advanced RNG state");
    if (ready) {
        preserved = m3_rc_get(&output.chunks[0], 0U);
    }
    ready = ready && m3_flow_test_run_init(&fixture, 4U, &clobber);
    for (index = 0U; ready && index < clobber.workspace.count; ++index) {
        ready = m3_flow_zero_storage(
                    clobber.workspace.storages[index], &error) ==
                M3_STATUS_OK;
    }
    m3_flow_test_run_dispose(&clobber);
    M3_TEST_EXPECT(
        test,
        ready && m3_rc_get(&output.chunks[0], 0U) == preserved,
        "owned flow chunks survive transient workspace destruction");
    m3_flow_output_dispose(&output);
    m3_flow_test_fixture_dispose(&fixture);
}

void m3_test_flow_runtime_atomic_retry(m3_test_context *test)
{
    m3_flow_test_fixture fixture;
    m3_runtime_progress_log cancel = {0};
    m3_tensor_view one_frame;
    m3_flow_output output;
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    m3_storage *preserved_storage = NULL;
    m3_rng rng;
    m3_rng preserved_rng;
    m3_rng reference;
    float normals[22];
    float preserved_value = 0.0F;
    m3_error error;
    bool ready = m3_flow_test_fixture_init(&fixture, 7U);

    M3_TEST_EXPECT(test, ready, "create atomic flow runtime fixture");
    if (!ready) {
        return;
    }
    m3_flow_output_init(&output);
    m3_tensor_view_init(&one_frame);
    ready = m3_rng_seed(
                &rng, UINT64_C(9901), UINT64_C(17), &error) ==
                M3_STATUS_OK &&
            m3_tensor_slice(
                &fixture.frames, 1U, 0U, 1U, &one_frame, &error) ==
                M3_STATUS_OK &&
            m3_flow_synthesize_core(
                fixture.fixture.backend, &fixture.config,
                &fixture.weights, &fixture.condition_weights, &one_frame,
                &rng, NULL, NULL, &output, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "establish a published flow result");
    if (!ready) {
        m3_flow_output_dispose(&output);
        m3_flow_test_fixture_dispose(&fixture);
        return;
    }
    preserved_storage = output.storages[0];
    preserved_value = m3_rc_get(&output.chunks[0], 0U);
    preserved_rng = rng;
    (void)m3_backend_get_allocation_stats(
        fixture.fixture.backend, &before, &error);
    cancel.cancel_at = 2U;
    M3_TEST_EXPECT(
        test,
        m3_flow_synthesize_core(
            fixture.fixture.backend, &fixture.config, &fixture.weights,
            &fixture.condition_weights, &fixture.frames, &rng,
            m3_rc_progress, &cancel, &output, NULL) ==
                M3_STATUS_CANCELLED &&
            cancel.call_count == 3U && output.storages[0] ==
                preserved_storage &&
            m3_rc_get(&output.chunks[0], 0U) == preserved_value &&
            m3_flow_rng_equal(&rng, &preserved_rng) &&
            m3_backend_get_allocation_stats(
                fixture.fixture.backend, &after, &error) == M3_STATUS_OK &&
            after.live_allocated_bytes == before.live_allocated_bytes &&
            after.live_storage_count == before.live_storage_count,
        "cancellation discards all new state and preserves output and RNG");
    reference = preserved_rng;
    ready = m3_flow_reference_normals(&reference, normals) &&
            m3_flow_synthesize_core(
                fixture.fixture.backend, &fixture.config,
                &fixture.weights, &fixture.condition_weights,
                &fixture.frames, &rng, NULL, NULL, &output, &error) ==
                M3_STATUS_OK;
    M3_TEST_EXPECT(
        test,
        ready && m3_flow_runtime_values(&output, normals) &&
            m3_flow_rng_equal(&rng, &reference),
        "retry after cancellation matches a clean result and RNG stream");
    m3_flow_output_dispose(&output);
    m3_flow_test_fixture_dispose(&fixture);
}

void m3_test_flow_runtime_validation(m3_test_context *test)
{
    m3_flow_test_fixture fixture;
    m3_op_test_fixture other;
    m3_flow_output output;
    m3_flow_output inconsistent = {0};
    m3_flow_weights bad_weights;
    m3_flow_config bad_config;
    m3_tensor_view bad_frames;
    m3_rng rng;
    m3_rng bad_rng = {0};
    m3_storage *preserved = NULL;
    m3_error error;
    bool ready = m3_flow_test_fixture_init(&fixture, 7U);

    M3_TEST_EXPECT(test, ready, "create flow validation fixture");
    if (!ready) {
        return;
    }
    m3_flow_output_init(&output);
    ready = m3_rng_seed(
                &rng, UINT64_C(77), UINT64_C(19), &error) ==
                M3_STATUS_OK &&
            m3_flow_synthesize_core(
                fixture.fixture.backend, &fixture.config,
                &fixture.weights, &fixture.condition_weights,
                &fixture.frames, &rng, NULL, NULL, &output, &error) ==
                M3_STATUS_OK &&
            m3_op_test_fixture_init(&other);
    M3_TEST_EXPECT(test, ready, "establish flow output and second backend");
    if (!ready) {
        m3_flow_output_dispose(&output);
        m3_flow_test_fixture_dispose(&fixture);
        return;
    }
    preserved = output.storages[0];
    bad_weights = fixture.weights;
    bad_weights.layers[0].query = fixture.weights.time_projection;
    bad_config = fixture.config;
    bad_config.maximum_frames = 6U;
    bad_frames = fixture.frames;
    bad_frames.metadata.shape[1] = 0U;
    inconsistent.chunk_count = 1U;
    M3_TEST_EXPECT(
        test,
        m3_flow_synthesize_core(
            fixture.fixture.backend, &fixture.config, &bad_weights,
            &fixture.condition_weights, &fixture.frames, &rng, NULL,
            NULL, &output, NULL) == M3_STATUS_INVALID_ARGUMENT &&
            output.storages[0] == preserved &&
            m3_flow_synthesize_core(
                fixture.fixture.backend, &bad_config, &fixture.weights,
                &fixture.condition_weights, &fixture.frames, &rng, NULL,
                NULL, &output, NULL) == M3_STATUS_OUT_OF_RANGE &&
            output.storages[0] == preserved &&
            m3_flow_synthesize_core(
                fixture.fixture.backend, &fixture.config,
                &fixture.weights, &fixture.condition_weights, &bad_frames,
                &rng, NULL, NULL, &output, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            output.storages[0] == preserved,
        "invalid weights, capacity, and F=0 preserve published output");
    M3_TEST_EXPECT(
        test,
        m3_flow_synthesize_core(
            other.backend, &fixture.config, &fixture.weights,
            &fixture.condition_weights, &fixture.frames, &rng, NULL,
            NULL, &output, NULL) == M3_STATUS_INVALID_ARGUMENT &&
            m3_flow_synthesize_core(
                fixture.fixture.backend, &fixture.config,
                &fixture.weights, &fixture.condition_weights,
                &fixture.frames, &bad_rng, NULL, NULL, &output, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            m3_flow_synthesize_core(
                fixture.fixture.backend, &fixture.config,
                &fixture.weights, &fixture.condition_weights,
                &fixture.frames, &rng, NULL, NULL, &inconsistent, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            m3_flow_synthesize_core(
                NULL, &fixture.config, &fixture.weights,
                &fixture.condition_weights, &fixture.frames, &rng, NULL,
                NULL, &output, NULL) == M3_STATUS_INVALID_ARGUMENT &&
            output.storages[0] == preserved,
        "backend, RNG, output, and null-error validation is atomic");
    m3_op_test_fixture_dispose(&other);
    m3_flow_output_dispose(&output);
    m3_flow_test_fixture_dispose(&fixture);
}

static bool m3_flow_plan_add_spec(const m3_runtime_tensor_spec *spec,
                                  uint64_t *total)
{
    m3_tensor_metadata metadata;
    m3_error error;

    if (m3_tensor_metadata_init(
            &metadata, spec->dtype, spec->rank, spec->shape, &error) !=
        M3_STATUS_OK) {
        return false;
    }
    *total += (uint64_t)metadata.byte_count;
    return true;
}

void m3_test_flow_preflight_live_set(m3_test_context *test)
{
    m3_flow_test_fixture fixture;
    m3_runtime_tensor_spec flow_specs[M3_FLOW_WORKSPACE_COUNT];
    m3_runtime_tensor_spec condition_specs[M3_CONDITION_WORKSPACE_COUNT];
    m3_weight_stage_fake_context *context = NULL;
    m3_backend *backend = NULL;
    m3_storage *live = NULL;
    uint64_t planned = 88U + 32U;
    size_t index;
    m3_error error;
    bool ready = m3_flow_test_fixture_init(&fixture, 7U);

    M3_TEST_EXPECT(test, ready, "create flow preflight fixture");
    if (!ready) {
        return;
    }
    m3_flow_workspace_specs(&fixture.config, 4U, flow_specs);
    m3_condition_workspace_specs(
        &fixture.config.condition, 4U, 4U, condition_specs);
    for (index = 0U; ready && index < M3_FLOW_WORKSPACE_COUNT;
         ++index) {
        ready = m3_flow_plan_add_spec(&flow_specs[index], &planned);
    }
    for (index = 0U; ready && index < M3_CONDITION_WORKSPACE_COUNT;
         ++index) {
        ready = m3_flow_plan_add_spec(&condition_specs[index], &planned);
    }
    ready = ready && m3_weight_stage_test_fake_backend_create(
                         UINT64_MAX, planned + 31U, 0U, &backend,
                         &context, &error) &&
            m3_storage_allocate(
                backend, 32U, 16U, &live, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "build a live-allocation bounded backend");
    if (!ready) {
        m3_backend_free(backend);
        m3_flow_test_fixture_dispose(&fixture);
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_flow_preflight(
            backend, &fixture.config, &fixture.frames, 3U, 4U,
            flow_specs, &error) == M3_STATUS_OUT_OF_MEMORY,
        "preflight adds new outputs and peak workspaces to live bytes");
    m3_storage_free(live);
    M3_TEST_EXPECT(
        test,
        m3_flow_preflight(
            backend, &fixture.config, &fixture.frames, 3U, 4U,
            flow_specs, &error) == M3_STATUS_OK &&
            context->allocation_calls == 1U && context->free_calls == 1U,
        "preflight does not double-count already-live staged allocations");
    m3_backend_free(backend);
    m3_flow_test_fixture_dispose(&fixture);
}

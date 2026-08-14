/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "semantic_runtime_test.h"
#include "weight_stage_fixture.h"

#include "m3_backend.h"

#include <stdint.h>

static m3_status m3_semantic_plan_preflight_case(
    const m3_semantic_plan *plan, uint64_t maximum_storage,
    uint64_t recommended_working_set)
{
    m3_weight_stage_fake_context *context = NULL;
    m3_backend *backend = NULL;
    m3_error error;
    m3_status status;

    m3_error_reset(&error);
    if (!m3_weight_stage_test_fake_backend_create(
            maximum_storage, recommended_working_set, SIZE_MAX,
            &backend, &context, &error)) {
        return M3_STATUS_INTERNAL;
    }
    (void)context;
    status = m3_semantic_preflight(backend, plan, &error);
    m3_backend_free(backend);
    return status;
}

static bool m3_semantic_plan_live_case(
    const m3_semantic_plan *plan, uint64_t recommended_working_set,
    m3_status expected_status)
{
    m3_weight_stage_fake_context *context = NULL;
    m3_backend_allocation_stats stats;
    m3_storage *storage = NULL;
    m3_backend *backend = NULL;
    m3_error error;
    m3_status status;
    bool valid;

    m3_error_reset(&error);
    if (!m3_weight_stage_test_fake_backend_create(
            UINT64_MAX, recommended_working_set, SIZE_MAX, &backend,
            &context, &error) ||
        m3_storage_allocate(
            backend, 64U, 64U, &storage, &error) != M3_STATUS_OK ||
        m3_backend_get_allocation_stats(backend, &stats, &error) !=
            M3_STATUS_OK) {
        m3_storage_free(storage);
        m3_backend_free(backend);
        return false;
    }
    status = m3_semantic_preflight(backend, plan, &error);
    valid = stats.live_allocated_bytes == 64U &&
            stats.live_storage_count == 1U && status == expected_status;
    m3_storage_free(storage);
    valid = valid && context->free_calls == 1U &&
            m3_backend_get_allocation_stats(backend, &stats, &error) ==
                M3_STATUS_OK &&
            stats.live_allocated_bytes == 0U &&
            stats.live_storage_count == 0U;
    m3_backend_free(backend);
    return valid;
}

void m3_test_semantic_plan_contract(m3_test_context *test)
{
    m3_semantic_plan plan;
    m3_error error;

    m3_error_reset(&error);
    M3_TEST_EXPECT(
        test,
        m3_semantic_plan_build(5000U, 9000U, &plan, &error) ==
            M3_STATUS_OK,
        "build maximum official semantic generation plan");
    M3_TEST_EXPECT(
        test,
        plan.prompt_tokens == 5000U && plan.frame_limit == 9000U &&
            plan.cache_capacity == 14000U &&
            plan.progress_total == UINT64_C(441048) &&
            plan.prefill_added_bytes == UINT64_C(5340885004) &&
            plan.frame_added_bytes == UINT64_C(4724113460) &&
            plan.maximum_added_bytes == UINT64_C(5340885004) &&
            plan.largest_storage_bytes == UINT64_C(589824000) &&
            plan.output_bytes == (size_t)UINT64_C(589824000),
        "maximum cache, phase peaks, progress, and output arithmetic are exact");
    M3_TEST_EXPECT(
        test,
        UINT64_C(18461001728) + plan.maximum_added_bytes ==
                UINT64_C(23801886732) &&
            UINT64_C(26800603136) - UINT64_C(23801886732) ==
                UINT64_C(2998716404),
        "official weights plus generation peak preserve audited M4 headroom");
    M3_TEST_EXPECT(
        test,
        m3_semantic_plan_preflight_case(
            &plan, plan.largest_storage_bytes,
            plan.maximum_added_bytes) == M3_STATUS_OK,
        "accept exact storage and recommended-working-set boundaries");
    M3_TEST_EXPECT(
        test,
        m3_semantic_plan_preflight_case(
            &plan, plan.largest_storage_bytes - 1U, 0U) ==
            M3_STATUS_OUT_OF_RANGE,
        "reject one byte below the largest semantic storage");
    M3_TEST_EXPECT(
        test,
        m3_semantic_plan_preflight_case(
            &plan, UINT64_MAX, plan.maximum_added_bytes - 1U) ==
            M3_STATUS_OUT_OF_MEMORY,
        "reject one byte below the semantic working-set peak");
    M3_TEST_EXPECT(
        test,
        m3_semantic_plan_live_case(
            &plan, plan.maximum_added_bytes + 64U, M3_STATUS_OK) &&
            m3_semantic_plan_live_case(
                &plan, plan.maximum_added_bytes + 63U,
                M3_STATUS_OUT_OF_MEMORY),
        "include unrelated live storage at the exact preflight boundary");
    M3_TEST_EXPECT(
        test,
        m3_semantic_plan_build(2U, 1U, &plan, NULL) ==
                M3_STATUS_OUT_OF_RANGE &&
            m3_semantic_plan_build(5001U, 1U, &plan, NULL) ==
                M3_STATUS_OUT_OF_RANGE &&
            m3_semantic_plan_build(3U, 0U, &plan, NULL) ==
                M3_STATUS_OUT_OF_RANGE &&
            m3_semantic_plan_build(3U, 9001U, &plan, NULL) ==
                M3_STATUS_OUT_OF_RANGE &&
            m3_semantic_plan_build(3U, 1U, NULL, NULL) ==
                M3_STATUS_OUT_OF_RANGE,
        "reject every public semantic plan bound outside the contract");
}

/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_internal.h"

#include "m3_flow_internal.h"

#include <stdint.h>

m3_status m3_music3_checked_add(uint64_t left, uint64_t right,
                                uint64_t *sum, m3_error *error)
{
    if (sum == NULL || right > UINT64_MAX - left) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Music3 allocation plan overflows");
    }
    *sum = left + right;
    return M3_STATUS_OK;
}

static m3_status m3_music3_plan_add(
    m3_music3_allocation_plan *plan, uint64_t bytes,
    m3_error *error)
{
    m3_status status = m3_music3_checked_add(
        plan->added_bytes, bytes, &plan->added_bytes, error);

    if (status == M3_STATUS_OK && bytes > plan->largest_storage_bytes) {
        plan->largest_storage_bytes = bytes;
    }
    return status;
}

static m3_status m3_music3_plan_merge(
    m3_music3_allocation_plan *target,
    const m3_music3_allocation_plan *source, m3_error *error)
{
    m3_status status = m3_music3_checked_add(
        target->added_bytes, source->added_bytes, &target->added_bytes,
        error);

    if (status == M3_STATUS_OK &&
        source->largest_storage_bytes > target->largest_storage_bytes) {
        target->largest_storage_bytes = source->largest_storage_bytes;
    }
    return status;
}

m3_status m3_music3_table_plan(
    const m3_weight_table *table, m3_music3_allocation_plan *plan,
    m3_error *error)
{
    m3_music3_allocation_plan built = {0U, 0U};
    size_t index;
    m3_status status = M3_STATUS_OK;

    if (table == NULL || plan == NULL || table->shard_count == 0U ||
        table->shards == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "Music3 weight table has no shards");
    }
    for (index = 0U; index < table->shard_count &&
                    status == M3_STATUS_OK; ++index) {
        uint64_t bytes = table->shards[index].payload_bytes;

        if (bytes == 0U) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "Music3 weight shard is empty");
        } else {
            status = m3_music3_plan_add(&built, bytes, error);
        }
    }
    if (status == M3_STATUS_OK &&
        built.added_bytes != table->aggregate_payload_bytes) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "Music3 weight table total is inconsistent");
    }
    if (status == M3_STATUS_OK) {
        *plan = built;
        m3_error_reset(error);
    }
    return status;
}

m3_status m3_music3_component_plan(
    const m3_music3_engine *engine, m3_component_id component,
    m3_music3_allocation_plan *plan, m3_error *error)
{
    size_t index = (size_t)component;

    if (engine == NULL || plan == NULL ||
        component < M3_COMPONENT_LANGUAGE_MODEL ||
        component > M3_COMPONENT_VOCODER ||
        engine->component_payload_bytes[index] == 0U ||
        engine->component_largest_shard_bytes[index] == 0U ||
        engine->component_largest_shard_bytes[index] >
            engine->component_payload_bytes[index]) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "Music3 component allocation plan is invalid");
    }
    plan->added_bytes = engine->component_payload_bytes[index];
    plan->largest_storage_bytes =
        engine->component_largest_shard_bytes[index];
    return M3_STATUS_OK;
}

m3_status m3_music3_preflight_added(
    m3_backend *backend, const m3_music3_allocation_plan *plan,
    m3_error *error)
{
    m3_backend_allocation_stats stats;
    m3_backend_info info;
    uint64_t live;
    m3_status status;

    if (backend == NULL || plan == NULL || plan->added_bytes == 0U ||
        plan->largest_storage_bytes == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 allocation preflight is invalid");
    }
    status = m3_backend_get_info(backend, &info, error);
    if (status == M3_STATUS_OK) {
        status = m3_backend_get_allocation_stats(backend, &stats, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    live = (uint64_t)stats.live_allocated_bytes;
    if ((size_t)live != stats.live_allocated_bytes ||
        plan->added_bytes > UINT64_MAX - live) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Music3 working-set plan overflows");
    }
    if (plan->largest_storage_bytes > info.maximum_storage_bytes) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "Music3 tensor exceeds backend storage limit");
    }
    if (info.recommended_working_set_bytes != 0U &&
        live + plan->added_bytes > info.recommended_working_set_bytes) {
        return m3_error_set(
            error, M3_STATUS_OUT_OF_MEMORY,
            "Music3 phase exceeds backend recommended working set");
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_music3_preflight_semantic(
    const m3_music3_engine *engine, const m3_semantic_plan *semantic,
    m3_error *error)
{
    m3_music3_allocation_plan aggregate = {0U, 0U};
    size_t index;
    m3_status status;

    if (engine == NULL || semantic == NULL ||
        semantic->maximum_added_bytes == 0U ||
        semantic->largest_storage_bytes == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 semantic preflight is invalid");
    }
    aggregate.added_bytes = semantic->maximum_added_bytes;
    aggregate.largest_storage_bytes = semantic->largest_storage_bytes;
    for (index = M3_COMPONENT_LANGUAGE_MODEL;
         index <= M3_COMPONENT_RVQ_DEPTH_DECODER; ++index) {
        m3_music3_allocation_plan table = {0U, 0U};

        status = m3_music3_component_plan(
            engine, (m3_component_id)index, &table, error);
        if (status == M3_STATUS_OK) {
            status = m3_music3_plan_merge(&aggregate, &table, error);
        }
        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    return m3_music3_preflight_added(
        engine->backend, &aggregate, error);
}

static m3_status m3_music3_flow_shape(
    const m3_music3_engine *engine,
    const m3_tensor_view *frame_hiddens, m3_flow_config *config,
    size_t *chunk_count, uint64_t *maximum_length, m3_error *error)
{
    m3_tensor_view checked;
    size_t index;
    m3_status status;

    if (engine == NULL || frame_hiddens == NULL ||
        frame_hiddens->storage == NULL ||
        frame_hiddens->metadata.dtype != M3_DTYPE_BF16 ||
        frame_hiddens->metadata.rank != 4U ||
        frame_hiddens->metadata.shape[0] != 1U ||
        frame_hiddens->metadata.shape[1] == 0U ||
        frame_hiddens->metadata.shape[1] > M3_FLOW_MAX_FRAMES ||
        frame_hiddens->metadata.shape[2] != M3_RVQ_CODEBOOK_COUNT ||
        frame_hiddens->metadata.shape[3] != M3_QWEN_HIDDEN_SIZE ||
        m3_storage_backend(frame_hiddens->storage) != engine->backend) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 semantic frames are invalid");
    }
    m3_tensor_view_init(&checked);
    status = m3_tensor_reshape(
        frame_hiddens, frame_hiddens->metadata.rank,
        frame_hiddens->metadata.shape, &checked, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_flow_config_init(config);
    status = m3_flow_chunk_count(
        config, frame_hiddens->metadata.shape[1], chunk_count, error);
    *maximum_length = 0U;
    for (index = 0U; status == M3_STATUS_OK &&
                    index < *chunk_count; ++index) {
        uint64_t start = 0U;
        uint64_t frames = 0U;
        uint64_t length = 0U;

        status = m3_flow_chunk_window(
            config, frame_hiddens->metadata.shape[1], index, &start,
            &frames, error);
        if (status == M3_STATUS_OK) {
            status = m3_condition_output_length(
                frames, config->condition.resize_numerator,
                config->condition.resize_denominator, &length, error);
        }
        if (status == M3_STATUS_OK && length > *maximum_length) {
            *maximum_length = length;
        }
        (void)start;
    }
    return status;
}

m3_status m3_music3_preflight_flow(
    const m3_music3_engine *engine,
    const m3_tensor_view *frame_hiddens, m3_error *error)
{
    m3_runtime_tensor_spec specs[M3_FLOW_WORKSPACE_COUNT];
    m3_flow_allocation_plan flow_plan;
    m3_music3_allocation_plan aggregate = {0U, 0U};
    m3_flow_config config;
    size_t chunk_count = 0U;
    uint64_t maximum_length = 0U;
    size_t index;
    m3_status status = m3_music3_flow_shape(
        engine, frame_hiddens, &config, &chunk_count, &maximum_length,
        error);

    if (status == M3_STATUS_OK) {
        m3_flow_workspace_specs(&config, maximum_length, specs);
        status = m3_flow_allocation_plan_build(
            &config, frame_hiddens, chunk_count, maximum_length, specs,
            &flow_plan, error);
    }
    if (status == M3_STATUS_OK) {
        aggregate.added_bytes = flow_plan.added_bytes;
        aggregate.largest_storage_bytes =
            flow_plan.largest_storage_bytes;
    }
    for (index = M3_COMPONENT_CONDITION_ENCODER;
         index <= M3_COMPONENT_TRANSFORMER && status == M3_STATUS_OK;
         ++index) {
        m3_music3_allocation_plan table = {0U, 0U};

        status = m3_music3_component_plan(
            engine, (m3_component_id)index, &table, error);
        if (status == M3_STATUS_OK) {
            status = m3_music3_plan_merge(&aggregate, &table, error);
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_preflight_added(
            engine->backend, &aggregate, error);
    }
    return status;
}

/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_vocoder_internal.h"

#include <stdlib.h>
#include <string.h>

static const m3_tensor_view *m3_vocoder_bind_next(
    m3_vocoder_runtime *runtime, size_t *index)
{
    const m3_tensor_view *view = &runtime->weights.views[*index];

    ++*index;
    return view;
}

m3_status m3_vocoder_runtime_bind(
    m3_vocoder_runtime *runtime, const m3_vocoder_plan *plan,
    m3_error *error)
{
    size_t expected;
    size_t index = 0U;
    size_t block;

    if (plan == NULL || plan->block_count > M3_VOCODER_BLOCK_COUNT ||
        plan->residual_count > M3_VOCODER_RESIDUAL_COUNT ||
        plan->block_count != plan->config.block_count ||
        plan->residual_count != plan->config.residual_count) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "vocoder runtime binding contract is invalid");
    }
    expected = 7U + plan->block_count *
        (3U + 6U * plan->residual_count);
    if (runtime == NULL || plan == NULL ||
        runtime->weights.count != plan->entry_count ||
        plan->entry_count != expected) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "vocoder runtime binding contract is invalid");
    }
    (void)memset(&runtime->bound, 0, sizeof(runtime->bound));
    runtime->bound.block_count = plan->block_count;
    runtime->bound.residual_count = plan->residual_count;
    runtime->bound.decoder_input_weight =
        m3_vocoder_bind_next(runtime, &index);
    runtime->bound.decoder_input_bias =
        m3_vocoder_bind_next(runtime, &index);
    runtime->bound.convolution_input_weight =
        m3_vocoder_bind_next(runtime, &index);
    runtime->bound.convolution_input_bias =
        m3_vocoder_bind_next(runtime, &index);
    for (block = 0U; block < plan->block_count; ++block) {
        m3_vocoder_block_weights *target = &runtime->bound.blocks[block];
        size_t residual;

        target->snake_alpha = m3_vocoder_bind_next(runtime, &index);
        target->transpose_weight = m3_vocoder_bind_next(runtime, &index);
        target->transpose_bias = m3_vocoder_bind_next(runtime, &index);
        for (residual = 0U; residual < plan->residual_count; ++residual) {
            m3_vocoder_residual_weights *unit =
                &target->residuals[residual];

            unit->snake1_alpha = m3_vocoder_bind_next(runtime, &index);
            unit->conv1_weight = m3_vocoder_bind_next(runtime, &index);
            unit->conv1_bias = m3_vocoder_bind_next(runtime, &index);
            unit->snake2_alpha = m3_vocoder_bind_next(runtime, &index);
            unit->conv2_weight = m3_vocoder_bind_next(runtime, &index);
            unit->conv2_bias = m3_vocoder_bind_next(runtime, &index);
        }
    }
    runtime->bound.snake_output_alpha =
        m3_vocoder_bind_next(runtime, &index);
    runtime->bound.convolution_output_weight =
        m3_vocoder_bind_next(runtime, &index);
    runtime->bound.convolution_output_bias =
        m3_vocoder_bind_next(runtime, &index);
    if (index != plan->entry_count) {
        (void)memset(&runtime->bound, 0, sizeof(runtime->bound));
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "vocoder runtime binding count is inconsistent");
    }
    return M3_STATUS_OK;
}

void m3_vocoder_runtime_free(m3_vocoder_runtime *runtime)
{
    if (runtime != NULL) {
        m3_runtime_workspace_dispose(&runtime->weights);
        free(runtime);
    }
}

m3_status m3_vocoder_runtime_create(
    m3_vocoder_runtime **runtime, const m3_weight_stage *vocoder,
    m3_progress_callback progress, void *progress_context,
    m3_error *error)
{
    m3_vocoder_materialize_io io;
    m3_vocoder_plan_config config;
    m3_vocoder_plan plan;
    m3_status status;

    m3_vocoder_materialize_io_init(&io);
    m3_vocoder_plan_official_config(&config);
    m3_vocoder_plan_init(&plan);
    status = m3_vocoder_plan_build(&config, &plan, error);
    if (status == M3_STATUS_OK &&
        (plan.entry_count != M3_VOCODER_RUNTIME_WEIGHT_COUNT ||
         plan.source_count != M3_VOCODER_SOURCE_WEIGHT_COUNT)) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "official vocoder plan count is inconsistent");
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_runtime_create_core(
            runtime, vocoder, &plan, &io, progress, progress_context,
            error);
    }
    m3_vocoder_plan_dispose(&plan);
    return status;
}

const m3_vocoder_weights *m3_vocoder_runtime_weights(
    const m3_vocoder_runtime *runtime)
{
    return runtime == NULL ? NULL : &runtime->bound;
}

static bool m3_vocoder_config_equal(
    const m3_vocoder_plan_config *left,
    const m3_vocoder_plan_config *right)
{
    size_t block;

    if (left->latent_channels != right->latent_channels ||
        left->maximum_latent_length != right->maximum_latent_length ||
        left->decoder_input_channels != right->decoder_input_channels ||
        left->decoder_output_channels != right->decoder_output_channels ||
        left->initial_channels != right->initial_channels ||
        left->block_count != right->block_count ||
        left->residual_count != right->residual_count) {
        return false;
    }
    for (block = 0U; block < M3_VOCODER_BLOCK_COUNT; ++block) {
        if (left->strides[block] != right->strides[block]) {
            return false;
        }
    }
    return true;
}

static bool m3_vocoder_metadata_equal(
    const m3_tensor_metadata *left, const m3_tensor_metadata *right)
{
    uint8_t axis;

    if (left->dtype != right->dtype || left->rank != right->rank ||
        left->element_count != right->element_count ||
        left->byte_count != right->byte_count) {
        return false;
    }
    for (axis = 0U; axis < left->rank; ++axis) {
        if (left->shape[axis] != right->shape[axis]) {
            return false;
        }
    }
    return true;
}

static bool m3_vocoder_bound_order(const m3_vocoder_runtime *runtime)
{
    const m3_vocoder_weights *weights = &runtime->bound;
    size_t index = 0U;
    size_t block;

    if (weights->block_count != M3_VOCODER_BLOCK_COUNT ||
        weights->residual_count != M3_VOCODER_RESIDUAL_COUNT) {
        return false;
    }
#define M3_VOCODER_EXPECT_BOUND(pointer) \
    do { \
        if ((pointer) != &runtime->weights.views[index++]) { \
            return false; \
        } \
    } while (0)
    M3_VOCODER_EXPECT_BOUND(weights->decoder_input_weight);
    M3_VOCODER_EXPECT_BOUND(weights->decoder_input_bias);
    M3_VOCODER_EXPECT_BOUND(weights->convolution_input_weight);
    M3_VOCODER_EXPECT_BOUND(weights->convolution_input_bias);
    for (block = 0U; block < M3_VOCODER_BLOCK_COUNT; ++block) {
        const m3_vocoder_block_weights *unit = &weights->blocks[block];
        size_t residual;

        M3_VOCODER_EXPECT_BOUND(unit->snake_alpha);
        M3_VOCODER_EXPECT_BOUND(unit->transpose_weight);
        M3_VOCODER_EXPECT_BOUND(unit->transpose_bias);
        for (residual = 0U; residual < M3_VOCODER_RESIDUAL_COUNT;
             ++residual) {
            const m3_vocoder_residual_weights *res =
                &unit->residuals[residual];

            M3_VOCODER_EXPECT_BOUND(res->snake1_alpha);
            M3_VOCODER_EXPECT_BOUND(res->conv1_weight);
            M3_VOCODER_EXPECT_BOUND(res->conv1_bias);
            M3_VOCODER_EXPECT_BOUND(res->snake2_alpha);
            M3_VOCODER_EXPECT_BOUND(res->conv2_weight);
            M3_VOCODER_EXPECT_BOUND(res->conv2_bias);
        }
    }
    M3_VOCODER_EXPECT_BOUND(weights->snake_output_alpha);
    M3_VOCODER_EXPECT_BOUND(weights->convolution_output_weight);
    M3_VOCODER_EXPECT_BOUND(weights->convolution_output_bias);
#undef M3_VOCODER_EXPECT_BOUND
    return index == runtime->weights.count;
}

static m3_status m3_vocoder_runtime_owners_validate(
    const m3_vocoder_runtime *runtime, m3_error *error)
{
    size_t index;

    for (index = 0U; index < runtime->weights.count; ++index) {
        size_t earlier;

        if (runtime->weights.storages[index] == NULL) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "published vocoder owner is null");
        }
        for (earlier = 0U; earlier < index; ++earlier) {
            if (runtime->weights.storages[index] ==
                runtime->weights.storages[earlier]) {
                return m3_error_set(
                    error, M3_STATUS_INVALID_FORMAT,
                    "published vocoder tensors share owned storage");
            }
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_vocoder_runtime_validate(
    const m3_vocoder_runtime *runtime, m3_backend *backend,
    m3_error *error)
{
    m3_vocoder_plan_config config;
    m3_vocoder_plan plan;
    size_t allocated = 0U;
    size_t index;
    m3_status status;

    if (runtime == NULL || backend == NULL || runtime->backend != backend ||
        runtime->weights.backend != backend ||
        runtime->weights.count != M3_VOCODER_RUNTIME_WEIGHT_COUNT ||
        runtime->weights.storages == NULL ||
        runtime->weights.views == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "published vocoder runtime is invalid");
    }
    m3_vocoder_plan_official_config(&config);
    if (!m3_vocoder_config_equal(&runtime->config, &config)) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "published vocoder config is not official");
    }
    status = m3_vocoder_runtime_owners_validate(runtime, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_vocoder_plan_init(&plan);
    status = m3_vocoder_plan_build(&config, &plan, error);
    for (index = 0U; index < plan.entry_count &&
                    status == M3_STATUS_OK; ++index) {
        const m3_tensor_metadata *expected =
            &plan.entries[index].output_metadata;
        const m3_tensor_view *view = &runtime->weights.views[index];
        m3_tensor_view checked;

        if (view->storage != runtime->weights.storages[index] ||
            view->byte_offset != 0U ||
            m3_storage_backend(view->storage) != backend ||
            m3_storage_size(view->storage) != expected->byte_count ||
            !m3_vocoder_metadata_equal(&view->metadata, expected) ||
            !m3_tensor_is_contiguous(view) ||
            expected->byte_count > SIZE_MAX - allocated) {
            status = m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "published vocoder tensor is invalid");
            break;
        }
        allocated += expected->byte_count;
        m3_tensor_view_init(&checked);
        status = m3_tensor_reshape(
            view, view->metadata.rank, view->metadata.shape, &checked,
            error);
    }
    if (status == M3_STATUS_OK &&
        (plan.entry_count != M3_VOCODER_RUNTIME_WEIGHT_COUNT ||
         allocated != runtime->weights.allocated_bytes ||
         !m3_vocoder_bound_order(runtime))) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "published vocoder binding is invalid");
    }
    m3_vocoder_plan_dispose(&plan);
    if (status == M3_STATUS_OK) {
        m3_error_reset(error);
    }
    return status;
}

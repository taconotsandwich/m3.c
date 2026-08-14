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
        plan->residual_count > M3_VOCODER_RESIDUAL_COUNT) {
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

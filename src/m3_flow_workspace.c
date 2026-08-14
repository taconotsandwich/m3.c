/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_flow_internal.h"

#include <stdint.h>
#include <string.h>

static void m3_flow_spec(m3_runtime_tensor_spec *spec, uint8_t rank,
                         const uint64_t *shape)
{
    uint8_t axis;

    (void)memset(spec, 0, sizeof(*spec));
    spec->dtype = M3_DTYPE_F32;
    spec->rank = rank;
    spec->alignment = 64U;
    for (axis = 0U; axis < rank; ++axis) {
        spec->shape[axis] = shape[axis];
    }
}

void m3_flow_workspace_specs(const m3_flow_config *config,
                             uint64_t maximum_length,
                             m3_runtime_tensor_spec *specs)
{
    const uint64_t inner = (uint64_t)config->attention_heads *
                           config->head_dimension;
    const uint64_t sequence = maximum_length + 1U;
    const uint64_t concatenated =
        (uint64_t)config->latent_channels * 2U +
        config->condition_dimension;
    const uint64_t concat[] = {2U, concatenated, maximum_length};
    const uint64_t preprocessed[] = {2U, concatenated, maximum_length};
    const uint64_t projected[] = {2U, maximum_length, inner};
    const uint64_t fourier[] = {2U, config->fourier_dimension};
    const uint64_t time[] = {2U, inner};
    const uint64_t hidden[] = {2U, sequence, inner};
    const uint64_t heads[] = {
        2U, config->attention_heads, sequence, config->head_dimension
    };
    const uint64_t reordered[] = {
        2U, sequence, config->attention_heads, config->head_dimension
    };
    const uint64_t feed_forward[] = {
        2U, sequence,
        (uint64_t)config->feed_forward_dimension * 2U
    };
    const uint64_t gated[] = {
        2U, sequence, config->feed_forward_dimension
    };
    const uint64_t rotary[] = {
        sequence, config->rotary_dimension / 2U
    };
    const uint64_t output_sequence[] = {
        2U, maximum_length, config->latent_channels
    };
    const uint64_t channels[] = {
        2U, config->latent_channels, maximum_length
    };
    const uint64_t velocity[] = {
        2U, maximum_length, config->latent_channels
    };
    const uint64_t latent[] = {
        1U, maximum_length, config->latent_channels
    };
    const uint64_t noise[] = {
        1U, config->latent_channels, maximum_length
    };
    const uint64_t latent_carry[] = {
        1U, config->carry_length, config->latent_channels
    };
    const uint64_t condition_carry[] = {
        1U, config->carry_length, config->condition_dimension
    };

    m3_flow_spec(&specs[M3_FLOW_WS_CONCATENATED], 3U, concat);
    m3_flow_spec(&specs[M3_FLOW_WS_PREPROCESSED], 3U, preprocessed);
    m3_flow_spec(&specs[M3_FLOW_WS_PROJECTED], 3U, projected);
    m3_flow_spec(&specs[M3_FLOW_WS_FOURIER], 2U, fourier);
    m3_flow_spec(&specs[M3_FLOW_WS_TIME_HIDDEN], 2U, time);
    m3_flow_spec(&specs[M3_FLOW_WS_TIME_ACTIVATED], 2U, time);
    m3_flow_spec(&specs[M3_FLOW_WS_TIME_EMBEDDING], 2U, time);
    m3_flow_spec(&specs[M3_FLOW_WS_TIME_ONES], 2U, time);
    m3_flow_spec(&specs[M3_FLOW_WS_HIDDEN], 3U, hidden);
    m3_flow_spec(&specs[M3_FLOW_WS_HIDDEN_TEMP], 3U, hidden);
    m3_flow_spec(&specs[M3_FLOW_WS_NORMALIZED], 3U, hidden);
    m3_flow_spec(&specs[M3_FLOW_WS_QUERY], 3U, hidden);
    m3_flow_spec(&specs[M3_FLOW_WS_KEY], 3U, hidden);
    m3_flow_spec(&specs[M3_FLOW_WS_VALUE], 3U, hidden);
    m3_flow_spec(&specs[M3_FLOW_WS_QUERY_ROTARY], 4U, heads);
    m3_flow_spec(&specs[M3_FLOW_WS_KEY_ROTARY], 4U, heads);
    m3_flow_spec(&specs[M3_FLOW_WS_ATTENTION], 4U, heads);
    m3_flow_spec(&specs[M3_FLOW_WS_REORDER], 4U, reordered);
    m3_flow_spec(&specs[M3_FLOW_WS_FEED_FORWARD], 3U, feed_forward);
    m3_flow_spec(&specs[M3_FLOW_WS_GATED], 3U, gated);
    m3_flow_spec(&specs[M3_FLOW_WS_COSINES], 2U, rotary);
    m3_flow_spec(&specs[M3_FLOW_WS_SINES], 2U, rotary);
    m3_flow_spec(&specs[M3_FLOW_WS_OUTPUT_SEQUENCE], 3U,
                 output_sequence);
    m3_flow_spec(&specs[M3_FLOW_WS_POSTPROCESSED], 3U, channels);
    m3_flow_spec(&specs[M3_FLOW_WS_VELOCITY], 3U, velocity);
    m3_flow_spec(&specs[M3_FLOW_WS_LATENT], 3U, latent);
    m3_flow_spec(&specs[M3_FLOW_WS_NOISE_CHANNELS], 3U, noise);
    m3_flow_spec(&specs[M3_FLOW_WS_NOISE_PROMPT], 3U, latent_carry);
    m3_flow_spec(&specs[M3_FLOW_WS_PREVIOUS_LATENT], 3U, latent_carry);
    m3_flow_spec(&specs[M3_FLOW_WS_PREVIOUS_CONDITION], 3U,
                 condition_carry);
    m3_flow_spec(&specs[M3_FLOW_WS_SCALAR], 3U, latent);
    m3_flow_spec(&specs[M3_FLOW_WS_ARITHMETIC], 3U, latent);
}

static m3_status m3_flow_plan_add(
    const m3_backend_info *info, uint64_t bytes, uint64_t *planned,
    m3_error *error)
{
    if (bytes > info->maximum_storage_bytes) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "flow runtime tensor exceeds storage limit");
    }
    if (bytes > UINT64_MAX - *planned) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "flow runtime allocation plan overflows");
    }
    *planned += bytes;
    return M3_STATUS_OK;
}

static m3_status m3_flow_plan_spec(
    const m3_backend_info *info, const m3_runtime_tensor_spec *spec,
    uint64_t *planned, m3_error *error)
{
    m3_tensor_metadata metadata;
    m3_status status = m3_tensor_metadata_init(
        &metadata, spec->dtype, spec->rank, spec->shape, error);

    if (status == M3_STATUS_OK) {
        status = m3_flow_plan_add(
            info, (uint64_t)metadata.byte_count, planned, error);
    }
    return status;
}

static m3_status m3_flow_plan_outputs(
    const m3_backend_info *info, const m3_flow_config *config,
    uint64_t frame_count, size_t chunk_count, uint64_t *planned,
    m3_error *error)
{
    size_t index;
    m3_status status = M3_STATUS_OK;

    for (index = 0U; index < chunk_count && status == M3_STATUS_OK;
         ++index) {
        uint64_t start = 0U;
        uint64_t frames = 0U;
        uint64_t length = 0U;
        uint64_t shape[3];
        m3_tensor_metadata metadata;

        status = m3_flow_chunk_window(
            config, frame_count, index, &start, &frames, error);
        if (status == M3_STATUS_OK) {
            status = m3_condition_output_length(
                frames, config->condition.resize_numerator,
                config->condition.resize_denominator, &length, error);
        }
        shape[0] = 1U;
        shape[1] = config->latent_channels;
        shape[2] = length;
        if (status == M3_STATUS_OK) {
            status = m3_tensor_metadata_init(
                &metadata, M3_DTYPE_F32, 3U, shape, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_flow_plan_add(
                info, (uint64_t)metadata.byte_count, planned, error);
        }
        (void)start;
    }
    return status;
}

m3_status m3_flow_preflight(
    m3_backend *backend, const m3_flow_config *config,
    const m3_tensor_view *frame_hiddens, size_t chunk_count,
    uint64_t maximum_length, const m3_runtime_tensor_spec *flow_specs,
    m3_error *error)
{
    m3_runtime_tensor_spec condition_specs[M3_CONDITION_WORKSPACE_COUNT];
    m3_backend_allocation_stats stats;
    m3_backend_info info;
    uint64_t maximum_frames;
    uint64_t condition_shape[3];
    m3_tensor_metadata condition_output;
    uint64_t planned;
    size_t index;
    m3_status status;

    if (backend == NULL || config == NULL || frame_hiddens == NULL ||
        flow_specs == NULL || chunk_count == 0U || maximum_length == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow allocation plan is invalid");
    }
    status = m3_backend_get_info(backend, &info, error);
    if (status == M3_STATUS_OK) {
        status = m3_backend_get_allocation_stats(backend, &stats, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    planned = (uint64_t)stats.live_allocated_bytes;
    if ((size_t)planned != stats.live_allocated_bytes) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "flow live allocation count overflows");
    }
    status = m3_flow_plan_outputs(
        &info, config, frame_hiddens->metadata.shape[1], chunk_count,
        &planned, error);
    for (index = 0U; index < M3_FLOW_WORKSPACE_COUNT &&
                    status == M3_STATUS_OK; ++index) {
        status = m3_flow_plan_spec(
            &info, &flow_specs[index], &planned, error);
    }
    maximum_frames = frame_hiddens->metadata.shape[1];
    if (maximum_frames > config->chunk_frames) {
        maximum_frames = config->chunk_frames;
    }
    condition_shape[0] = 1U;
    condition_shape[1] = maximum_length;
    condition_shape[2] = config->condition.output_size;
    if (status == M3_STATUS_OK) {
        status = m3_tensor_metadata_init(
            &condition_output, M3_DTYPE_F32, 3U, condition_shape, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_flow_plan_add(
            &info, (uint64_t)condition_output.byte_count, &planned,
            error);
    }
    m3_condition_workspace_specs(
        &config->condition, maximum_frames, maximum_length,
        condition_specs);
    for (index = 0U; index < M3_CONDITION_WORKSPACE_COUNT &&
                    status == M3_STATUS_OK; ++index) {
        status = m3_flow_plan_spec(
            &info, &condition_specs[index], &planned, error);
    }
    if (status == M3_STATUS_OK &&
        info.recommended_working_set_bytes != 0U &&
        planned > info.recommended_working_set_bytes) {
        status = m3_error_set(
            error, M3_STATUS_OUT_OF_MEMORY,
            "flow runtime exceeds recommended working set");
    }
    if (status == M3_STATUS_OK) {
        m3_error_reset(error);
    }
    return status;
}

m3_status m3_flow_view(m3_flow_run *run, size_t slot, m3_dtype dtype,
                       uint8_t rank, const uint64_t *shape,
                       m3_tensor_view *view, m3_error *error)
{
    if (run == NULL || slot >= run->workspace.count ||
        run->workspace.storages == NULL || view == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow workspace view is invalid");
    }
    m3_tensor_view_init(view);
    return m3_tensor_view_contiguous(
        view, run->workspace.storages[slot], dtype, rank, shape, 0U,
        error);
}

m3_status m3_flow_fill_f32(m3_tensor_view *view, float value,
                           m3_error *error)
{
    float *data = NULL;
    size_t index;
    m3_status status;

    if (view == NULL || view->metadata.dtype != M3_DTYPE_F32 ||
        !m3_tensor_is_contiguous(view)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow fill view is invalid");
    }
    status = m3_tensor_data(view, (void **)&data, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (data == NULL) {
        return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                            "flow fill storage is not host visible");
    }
    for (index = 0U; index < view->metadata.element_count; ++index) {
        data[index] = value;
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_flow_zero_storage(m3_storage *storage, m3_error *error)
{
    static const unsigned char zeros[4096] = {0};
    size_t offset = 0U;
    size_t total;

    if (storage == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "flow zero storage is null");
    }
    total = m3_storage_size(storage);
    while (offset < total) {
        size_t remaining = total - offset;
        size_t amount = remaining < sizeof(zeros) ? remaining
                                                   : sizeof(zeros);
        m3_status status = m3_storage_write(
            storage, offset, zeros, amount, error);

        if (status != M3_STATUS_OK) {
            return status;
        }
        offset += amount;
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

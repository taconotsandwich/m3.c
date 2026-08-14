/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_rvq_condition_internal.h"

#include "m3_music3_schema.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static bool m3_runtime_alignment(size_t alignment)
{
    return alignment >= sizeof(void *) &&
           (alignment & (alignment - 1U)) == 0U;
}

m3_status m3_rvq_condition_preflight(
    m3_backend *backend, size_t output_bytes,
    const m3_runtime_tensor_spec *specs, size_t spec_count,
    m3_error *error)
{
    m3_backend_allocation_stats stats;
    m3_backend_info info;
    uint64_t planned;
    size_t index;
    m3_status status;

    if (backend == NULL || output_bytes == 0U ||
        (spec_count != 0U && specs == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "runtime allocation plan is invalid");
    }
    status = m3_backend_get_info(backend, &info, error);
    if (status == M3_STATUS_OK) {
        status = m3_backend_get_allocation_stats(backend, &stats, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if ((uint64_t)output_bytes > info.maximum_storage_bytes) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "runtime output exceeds backend storage limit");
    }
    planned = (uint64_t)stats.live_allocated_bytes;
    if ((size_t)planned != stats.live_allocated_bytes ||
        (uint64_t)output_bytes > UINT64_MAX - planned) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "runtime allocation plan overflows");
    }
    planned += (uint64_t)output_bytes;
    for (index = 0U; index < spec_count; ++index) {
        m3_tensor_metadata metadata;

        if (!m3_runtime_alignment(specs[index].alignment)) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "runtime tensor alignment is invalid");
        }
        status = m3_tensor_metadata_init(
            &metadata, specs[index].dtype, specs[index].rank,
            specs[index].shape, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        if ((uint64_t)metadata.byte_count > info.maximum_storage_bytes) {
            return m3_error_set(
                error, M3_STATUS_OUT_OF_RANGE,
                "runtime tensor exceeds backend storage limit");
        }
        if ((uint64_t)metadata.byte_count > UINT64_MAX - planned) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "runtime allocation plan overflows");
        }
        planned += (uint64_t)metadata.byte_count;
    }
    if (info.recommended_working_set_bytes != 0U &&
        planned > info.recommended_working_set_bytes) {
        return m3_error_set(
            error, M3_STATUS_OUT_OF_MEMORY,
            "runtime plan exceeds backend recommended working set");
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_rvq_condition_cancel(m3_progress_callback progress,
                                  void *context, uint64_t completed,
                                  uint64_t total, const char *operation,
                                  m3_error *error)
{
    if (progress != NULL && !progress(context, completed, total)) {
        return m3_error_set(error, M3_STATUS_CANCELLED,
                            "%s was cancelled", operation);
    }
    return M3_STATUS_OK;
}

m3_status m3_rvq_condition_check_view(
    m3_backend *backend, const m3_tensor_view *view, m3_dtype dtype,
    uint8_t rank, const uint64_t *shape, const char *name,
    m3_error *error)
{
    const void *data = NULL;
    uint8_t axis;
    m3_status status;

    if (backend == NULL || view == NULL || view->storage == NULL ||
        shape == NULL || name == NULL ||
        m3_storage_backend(view->storage) != backend) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s is not on the runtime backend", name);
    }
    status = m3_tensor_const_data(view, &data, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (view->metadata.dtype != dtype || view->metadata.rank != rank) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s has the wrong dtype or rank", name);
    }
    for (axis = 0U; axis < rank; ++axis) {
        if (view->metadata.shape[axis] != shape[axis]) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "%s has the wrong shape", name);
        }
    }
    (void)data;
    return M3_STATUS_OK;
}

size_t m3_rvq_sequence_length(size_t head_index)
{
    return head_index < M3_RVQ_RESIDUAL_COUNT ? head_index + 2U : 0U;
}

bool m3_rvq_next_embedding_id(size_t head_index, uint32_t code,
                              int32_t *embedding_id)
{
    uint32_t offset;

    if (head_index >= M3_RVQ_RESIDUAL_COUNT - 1U ||
        code >= M3_RVQ_CODEBOOK_SIZE || embedding_id == NULL) {
        return false;
    }
    offset = (uint32_t)head_index * M3_RVQ_CODEBOOK_SIZE;
    *embedding_id = (int32_t)(offset + code);
    return true;
}

void m3_rvq_frame_init(m3_rvq_frame *frame)
{
    if (frame != NULL) {
        (void)memset(frame, 0, sizeof(*frame));
    }
}

void m3_rvq_frame_dispose(m3_rvq_frame *frame)
{
    if (frame == NULL) {
        return;
    }
    m3_storage_free(frame->storage);
    m3_rvq_frame_init(frame);
}

void m3_rvq_feedback_init(m3_rvq_feedback *feedback)
{
    if (feedback != NULL) {
        (void)memset(feedback, 0, sizeof(*feedback));
    }
}

void m3_rvq_feedback_dispose(m3_rvq_feedback *feedback)
{
    if (feedback == NULL) {
        return;
    }
    m3_storage_free(feedback->storage);
    m3_rvq_feedback_init(feedback);
}

void m3_condition_output_init(m3_condition_output *output)
{
    if (output != NULL) {
        (void)memset(output, 0, sizeof(*output));
    }
}

void m3_condition_output_dispose(m3_condition_output *output)
{
    if (output == NULL) {
        return;
    }
    m3_storage_free(output->storage);
    m3_condition_output_init(output);
}

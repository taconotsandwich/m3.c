/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_waveform_internal.h"

#include "m3_backend.h"
#include "m3_flow_runtime.h"
#include "m3_vocoder_internal.h"

#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define M3_WAVEFORM_LEFT_CROP_HOPS 86U
#define M3_WAVEFORM_RIGHT_CROP_HOPS 258U
#define M3_WAVEFORM_OVERLAP_HOPS 344U
#define M3_WAVEFORM_SAMPLES_PER_HOP 512U
#define M3_WAVEFORM_MAXIMUM_CHUNKS 89U
#define M3_WAVEFORM_COPY_FLOATS 256U

_Static_assert(sizeof(float) == sizeof(uint32_t),
               "waveform assembly requires 32-bit float");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "waveform assembly requires IEEE binary32 float");

typedef struct {
    size_t sample_count;
    size_t crop_start;
    size_t copy_count;
} m3_waveform_chunk_plan;

typedef struct {
    m3_waveform_chunk_plan chunks[M3_WAVEFORM_MAXIMUM_CHUNKS];
    m3_backend *backend;
    size_t chunk_count;
    size_t output_samples;
    size_t output_bytes;
    uint64_t progress_total;
} m3_waveform_plan;

static bool m3_waveform_size_add(size_t left, size_t right,
                                 size_t *result)
{
    if (left > SIZE_MAX - right) {
        return false;
    }
    *result = left + right;
    return true;
}

static bool m3_waveform_size_multiply(size_t left, size_t right,
                                      size_t *result)
{
    if (left != 0U && right > SIZE_MAX / left) {
        return false;
    }
    *result = left * right;
    return true;
}

static m3_status m3_waveform_existing_output_validate(
    const m3_vocoder_output *output, m3_error *error)
{
    const void *unused = NULL;
    m3_status status;

    if (output == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "waveform output is null");
    }
    if (output->storage == NULL) {
        if (output->waveform.storage != NULL) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "waveform output ownership is inconsistent");
        }
        return M3_STATUS_OK;
    }
    if (output->waveform.storage != output->storage) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "waveform output ownership is inconsistent");
    }
    status = m3_tensor_const_data(&output->waveform, &unused, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (output->waveform.metadata.dtype != M3_DTYPE_F32 ||
        output->waveform.metadata.rank != 3U ||
        output->waveform.metadata.shape[0] != 1U ||
        output->waveform.metadata.shape[1] != 2U ||
        !m3_tensor_is_contiguous(&output->waveform)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "waveform output view is inconsistent");
    }
    return M3_STATUS_OK;
}

static m3_status m3_waveform_chunk_count(
    uint64_t frame_count, size_t supplied, size_t *expected,
    m3_error *error)
{
    uint64_t count;

    if (frame_count == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "waveform frame count is zero");
    }
    if (frame_count > M3_FLOW_MAX_FRAMES) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "waveform frame count exceeds capacity");
    }
    count = frame_count <= M3_FLOW_CHUNK_FRAMES
                ? 1U
                : (frame_count - 1U) / M3_FLOW_CHUNK_HOP;
    if (count == 0U || count > M3_WAVEFORM_MAXIMUM_CHUNKS ||
        count > SIZE_MAX) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "waveform chunk count is inconsistent");
    }
    *expected = (size_t)count;
    if (supplied != *expected) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "waveform chunk count does not match frames");
    }
    return M3_STATUS_OK;
}

static m3_status m3_waveform_plan_chunk(
    uint64_t frame_count, size_t index, size_t chunk_count,
    m3_waveform_chunk_plan *chunk, size_t *latent_total,
    size_t *sample_total, m3_error *error)
{
    uint64_t chunk_frames;
    uint64_t latent_length;
    size_t left;
    size_t right;
    size_t cropped;

    if (index + 1U < chunk_count) {
        chunk_frames = M3_FLOW_CHUNK_FRAMES;
    } else {
        uint64_t start = (uint64_t)index * M3_FLOW_CHUNK_HOP;

        if (start >= frame_count) {
            return m3_error_set(error, M3_STATUS_INTERNAL,
                                "waveform final chunk starts out of range");
        }
        chunk_frames = frame_count - start;
    }
    if ((chunk_count == 1U &&
         (chunk_frames == 0U || chunk_frames > M3_FLOW_CHUNK_FRAMES)) ||
        (chunk_count > 1U && index + 1U == chunk_count &&
         (chunk_frames <= M3_FLOW_CHUNK_HOP ||
          chunk_frames > M3_FLOW_CHUNK_FRAMES))) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "waveform final chunk length is inconsistent");
    }
    latent_length = chunk_frames * 441U / 128U;
    if (latent_length == 0U ||
        latent_length > M3_VOCODER_MAXIMUM_LATENT_LENGTH ||
        latent_length > SIZE_MAX / M3_WAVEFORM_SAMPLES_PER_HOP) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "waveform latent length is invalid");
    }
    chunk->sample_count =
        (size_t)latent_length * M3_WAVEFORM_SAMPLES_PER_HOP;
    left = index == 0U
               ? 0U
               : M3_WAVEFORM_LEFT_CROP_HOPS *
                     M3_WAVEFORM_SAMPLES_PER_HOP;
    right = index + 1U == chunk_count
                ? 0U
                : M3_WAVEFORM_RIGHT_CROP_HOPS *
                      M3_WAVEFORM_SAMPLES_PER_HOP;
    if (!m3_waveform_size_add(left, right, &cropped) ||
        cropped >= chunk->sample_count) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "waveform crop exceeds chunk length");
    }
    chunk->crop_start = left;
    chunk->copy_count = chunk->sample_count - cropped;
    if (!m3_waveform_size_add(*latent_total, (size_t)latent_length,
                              latent_total) ||
        !m3_waveform_size_add(*sample_total, chunk->copy_count,
                              sample_total)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "waveform output length overflows");
    }
    return M3_STATUS_OK;
}

static m3_status m3_waveform_plan_build(
    uint64_t frame_count, size_t chunk_count, m3_waveform_plan *plan,
    m3_error *error)
{
    const uint64_t output_prefix[] = {1U, 2U};
    m3_tensor_metadata metadata;
    size_t latent_total = 0U;
    size_t sample_total = 0U;
    size_t overlap;
    size_t expected_samples;
    size_t expected_count = 0U;
    size_t index;
    uint64_t output_shape[3];
    m3_status status;

    if (plan == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "waveform plan output is null");
    }
    (void)memset(plan, 0, sizeof(*plan));
    status = m3_waveform_chunk_count(
        frame_count, chunk_count, &expected_count, error);
    for (index = 0U; index < expected_count && status == M3_STATUS_OK;
         ++index) {
        status = m3_waveform_plan_chunk(
            frame_count, index, expected_count, &plan->chunks[index],
            &latent_total, &sample_total, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (!m3_waveform_size_multiply(
            expected_count - 1U, M3_WAVEFORM_OVERLAP_HOPS, &overlap) ||
        overlap >= latent_total ||
        !m3_waveform_size_multiply(
            latent_total - overlap, M3_WAVEFORM_SAMPLES_PER_HOP,
            &expected_samples) ||
        expected_samples != sample_total) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "waveform overlap length is inconsistent");
    }
    output_shape[0] = output_prefix[0];
    output_shape[1] = output_prefix[1];
    output_shape[2] = (uint64_t)sample_total;
    if ((size_t)output_shape[2] != sample_total) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "waveform output shape overflows");
    }
    status = m3_tensor_metadata_init(
        &metadata, M3_DTYPE_F32, 3U, output_shape, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    plan->chunk_count = expected_count;
    plan->output_samples = sample_total;
    plan->output_bytes = metadata.byte_count;
    plan->progress_total = (uint64_t)expected_count * 2U;
    return M3_STATUS_OK;
}

m3_status m3_waveform_measure(uint64_t frame_count, size_t chunk_count,
                              m3_waveform_measurement *measurement,
                              m3_error *error)
{
    m3_waveform_plan plan;
    m3_status status;

    if (measurement == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "waveform measurement output is null");
    }
    status = m3_waveform_plan_build(
        frame_count, chunk_count, &plan, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    measurement->output_samples = plan.output_samples;
    measurement->output_bytes = plan.output_bytes;
    measurement->progress_total = plan.progress_total;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

static m3_status m3_waveform_validate_chunks(
    const m3_tensor_view *chunks, m3_waveform_plan *plan,
    m3_error *error)
{
    size_t index;

    if (chunks == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "waveform chunks are null");
    }
    for (index = 0U; index < plan->chunk_count; ++index) {
        const m3_tensor_view *chunk = &chunks[index];
        const void *unused = NULL;
        m3_backend *backend;
        m3_status status = m3_tensor_const_data(chunk, &unused, error);

        if (status != M3_STATUS_OK) {
            return status;
        }
        backend = m3_storage_backend(chunk->storage);
        if (chunk->metadata.dtype != M3_DTYPE_F32 ||
            chunk->metadata.rank != 3U ||
            chunk->metadata.shape[0] != 1U ||
            chunk->metadata.shape[1] != 2U ||
            chunk->metadata.shape[2] != plan->chunks[index].sample_count ||
            !m3_tensor_is_contiguous(chunk) || backend == NULL ||
            (index != 0U && backend != plan->backend)) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "waveform chunk view is invalid");
        }
        if (index == 0U) {
            plan->backend = backend;
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_waveform_preflight(
    const m3_waveform_plan *plan, m3_error *error)
{
    m3_backend_allocation_stats stats;
    m3_backend_info info;
    uint64_t live;
    uint64_t added = (uint64_t)plan->output_bytes;
    m3_status status = m3_backend_get_info(plan->backend, &info, error);

    if (status == M3_STATUS_OK) {
        status = m3_backend_get_allocation_stats(
            plan->backend, &stats, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if ((size_t)added != plan->output_bytes) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "waveform output allocation overflows");
    }
    if (added > info.maximum_storage_bytes) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "waveform output exceeds storage limit");
    }
    live = (uint64_t)stats.live_allocated_bytes;
    if ((size_t)live != stats.live_allocated_bytes ||
        added > UINT64_MAX - live) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "waveform working set overflows");
    }
    if (info.recommended_working_set_bytes != 0U &&
        live + added > info.recommended_working_set_bytes) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "waveform exceeds recommended working set");
    }
    return M3_STATUS_OK;
}

static m3_status m3_waveform_checkpoint(
    m3_progress_callback progress, void *context, uint64_t completed,
    uint64_t total, m3_error *error)
{
    if (progress != NULL && !progress(context, completed, total)) {
        return m3_error_set(error, M3_STATUS_CANCELLED,
                            "waveform assembly was cancelled");
    }
    return M3_STATUS_OK;
}

static void m3_waveform_clamp(uint32_t *samples, size_t count)
{
    const float positive_one = 1.0F;
    const float negative_one = -1.0F;
    uint32_t positive_bits;
    uint32_t negative_bits;
    size_t index;

    (void)memcpy(&positive_bits, &positive_one, sizeof(positive_bits));
    (void)memcpy(&negative_bits, &negative_one, sizeof(negative_bits));
    for (index = 0U; index < count; ++index) {
        float value;

        (void)memcpy(&value, &samples[index], sizeof(value));
        if (value > positive_one) {
            samples[index] = positive_bits;
        } else if (value < negative_one) {
            samples[index] = negative_bits;
        }
    }
}

static m3_status m3_waveform_offset(
    size_t base, size_t sample_offset, size_t *byte_offset,
    m3_error *error)
{
    size_t bytes;

    if (!m3_waveform_size_multiply(sample_offset, sizeof(float), &bytes) ||
        !m3_waveform_size_add(base, bytes, byte_offset)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "waveform byte offset overflows");
    }
    return M3_STATUS_OK;
}

static m3_status m3_waveform_copy_channel(
    const m3_tensor_view *source, const m3_waveform_chunk_plan *chunk,
    size_t channel, const m3_waveform_plan *plan, size_t written,
    const m3_waveform_io *io, m3_storage *destination, m3_error *error)
{
    uint32_t buffer[M3_WAVEFORM_COPY_FLOATS];
    size_t source_sample;
    size_t destination_sample;
    size_t remaining = chunk->copy_count;
    m3_status status;

    if (!m3_waveform_size_multiply(
            channel, chunk->sample_count, &source_sample) ||
        !m3_waveform_size_add(
            source_sample, chunk->crop_start, &source_sample) ||
        !m3_waveform_size_multiply(
            channel, plan->output_samples, &destination_sample) ||
        !m3_waveform_size_add(
            destination_sample, written, &destination_sample)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "waveform channel offset overflows");
    }
    while (remaining != 0U) {
        size_t amount = remaining < M3_WAVEFORM_COPY_FLOATS
                            ? remaining
                            : M3_WAVEFORM_COPY_FLOATS;
        size_t byte_count = amount * sizeof(float);
        size_t source_offset = 0U;
        size_t destination_offset = 0U;

        status = m3_waveform_offset(
            source->byte_offset, source_sample, &source_offset, error);
        if (status == M3_STATUS_OK) {
            status = m3_waveform_offset(
                0U, destination_sample, &destination_offset, error);
        }
        if (status == M3_STATUS_OK) {
            status = io->read_storage(
                io->context, source->storage, source_offset, buffer,
                byte_count, error);
        }
        if (status == M3_STATUS_OK) {
            m3_waveform_clamp(buffer, amount);
            status = io->write_storage(
                io->context, destination, destination_offset, buffer,
                byte_count, error);
        }
        if (status != M3_STATUS_OK) {
            return status;
        }
        source_sample += amount;
        destination_sample += amount;
        remaining -= amount;
    }
    return M3_STATUS_OK;
}

static m3_status m3_waveform_allocate(
    const m3_waveform_plan *plan, m3_vocoder_output *built,
    m3_error *error)
{
    const uint64_t shape[] = {1U, 2U, (uint64_t)plan->output_samples};
    m3_status status = m3_storage_allocate(
        plan->backend, plan->output_bytes, 64U, &built->storage, error);

    if (status == M3_STATUS_OK) {
        status = m3_tensor_view_contiguous(
            &built->waveform, built->storage, M3_DTYPE_F32, 3U, shape,
            0U, error);
    }
    return status;
}

m3_status m3_waveform_assemble_io(
    const m3_tensor_view *chunks, size_t chunk_count,
    uint64_t frame_count, const m3_waveform_io *io,
    m3_progress_callback progress, void *progress_context,
    m3_vocoder_output *output, m3_error *error)
{
    m3_waveform_plan plan;
    m3_vocoder_output built;
    size_t written[2U] = {0U, 0U};
    size_t chunk_index;
    uint64_t completed = 0U;
    m3_status status;

    if (io == NULL || io->read_storage == NULL ||
        io->write_storage == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "waveform I/O contract is invalid");
    }
    status = m3_waveform_existing_output_validate(output, error);
    if (status == M3_STATUS_OK) {
        status = m3_waveform_plan_build(
            frame_count, chunk_count, &plan, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_waveform_validate_chunks(chunks, &plan, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_waveform_preflight(&plan, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_vocoder_output_init(&built);
    status = m3_waveform_checkpoint(
        progress, progress_context, 0U, plan.progress_total, error);
    if (status == M3_STATUS_OK) {
        status = m3_waveform_allocate(&plan, &built, error);
    }
    for (chunk_index = 0U;
         chunk_index < plan.chunk_count && status == M3_STATUS_OK;
         ++chunk_index) {
        size_t channel;

        for (channel = 0U; channel < 2U && status == M3_STATUS_OK;
             ++channel) {
            status = m3_waveform_copy_channel(
                &chunks[chunk_index], &plan.chunks[chunk_index], channel,
                &plan, written[channel], io, built.storage, error);
            if (status == M3_STATUS_OK) {
                written[channel] += plan.chunks[chunk_index].copy_count;
                ++completed;
                status = m3_waveform_checkpoint(
                    progress, progress_context, completed,
                    plan.progress_total, error);
            }
        }
    }
    if (status == M3_STATUS_OK &&
        (written[0] != plan.output_samples ||
         written[1] != plan.output_samples ||
         completed != plan.progress_total)) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "waveform copy length is inconsistent");
    }
    if (status != M3_STATUS_OK) {
        m3_vocoder_output_dispose(&built);
        return status;
    }
    m3_vocoder_output_dispose(output);
    *output = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

static m3_status m3_waveform_read_storage(
    void *context, const m3_storage *storage, size_t byte_offset,
    void *destination, size_t byte_count, m3_error *error)
{
    (void)context;
    return m3_storage_read(
        storage, byte_offset, destination, byte_count, error);
}

static m3_status m3_waveform_write_storage(
    void *context, m3_storage *storage, size_t byte_offset,
    const void *source, size_t byte_count, m3_error *error)
{
    (void)context;
    return m3_storage_write(
        storage, byte_offset, source, byte_count, error);
}

m3_status m3_waveform_assemble(
    const m3_tensor_view *chunks, size_t chunk_count,
    uint64_t frame_count, m3_progress_callback progress,
    void *progress_context, m3_vocoder_output *output, m3_error *error)
{
    const m3_waveform_io io = {
        NULL, m3_waveform_read_storage, m3_waveform_write_storage
    };

    return m3_waveform_assemble_io(
        chunks, chunk_count, frame_count, &io, progress,
        progress_context, output, error);
}

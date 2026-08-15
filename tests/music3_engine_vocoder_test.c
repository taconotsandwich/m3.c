/* SPDX-License-Identifier: GPL-2.0-only */

#include "music3_engine_test.h"

#include "m3_waveform_internal.h"

#include <stdlib.h>
#include <string.h>

static m3_status m3_music3_test_progress_range(
    m3_progress_callback progress, void *context, uint64_t first,
    uint64_t total, m3_error *error)
{
    uint64_t completed;

    for (completed = first; completed <= total; ++completed) {
        if (progress != NULL && !progress(context, completed, total)) {
            return m3_error_set(error, M3_STATUS_CANCELLED,
                                "Music3 test operation was cancelled");
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_music3_test_runtime_script(
    m3_music3_test_fixture *fixture, m3_backend *backend,
    m3_progress_callback progress, void *progress_context,
    m3_vocoder_runtime **runtime, m3_error *error)
{
    const uint64_t shape[] = {1U};
    m3_vocoder_runtime *built = calloc(1U, sizeof(*built));
    m3_storage *borrowed = NULL;
    size_t index;
    m3_status status;

    if (built == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate Music3 test runtime");
    }
    built->backend = backend;
    m3_runtime_workspace_init(&built->weights);
    built->weights.backend = backend;
    built->weights.count = 2U;
    built->weights.storages = calloc(
        built->weights.count, sizeof(*built->weights.storages));
    built->weights.views = calloc(
        built->weights.count, sizeof(*built->weights.views));
    if (built->weights.storages == NULL || built->weights.views == NULL) {
        status = m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                              "cannot allocate Music3 test weights");
        m3_vocoder_runtime_free(built);
        return status;
    } else {
        status = M3_STATUS_OK;
    }
    for (index = 0U; status == M3_STATUS_OK &&
                     index < built->weights.count; ++index) {
        status = m3_storage_allocate(
            backend, sizeof(float), 64U, &built->weights.storages[index],
            error);
        if (status == M3_STATUS_OK) {
            status = m3_tensor_view_contiguous(
                &built->weights.views[index],
                built->weights.storages[index], M3_DTYPE_F32, 1U, shape,
                0U, error);
        }
    }
    built->weights.allocated_bytes = 2U * sizeof(float);
    if (status == M3_STATUS_OK &&
        fixture->corruption == M3_MUSIC3_TEST_CORRUPT_RUNTIME_VIEW) {
        built->weights.views[0].byte_offset = 1U;
    }
    if (status == M3_STATUS_OK &&
        fixture->corruption == M3_MUSIC3_TEST_CORRUPT_RUNTIME_DUPLICATE) {
        m3_storage_free(built->weights.storages[1]);
        built->weights.storages[1] = built->weights.storages[0];
        built->weights.views[1] = built->weights.views[0];
    }
    if (status == M3_STATUS_OK &&
        fixture->corruption == M3_MUSIC3_TEST_ALIAS_OLD_RUNTIME) {
        borrowed = fixture->alias_storage;
        m3_storage_free(built->weights.storages[0]);
        built->weights.storages[0] = borrowed;
        m3_tensor_view_init(&built->weights.views[0]);
    }
    if (status == M3_STATUS_OK && progress != NULL &&
        !progress(progress_context, 0U, M3_VOCODER_SOURCE_WEIGHT_COUNT)) {
        status = m3_error_set(error, M3_STATUS_CANCELLED,
                              "Music3 runtime was cancelled");
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_test_progress_range(
            progress, progress_context, 1U,
            M3_VOCODER_SOURCE_WEIGHT_COUNT, error);
    }
    if (status != M3_STATUS_OK) {
        if (built->weights.storages != NULL &&
            built->weights.storages[0] == borrowed) {
            built->weights.storages[0] = NULL;
            if (built->weights.views != NULL) {
                m3_tensor_view_init(&built->weights.views[0]);
            }
        }
        m3_vocoder_runtime_free(built);
        return status;
    }
    *runtime = built;
    return M3_STATUS_OK;
}

static m3_status m3_music3_test_vocoder_create(
    void *context, const m3_weight_stage *vocoder,
    m3_progress_callback progress, void *progress_context,
    m3_vocoder_runtime **runtime, m3_error *error)
{
    m3_music3_test_fixture *fixture = context;
    m3_vocoder_materialize_io io;

    ++fixture->materialize_calls;
    if (!fixture->real_pipeline) {
        return m3_music3_test_runtime_script(
            fixture, fixture->engine->backend, progress,
            progress_context, runtime, error);
    }
    m3_vocoder_materialize_io_init(&io);
    return m3_vocoder_runtime_create_core(
        runtime, vocoder, &fixture->vocoder_source.plan, &io, progress,
        progress_context, error);
}

static void m3_music3_test_vocoder_free(
    void *context, m3_vocoder_runtime *runtime)
{
    (void)context;
    m3_vocoder_runtime_free(runtime);
}

static m3_status m3_music3_test_vocoder_validate(
    void *context, const m3_vocoder_runtime *runtime,
    m3_backend *backend, m3_error *error)
{
    m3_music3_test_fixture *fixture = context;
    m3_tensor_view checked;

    ++fixture->runtime_validate_calls;
    if (fixture->real_pipeline) {
        return m3_vocoder_runtime_validate(runtime, backend, error);
    }
    if (runtime == NULL || runtime->backend != backend ||
        runtime->weights.backend != backend ||
        runtime->weights.count != 2U ||
        runtime->weights.storages == NULL ||
        runtime->weights.views == NULL ||
        runtime->weights.allocated_bytes != 2U * sizeof(float)) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "Music3 test runtime is invalid");
    }
    for (size_t index = 0U; index < runtime->weights.count; ++index) {
        size_t prior;

        if (runtime->weights.storages[index] == NULL ||
            runtime->weights.views[index].storage !=
                runtime->weights.storages[index] ||
            runtime->weights.views[index].byte_offset != 0U ||
            m3_storage_backend(runtime->weights.storages[index]) !=
                backend ||
            m3_storage_size(runtime->weights.storages[index]) !=
                sizeof(float) ||
            !m3_tensor_is_contiguous(&runtime->weights.views[index])) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "Music3 test runtime is invalid");
        }
        for (prior = 0U; prior < index; ++prior) {
            if (runtime->weights.storages[prior] ==
                runtime->weights.storages[index]) {
                return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                    "Music3 test runtime owns duplicates");
            }
        }
        m3_tensor_view_init(&checked);
        if (m3_tensor_reshape(
                &runtime->weights.views[index], 1U,
                runtime->weights.views[index].metadata.shape, &checked,
                error) != M3_STATUS_OK) {
            return error == NULL ? M3_STATUS_INVALID_FORMAT :
                                   error->status;
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_music3_test_decode_script(
    m3_music3_test_fixture *fixture, const m3_tensor_view *latents,
    m3_progress_callback progress, void *progress_context,
    m3_vocoder_output *output, m3_error *error)
{
    m3_vocoder_output built;
    m3_storage *borrowed = NULL;
    uint64_t length = latents->metadata.shape[2] * 512U;
    uint64_t shape[3];
    m3_status status;

    if (fixture->corruption == M3_MUSIC3_TEST_CORRUPT_DECODE_LENGTH) {
        --length;
    }
    shape[0] = 1U;
    shape[1] = M3_MUSIC3_CHANNEL_COUNT;
    shape[2] = length;
    m3_vocoder_output_init(&built);
    if (fixture->corruption == M3_MUSIC3_TEST_ALIAS_OLD_DECODE) {
        borrowed = fixture->alias_storage;
        built.storage = borrowed;
        status = M3_STATUS_OK;
    } else if (fixture->corruption ==
                   M3_MUSIC3_TEST_CORRUPT_DECODE_DUPLICATE &&
        fixture->decode_calls > 1U) {
        built.storage = fixture->last_decoded_storage;
        status = m3_tensor_view_contiguous(
            &built.waveform, built.storage, M3_DTYPE_F32, 3U, shape, 0U,
            error);
    } else {
        status = m3_storage_allocate(
            fixture->engine->backend,
            (size_t)(2U * length) * sizeof(float), 64U, &built.storage,
            error);
    if (status == M3_STATUS_OK && borrowed == NULL) {
        status = m3_tensor_view_contiguous(
            &built.waveform, built.storage, M3_DTYPE_F32, 3U, shape,
            0U, error);
    }
    if (status == M3_STATUS_OK &&
        built.storage != fixture->last_decoded_storage &&
        built.storage != borrowed) {
        float *values = malloc((size_t)(2U * length) * sizeof(*values));
        size_t index;

        if (values == NULL) {
            status = m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                  "cannot allocate test waveform data");
        } else {
            for (index = 0U; index < (size_t)length; ++index) {
                float step = (float)(index % 4U) * 0.0625F;

                values[index] = 0.125F + step;
                values[(size_t)length + index] = -0.25F - step;
            }
            status = m3_storage_write(
                built.storage, 0U, values,
                (size_t)(2U * length) * sizeof(*values), error);
            free(values);
        }
    }
    }
    if (status == M3_STATUS_OK && progress != NULL &&
        !progress(progress_context, 0U,
                  M3_VOCODER_DECODE_OPERATION_COUNT)) {
        status = m3_error_set(error, M3_STATUS_CANCELLED,
                              "Music3 decode was cancelled");
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_test_progress_range(
            progress, progress_context, 1U,
            M3_VOCODER_DECODE_OPERATION_COUNT, error);
    }
    if (status != M3_STATUS_OK) {
        if (built.storage == fixture->last_decoded_storage ||
            built.storage == borrowed) {
            built.storage = NULL;
        }
        m3_vocoder_output_dispose(&built);
        return status;
    }
    *output = built;
    fixture->last_decoded_storage = built.storage;
    return M3_STATUS_OK;
}

static m3_status m3_music3_test_decode(
    void *context, m3_vocoder_runtime *runtime,
    const m3_tensor_view *latents, m3_progress_callback progress,
    void *progress_context, m3_vocoder_output *output,
    m3_error *error)
{
    m3_music3_test_fixture *fixture = context;

    ++fixture->decode_calls;
    if (!fixture->real_pipeline) {
        return m3_music3_test_decode_script(
            fixture, latents, progress, progress_context, output, error);
    }
    return m3_vocoder_decode_chunk(
        runtime, latents, progress, progress_context, output, error);
}

static m3_status m3_music3_test_assemble_invalid(
    m3_music3_test_fixture *fixture, uint64_t frame_count,
    size_t chunk_count, m3_progress_callback progress,
    void *progress_context, m3_vocoder_output *output, m3_error *error)
{
    m3_waveform_measurement measurement = {0U, 0U, 0U};
    m3_vocoder_output built;
    m3_storage *borrowed = NULL;
    uint64_t shape[3];
    uint64_t total = (uint64_t)chunk_count * 2U;
    m3_status status = m3_waveform_measure(
        frame_count, chunk_count, &measurement, error);

    (void)memset(shape, 0, sizeof(shape));
    if (status == M3_STATUS_OK) {
        shape[0] = 1U;
        shape[1] = M3_MUSIC3_CHANNEL_COUNT;
        shape[2] = (uint64_t)measurement.output_samples;
        if (fixture->corruption ==
            M3_MUSIC3_TEST_CORRUPT_ASSEMBLE_SHORT) {
            --shape[2];
        } else if (fixture->corruption ==
                   M3_MUSIC3_TEST_CORRUPT_ASSEMBLE_LONG) {
            ++shape[2];
        }
    }
    m3_vocoder_output_init(&built);
    if (status == M3_STATUS_OK &&
        fixture->corruption == M3_MUSIC3_TEST_ALIAS_OLD_ASSEMBLE) {
        borrowed = fixture->alias_storage;
        built.storage = borrowed;
    } else if (status == M3_STATUS_OK) {
        size_t bytes = (size_t)(2U * shape[2]) * sizeof(float);

        if (fixture->corruption ==
            M3_MUSIC3_TEST_CORRUPT_ASSEMBLE_OVERSIZE) {
            bytes += 64U;
        }
        status = m3_storage_allocate(
            fixture->engine->backend, bytes, 64U, &built.storage, error);
    }
    if (status == M3_STATUS_OK && borrowed == NULL) {
        status = m3_tensor_view_contiguous(
            &built.waveform, built.storage, M3_DTYPE_F32, 3U, shape, 0U,
            error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_music3_test_progress_range(
            progress, progress_context, 0U, total, error);
    }
    if (status != M3_STATUS_OK) {
        if (built.storage == borrowed) {
            built.storage = NULL;
            m3_tensor_view_init(&built.waveform);
        }
        m3_vocoder_output_dispose(&built);
        return status;
    }
    *output = built;
    return M3_STATUS_OK;
}

static m3_status m3_music3_test_assemble(
    void *context, const m3_tensor_view *chunks, size_t chunk_count,
    uint64_t frame_count, m3_progress_callback progress,
    void *progress_context, m3_vocoder_output *output,
    m3_error *error)
{
    m3_music3_test_fixture *fixture = context;

    ++fixture->assemble_calls;
    if (fixture->corruption == M3_MUSIC3_TEST_CORRUPT_ASSEMBLE_SHORT ||
        fixture->corruption == M3_MUSIC3_TEST_CORRUPT_ASSEMBLE_LONG ||
        fixture->corruption == M3_MUSIC3_TEST_CORRUPT_ASSEMBLE_OVERSIZE ||
        fixture->corruption == M3_MUSIC3_TEST_ALIAS_OLD_ASSEMBLE) {
        return m3_music3_test_assemble_invalid(
            fixture, frame_count, chunk_count, progress,
            progress_context, output, error);
    }
    return m3_waveform_assemble(
        chunks, chunk_count, frame_count, progress, progress_context,
        output, error);
}

void m3_music3_test_vocoder_operations(
    m3_music3_operations *operations)
{
    operations->vocoder_create = m3_music3_test_vocoder_create;
    operations->vocoder_free = m3_music3_test_vocoder_free;
    operations->vocoder_validate = m3_music3_test_vocoder_validate;
    operations->vocoder_decode = m3_music3_test_decode;
    operations->waveform_assemble = m3_music3_test_assemble;
}

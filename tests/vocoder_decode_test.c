/* SPDX-License-Identifier: GPL-2.0-only */

#include "vocoder_decode_test.h"

#include "vocoder_runtime_test.h"

#include <stdlib.h>
#include <string.h>

void m3_vocoder_decode_test_config(m3_vocoder_plan_config *config)
{
    size_t block;

    (void)memset(config, 0, sizeof(*config));
    config->latent_channels = 2U;
    config->maximum_latent_length = 4U;
    config->decoder_input_channels = 1U;
    config->decoder_output_channels = 2U;
    config->initial_channels = 16U;
    config->block_count = M3_VOCODER_BLOCK_COUNT;
    config->residual_count = M3_VOCODER_RESIDUAL_COUNT;
    for (block = 0U; block < M3_VOCODER_BLOCK_COUNT; ++block) {
        config->strides[block] = 2U;
    }
}

static bool m3_vocoder_decode_test_rewrite(
    m3_vocoder_test_fixture *fixture, m3_error *error)
{
    size_t index;

    for (index = 0U; index < fixture->stage.view_count; ++index) {
        m3_tensor_view *view = &fixture->stage.views[index];
        const char *name = fixture->table.bindings[index].name;
        float *values = malloc(view->metadata.byte_count);
        size_t element;
        m3_status status;

        if (values == NULL) {
            (void)m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                               "cannot allocate decoder test values");
            return false;
        }
        status = m3_storage_read(
            view->storage, view->byte_offset, values,
            view->metadata.byte_count, error);
        for (element = 0U; element < view->metadata.element_count &&
                           status == M3_STATUS_OK; ++element) {
            if (strstr(name, "weight_g") != NULL) {
                values[element] *= strstr(name, "res_unit") != NULL
                                       ? 0.05F
                                       : 0.5F;
            } else if (strstr(name, "weight_v") != NULL) {
                continue;
            } else if (strstr(name, ".bias") != NULL) {
                values[element] = 0.0F;
            } else if (strstr(name, ".alpha") != NULL) {
                values[element] = 0.5F + 0.01F * (float)element;
            } else {
                values[element] = element == 0U ? 0.75F : -0.5F;
            }
        }
        if (status == M3_STATUS_OK) {
            status = m3_storage_write(
                view->storage, view->byte_offset, values,
                view->metadata.byte_count, error);
        }
        free(values);
        if (status != M3_STATUS_OK) {
            return false;
        }
    }
    return true;
}

bool m3_vocoder_decode_test_runtime_config(
    m3_backend *backend, const m3_vocoder_plan_config *config,
    m3_vocoder_runtime **runtime, m3_error *error)
{
    m3_vocoder_test_fixture fixture = {0};
    m3_vocoder_materialize_io io;
    m3_status status;

    if (backend == NULL || config == NULL || runtime == NULL ||
        *runtime != NULL) {
        return false;
    }
    m3_vocoder_materialize_io_init(&io);
    if (!m3_vocoder_test_fixture_create_config(
            &fixture, backend, false, config, error) ||
        !m3_vocoder_decode_test_rewrite(&fixture, error)) {
        m3_vocoder_test_fixture_dispose(&fixture);
        return false;
    }
    status = m3_vocoder_runtime_create_core(
        runtime, &fixture.stage, &fixture.plan, &io, NULL, NULL, error);
    m3_vocoder_test_fixture_dispose(&fixture);
    return status == M3_STATUS_OK;
}

bool m3_vocoder_decode_test_runtime(
    m3_backend *backend, m3_vocoder_runtime **runtime, m3_error *error)
{
    m3_vocoder_plan_config config;

    m3_vocoder_decode_test_config(&config);
    return m3_vocoder_decode_test_runtime_config(
        backend, &config, runtime, error);
}

bool m3_vocoder_decode_test_latents(
    m3_backend *backend, uint64_t length, const float *values,
    m3_storage **storage, m3_tensor_view *view, m3_error *error)
{
    const uint64_t shape[] = {1U, 2U, length};
    m3_tensor_metadata metadata;
    m3_status status;

    if (backend == NULL || values == NULL || storage == NULL ||
        *storage != NULL || view == NULL) {
        return false;
    }
    m3_tensor_view_init(view);
    status = m3_tensor_metadata_init(
        &metadata, M3_DTYPE_F32, 3U, shape, error);
    if (status == M3_STATUS_OK) {
        status = m3_storage_allocate(
            backend, metadata.byte_count, 64U, storage, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tensor_view_contiguous(
            view, *storage, M3_DTYPE_F32, 3U, shape, 0U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_storage_write(
            *storage, 0U, values, metadata.byte_count, error);
    }
    if (status != M3_STATUS_OK) {
        m3_storage_free(*storage);
        *storage = NULL;
    }
    return status == M3_STATUS_OK;
}

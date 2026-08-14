/* SPDX-License-Identifier: GPL-2.0-only */

#include "vocoder_runtime_test.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    const m3_tensor_metadata *metadata;
} m3_vocoder_test_source_spec;

static int m3_vocoder_test_compare_source(const void *left,
                                           const void *right)
{
    const m3_vocoder_test_source_spec *left_source = left;
    const m3_vocoder_test_source_spec *right_source = right;

    return strcmp(left_source->name, right_source->name);
}

static char *m3_vocoder_test_copy_name(const char *name)
{
    size_t length = strlen(name);
    char *copy = malloc(length + 1U);

    if (copy != NULL) {
        (void)memcpy(copy, name, length + 1U);
    }
    return copy;
}

void m3_vocoder_test_config(m3_vocoder_plan_config *config)
{
    (void)memset(config, 0, sizeof(*config));
    config->latent_channels = 4U;
    config->maximum_latent_length = 8U;
    config->decoder_input_channels = 2U;
    config->decoder_output_channels = 3U;
    config->initial_channels = 4U;
    config->block_count = 1U;
    config->residual_count = 1U;
    config->strides[0] = 2U;
}

static bool m3_vocoder_test_collect_sources(
    const m3_vocoder_plan *plan, m3_vocoder_test_source_spec *sources)
{
    size_t source_index = 0U;
    size_t entry;

    for (entry = 0U; entry < plan->entry_count; ++entry) {
        size_t count = plan->entries[entry].kind ==
                               M3_VOCODER_MATERIAL_COPY
                           ? 1U
                           : 2U;
        size_t source;

        for (source = 0U; source < count; ++source) {
            if (source_index >= plan->source_count) {
                return false;
            }
            sources[source_index].name =
                plan->entries[entry].source_names[source];
            sources[source_index].metadata =
                &plan->entries[entry].source_metadata[source];
            ++source_index;
        }
    }
    return source_index == plan->source_count;
}

static bool m3_vocoder_test_default_values(
    const char *name, const m3_tensor_metadata *metadata, float *values)
{
    size_t row_count = (size_t)metadata->shape[0];
    size_t row_elements = metadata->element_count / row_count;
    size_t row;

    for (row = 0U; row < row_count; ++row) {
        size_t element;

        for (element = 0U; element < row_elements; ++element) {
            size_t index = row * row_elements + element;

            if (strstr(name, "weight_g") != NULL) {
                values[index] = 1.0F + 0.25F * (float)row;
            } else if (strstr(name, "weight_v") != NULL) {
                int centered = (int)(element % 7U) - 3;

                values[index] = 0.25F * (float)centered +
                                0.125F * (float)(row + 1U);
            } else {
                values[index] = 0.03125F * (float)(index + 1U);
            }
        }
    }
    return true;
}

static bool m3_vocoder_test_allocate_source(
    m3_vocoder_test_fixture *fixture,
    const m3_vocoder_test_source_spec *source, size_t index,
    m3_error *error)
{
    m3_weight_binding *binding = &fixture->table.bindings[index];
    float *values;
    m3_status status;

    binding->name = m3_vocoder_test_copy_name(source->name);
    if (binding->name == NULL) {
        (void)m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                           "cannot allocate vocoder fixture name");
        return false;
    }
    binding->tensor = *source->metadata;
    status = m3_storage_allocate(
        fixture->backend, source->metadata->byte_count, 64U,
        &fixture->stage.storages[index], error);
    if (status == M3_STATUS_OK) {
        status = m3_tensor_view_contiguous(
            &fixture->stage.views[index], fixture->stage.storages[index],
            M3_DTYPE_F32, source->metadata->rank, source->metadata->shape,
            0U, error);
    }
    if (status != M3_STATUS_OK) {
        return false;
    }
    values = malloc(source->metadata->byte_count);
    if (values == NULL) {
        (void)m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                           "cannot allocate vocoder fixture values");
        return false;
    }
    (void)m3_vocoder_test_default_values(
        source->name, source->metadata, values);
    status = m3_storage_write(
        fixture->stage.storages[index], 0U, values,
        source->metadata->byte_count, error);
    free(values);
    return status == M3_STATUS_OK;
}

bool m3_vocoder_test_fixture_create_config(
    m3_vocoder_test_fixture *fixture, m3_backend *backend,
    bool owns_backend, const m3_vocoder_plan_config *config,
    m3_error *error)
{
    m3_vocoder_test_source_spec *sources = NULL;
    size_t index;
    m3_status status;

    if (fixture == NULL || backend == NULL || config == NULL) {
        return false;
    }
    (void)memset(fixture, 0, sizeof(*fixture));
    m3_vocoder_plan_init(&fixture->plan);
    m3_weight_table_init(&fixture->table);
    m3_weight_stage_init(&fixture->stage);
    fixture->backend = backend;
    fixture->owns_backend = owns_backend;
    status = m3_vocoder_plan_build(config, &fixture->plan, error);
    if (status != M3_STATUS_OK) {
        m3_vocoder_test_fixture_dispose(fixture);
        return false;
    }
    sources = calloc(fixture->plan.source_count, sizeof(*sources));
    fixture->table.bindings = calloc(
        fixture->plan.source_count, sizeof(*fixture->table.bindings));
    fixture->stage.storages = calloc(
        fixture->plan.source_count, sizeof(*fixture->stage.storages));
    fixture->stage.views = calloc(
        fixture->plan.source_count, sizeof(*fixture->stage.views));
    if (sources == NULL || fixture->table.bindings == NULL ||
        fixture->stage.storages == NULL || fixture->stage.views == NULL ||
        !m3_vocoder_test_collect_sources(&fixture->plan, sources)) {
        free(sources);
        (void)m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                           "cannot allocate vocoder fixture state");
        m3_vocoder_test_fixture_dispose(fixture);
        return false;
    }
    qsort(sources, fixture->plan.source_count, sizeof(*sources),
          m3_vocoder_test_compare_source);
    fixture->table.binding_count = fixture->plan.source_count;
    fixture->stage.table = &fixture->table;
    fixture->stage.backend = backend;
    fixture->stage.storage_count = fixture->plan.source_count;
    fixture->stage.view_count = fixture->plan.source_count;
    for (index = 0U; index < fixture->plan.source_count; ++index) {
        if (!m3_vocoder_test_allocate_source(
                fixture, &sources[index], index, error)) {
            free(sources);
            m3_vocoder_test_fixture_dispose(fixture);
            return false;
        }
    }
    free(sources);
    return true;
}

bool m3_vocoder_test_fixture_create(
    m3_vocoder_test_fixture *fixture, m3_backend *backend,
    bool owns_backend, m3_error *error)
{
    m3_vocoder_plan_config config;

    m3_vocoder_test_config(&config);
    return m3_vocoder_test_fixture_create_config(
        fixture, backend, owns_backend, &config, error);
}

void m3_vocoder_test_fixture_dispose(m3_vocoder_test_fixture *fixture)
{
    m3_backend *backend;
    bool owns_backend;

    if (fixture == NULL) {
        return;
    }
    backend = fixture->backend;
    owns_backend = fixture->owns_backend;
    m3_weight_stage_dispose(&fixture->stage);
    m3_weight_table_dispose(&fixture->table);
    m3_vocoder_plan_dispose(&fixture->plan);
    if (owns_backend) {
        m3_backend_free(backend);
    }
    (void)memset(fixture, 0, sizeof(*fixture));
}

m3_tensor_view *m3_vocoder_test_source(
    m3_vocoder_test_fixture *fixture, const char *name)
{
    const m3_tensor_view *view = m3_weight_stage_find_view(
        &fixture->stage, name);

    return (m3_tensor_view *)view;
}

bool m3_vocoder_test_fill(m3_tensor_view *view, float value,
                          m3_error *error)
{
    float *values;
    size_t index;
    m3_status status;

    if (view == NULL) {
        return false;
    }
    values = malloc(view->metadata.byte_count);
    if (values == NULL) {
        return false;
    }
    for (index = 0U; index < view->metadata.element_count; ++index) {
        values[index] = value;
    }
    status = m3_storage_write(
        view->storage, view->byte_offset, values,
        view->metadata.byte_count, error);
    free(values);
    return status == M3_STATUS_OK;
}

bool m3_vocoder_test_write_values(m3_tensor_view *view,
                                  const float *values, size_t count,
                                  m3_error *error)
{
    if (view == NULL || values == NULL ||
        count != view->metadata.element_count) {
        return false;
    }
    return m3_storage_write(
               view->storage, view->byte_offset, values,
               view->metadata.byte_count, error) == M3_STATUS_OK;
}

bool m3_vocoder_test_read_values(const m3_tensor_view *view,
                                 float *values, size_t count,
                                 m3_error *error)
{
    if (view == NULL || values == NULL ||
        count != view->metadata.element_count) {
        return false;
    }
    return m3_storage_read(
               view->storage, view->byte_offset, values,
               view->metadata.byte_count, error) == M3_STATUS_OK;
}

size_t m3_vocoder_test_source_bytes(const m3_vocoder_test_fixture *fixture)
{
    size_t total = 0U;
    size_t index;

    for (index = 0U; index < fixture->stage.view_count; ++index) {
        total += fixture->stage.views[index].metadata.byte_count;
    }
    return total;
}

size_t m3_vocoder_test_runtime_bytes(const m3_vocoder_test_fixture *fixture)
{
    size_t total = 0U;
    size_t index;

    for (index = 0U; index < fixture->plan.entry_count; ++index) {
        total += fixture->plan.entries[index].output_metadata.byte_count;
    }
    return total;
}

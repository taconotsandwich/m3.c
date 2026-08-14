/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_vocoder_internal.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const m3_vocoder_plan *plan;
    const m3_weight_stage *stage;
    const m3_vocoder_materialize_io *io;
    m3_vocoder_runtime *runtime;
    m3_progress_callback progress;
    void *progress_context;
    unsigned char *row;
    float *gains;
    size_t gain_capacity;
    size_t completed;
} m3_vocoder_build;

static m3_status m3_vocoder_default_read(
    void *context, const m3_storage *storage, size_t byte_offset,
    void *destination, size_t byte_count, m3_error *error)
{
    (void)context;
    return m3_storage_read(storage, byte_offset, destination, byte_count,
                           error);
}

static m3_status m3_vocoder_default_write(
    void *context, m3_storage *storage, size_t byte_offset,
    const void *source, size_t byte_count, m3_error *error)
{
    (void)context;
    return m3_storage_write(storage, byte_offset, source, byte_count,
                            error);
}

void m3_vocoder_materialize_io_init(m3_vocoder_materialize_io *io)
{
    if (io != NULL) {
        io->context = NULL;
        io->read_storage = m3_vocoder_default_read;
        io->write_storage = m3_vocoder_default_write;
    }
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

static m3_status m3_vocoder_row_bytes(
    const m3_tensor_metadata *metadata, size_t *row_bytes,
    m3_error *error)
{
    uint64_t rows;

    if (metadata->dtype != M3_DTYPE_F32 || metadata->rank == 0U ||
        metadata->shape[0] == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "vocoder materialization tensor is invalid");
    }
    rows = metadata->shape[0];
    if (rows > (uint64_t)SIZE_MAX ||
        metadata->byte_count % (size_t)rows != 0U) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "vocoder materialization row size overflows");
    }
    *row_bytes = metadata->byte_count / (size_t)rows;
    if (*row_bytes == 0U || *row_bytes > M3_VOCODER_MAXIMUM_ROW_BYTES) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "vocoder materialization row is too large");
    }
    return M3_STATUS_OK;
}

static m3_status m3_vocoder_validate_source(
    const m3_weight_stage *stage, const char *name,
    const m3_tensor_metadata *expected, const m3_tensor_view **view,
    m3_error *error)
{
    const m3_weight_binding *binding = m3_weight_table_find(
        stage->table, name);
    size_t index;

    *view = NULL;
    if (binding == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "vocoder weight '%s' is missing", name);
    }
    index = (size_t)(binding - stage->table->bindings);
    if (index >= stage->view_count ||
        !m3_vocoder_metadata_equal(&binding->tensor, expected) ||
        !m3_vocoder_metadata_equal(&stage->views[index].metadata,
                                    expected)) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "vocoder weight '%s' has the wrong tensor contract",
                            name);
    }
    if (!m3_tensor_is_contiguous(&stage->views[index]) ||
        stage->views[index].storage == NULL ||
        m3_storage_backend(stage->views[index].storage) != stage->backend) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "vocoder weight '%s' is not a staged tensor",
                            name);
    }
    *view = &stage->views[index];
    return M3_STATUS_OK;
}

static m3_status m3_vocoder_validate_entry(
    const m3_weight_stage *stage, const m3_vocoder_plan_entry *entry,
    size_t *maximum_rows, m3_error *error)
{
    const m3_tensor_view *unused;
    size_t row_bytes;
    size_t source_count;
    size_t source;
    m3_status status;

    if (entry->output_name[0] == '\0' ||
        (entry->kind != M3_VOCODER_MATERIAL_COPY &&
         entry->kind != M3_VOCODER_MATERIAL_WEIGHT_NORM)) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "vocoder materialization plan is invalid");
    }
    source_count = entry->kind == M3_VOCODER_MATERIAL_COPY ? 1U : 2U;
    for (source = 0U; source < source_count; ++source) {
        status = m3_vocoder_validate_source(
            stage, entry->source_names[source],
            &entry->source_metadata[source], &unused, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    status = m3_vocoder_row_bytes(
        &entry->output_metadata, &row_bytes, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (entry->kind == M3_VOCODER_MATERIAL_COPY) {
        if (!m3_vocoder_metadata_equal(
                &entry->source_metadata[0], &entry->output_metadata)) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "vocoder copy plan metadata disagrees");
        }
        return M3_STATUS_OK;
    }
    if (entry->source_metadata[0].rank != 3U ||
        entry->source_metadata[0].shape[0] !=
            entry->source_metadata[1].shape[0] ||
        entry->source_metadata[0].shape[1] != 1U ||
        entry->source_metadata[0].shape[2] != 1U ||
        !m3_vocoder_metadata_equal(
            &entry->source_metadata[1], &entry->output_metadata) ||
        entry->source_metadata[0].shape[0] > (uint64_t)SIZE_MAX) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "vocoder weight normalization plan disagrees");
    }
    if ((size_t)entry->source_metadata[0].shape[0] > *maximum_rows) {
        *maximum_rows = (size_t)entry->source_metadata[0].shape[0];
    }
    return M3_STATUS_OK;
}

static m3_status m3_vocoder_validate_build(
    const m3_weight_stage *stage, const m3_vocoder_plan *plan,
    const m3_vocoder_materialize_io *io, size_t *maximum_rows,
    m3_error *error)
{
    size_t source_count = 0U;
    size_t entry;

    if (stage == NULL || stage->backend == NULL || stage->table == NULL ||
        plan == NULL || plan->entry_count == 0U || plan->entries == NULL ||
        io == NULL || io->read_storage == NULL ||
        io->write_storage == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "vocoder stage, plan, and I/O are required");
    }
    if (stage->table->binding_count != plan->source_count ||
        stage->view_count != plan->source_count ||
        (stage->view_count != 0U && stage->views == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "vocoder stage does not contain the exact inventory");
    }
    *maximum_rows = 0U;
    for (entry = 0U; entry < plan->entry_count; ++entry) {
        m3_status status = m3_vocoder_validate_entry(
            stage, &plan->entries[entry], maximum_rows, error);

        if (status != M3_STATUS_OK) {
            return status;
        }
        source_count += plan->entries[entry].kind ==
                                M3_VOCODER_MATERIAL_COPY
                            ? 1U
                            : 2U;
    }
    if (source_count != plan->source_count || *maximum_rows == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "vocoder plan source count is inconsistent");
    }
    return M3_STATUS_OK;
}

static m3_status m3_vocoder_checkpoint(m3_vocoder_build *build,
                                       m3_error *error)
{
    ++build->completed;
    if (build->progress != NULL &&
        !build->progress(build->progress_context,
                         (uint64_t)build->completed,
                         (uint64_t)build->plan->source_count)) {
        return m3_error_set(error, M3_STATUS_CANCELLED,
                            "vocoder weight materialization was cancelled");
    }
    return M3_STATUS_OK;
}

static m3_status m3_vocoder_read(
    m3_vocoder_build *build, const m3_tensor_view *view,
    size_t relative_offset, void *destination, size_t byte_count,
    m3_error *error)
{
    if (relative_offset > SIZE_MAX - view->byte_offset) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "vocoder source offset overflows");
    }
    return build->io->read_storage(
        build->io->context, view->storage,
        view->byte_offset + relative_offset, destination, byte_count,
        error);
}

static m3_status m3_vocoder_write(
    m3_vocoder_build *build, m3_tensor_view *view, size_t relative_offset,
    const void *source, size_t byte_count, m3_error *error)
{
    if (relative_offset > SIZE_MAX - view->byte_offset) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "vocoder destination offset overflows");
    }
    return build->io->write_storage(
        build->io->context, view->storage,
        view->byte_offset + relative_offset, source, byte_count, error);
}

static m3_status m3_vocoder_finite_values(
    const char *name, const float *values, size_t count, m3_error *error)
{
    size_t index;

    for (index = 0U; index < count; ++index) {
        if (!isfinite(values[index])) {
            return m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "vocoder weight '%s' contains a non-finite value", name);
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_vocoder_copy_entry(
    m3_vocoder_build *build, const m3_vocoder_plan_entry *entry,
    m3_tensor_view *output, m3_error *error)
{
    const m3_tensor_view *source = NULL;
    size_t row_bytes = 0U;
    size_t row;
    size_t rows = (size_t)entry->output_metadata.shape[0];
    m3_status status = m3_vocoder_validate_source(
        build->stage, entry->source_names[0],
        &entry->source_metadata[0], &source, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (source == NULL) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "vocoder copy source was not resolved");
    }
    status = m3_vocoder_row_bytes(
        &entry->output_metadata, &row_bytes, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    for (row = 0U; row < rows && status == M3_STATUS_OK; ++row) {
        size_t offset = row * row_bytes;

        status = m3_vocoder_read(
            build, source, offset, build->row, row_bytes, error);
        if (status == M3_STATUS_OK) {
            status = m3_vocoder_finite_values(
                entry->source_names[0], (const float *)build->row,
                row_bytes / sizeof(float), error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_vocoder_write(
                build, output, offset, build->row, row_bytes, error);
        }
    }
    return status == M3_STATUS_OK ? m3_vocoder_checkpoint(build, error)
                                  : status;
}

static m3_status m3_vocoder_read_gains(
    m3_vocoder_build *build, const m3_vocoder_plan_entry *entry,
    const m3_tensor_view *gain_view, size_t rows, m3_error *error)
{
    size_t byte_count;
    size_t offset = 0U;
    size_t index;
    m3_status status;

    if (rows > build->gain_capacity || rows > SIZE_MAX / sizeof(float)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "vocoder gain storage overflows");
    }
    byte_count = rows * sizeof(float);
    while (offset < byte_count) {
        size_t chunk = byte_count - offset;

        if (chunk > M3_VOCODER_MAXIMUM_ROW_BYTES) {
            chunk = M3_VOCODER_MAXIMUM_ROW_BYTES;
        }
        status = m3_vocoder_read(
            build, gain_view, offset,
            (unsigned char *)build->gains + offset, chunk, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        offset += chunk;
    }
    for (index = 0U; index < rows; ++index) {
        if (!isfinite(build->gains[index])) {
            return m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "vocoder gain '%s' contains a non-finite value",
                entry->source_names[0]);
        }
    }
    return m3_vocoder_checkpoint(build, error);
}

#pragma STDC FP_CONTRACT OFF
static m3_status m3_vocoder_normalize_row(
    const char *name, float gain, float *values, size_t count,
    m3_error *error)
{
    float square_sum = 0.0F;
    float norm;
    volatile float scale;
    size_t index;
    m3_status status = m3_vocoder_finite_values(
        name, values, count, error);

    if (status != M3_STATUS_OK) {
        return status;
    }

    for (index = 0U; index < count; ++index) {
        volatile float product;
        volatile float next;

        product = values[index] * values[index];
        next = square_sum + product;
        square_sum = next;
    }
    norm = sqrtf(square_sum);
    if (!isfinite(norm) || !(norm > 0.0F)) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "vocoder weight '%s' has an invalid norm", name);
    }
    scale = gain / norm;
    if (!isfinite(scale)) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "vocoder weight '%s' has a non-finite scale",
                            name);
    }
    for (index = 0U; index < count; ++index) {
        volatile float product = values[index] * scale;

        if (!isfinite(product)) {
            return m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "vocoder weight '%s' normalizes to a non-finite value",
                name);
        }
        values[index] = product;
    }
    return M3_STATUS_OK;
}
#pragma STDC FP_CONTRACT DEFAULT

static m3_status m3_vocoder_norm_entry(
    m3_vocoder_build *build, const m3_vocoder_plan_entry *entry,
    m3_tensor_view *output, m3_error *error)
{
    const m3_tensor_view *gain_view = NULL;
    const m3_tensor_view *value_view = NULL;
    size_t row_bytes = 0U;
    size_t rows = (size_t)entry->output_metadata.shape[0];
    size_t row;
    m3_status status = m3_vocoder_validate_source(
        build->stage, entry->source_names[0],
        &entry->source_metadata[0], &gain_view, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    status = m3_vocoder_validate_source(
        build->stage, entry->source_names[1],
        &entry->source_metadata[1], &value_view, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (gain_view == NULL || value_view == NULL) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "vocoder norm sources were not resolved");
    }
    status = m3_vocoder_row_bytes(
        &entry->output_metadata, &row_bytes, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    status = m3_vocoder_read_gains(
        build, entry, gain_view, rows, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    for (row = 0U; row < rows && status == M3_STATUS_OK; ++row) {
        size_t offset = row * row_bytes;

        status = m3_vocoder_read(
            build, value_view, offset, build->row, row_bytes, error);
        if (status == M3_STATUS_OK) {
            status = m3_vocoder_normalize_row(
                entry->source_names[1], build->gains[row],
                (float *)build->row, row_bytes / sizeof(float), error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_vocoder_write(
                build, output, offset, build->row, row_bytes, error);
        }
    }
    return status == M3_STATUS_OK ? m3_vocoder_checkpoint(build, error)
                                  : status;
}

static m3_status m3_vocoder_build_weights(m3_vocoder_build *build,
                                          m3_error *error)
{
    size_t entry;

    for (entry = 0U; entry < build->plan->entry_count; ++entry) {
        const m3_vocoder_plan_entry *spec = &build->plan->entries[entry];
        m3_tensor_view *output = &build->runtime->weights.views[entry];
        m3_status status;

        if (spec->kind == M3_VOCODER_MATERIAL_COPY) {
            status = m3_vocoder_copy_entry(build, spec, output, error);
        } else {
            status = m3_vocoder_norm_entry(build, spec, output, error);
        }
        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_vocoder_allocate_runtime(
    m3_vocoder_runtime *runtime, m3_backend *backend,
    const m3_vocoder_plan *plan, m3_error *error)
{
    m3_runtime_tensor_spec *specs;
    size_t index;
    m3_status status;

    if (plan->entry_count > SIZE_MAX / sizeof(*specs)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "vocoder runtime specification overflows");
    }
    specs = calloc(plan->entry_count, sizeof(*specs));
    if (specs == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate vocoder runtime specification");
    }
    for (index = 0U; index < plan->entry_count; ++index) {
        uint8_t axis;

        specs[index].dtype = M3_DTYPE_F32;
        specs[index].rank = plan->entries[index].output_metadata.rank;
        specs[index].alignment = 64U;
        for (axis = 0U; axis < specs[index].rank; ++axis) {
            specs[index].shape[axis] =
                plan->entries[index].output_metadata.shape[axis];
        }
    }
    status = m3_runtime_workspace_build(
        &runtime->weights, backend, specs, plan->entry_count, error);
    free(specs);
    return status;
}

m3_status m3_vocoder_runtime_create_core(
    m3_vocoder_runtime **runtime, const m3_weight_stage *vocoder,
    const m3_vocoder_plan *plan, const m3_vocoder_materialize_io *io,
    m3_progress_callback progress, void *progress_context,
    m3_error *error)
{
    m3_vocoder_runtime *built;
    m3_vocoder_build build;
    size_t maximum_rows = 0U;
    m3_status status;

    if (runtime == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "vocoder runtime output is null");
    }
    if (*runtime != NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "vocoder runtime output must be empty");
    }
    status = m3_vocoder_validate_build(
        vocoder, plan, io, &maximum_rows, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (progress != NULL &&
        !progress(progress_context, 0U, (uint64_t)plan->source_count)) {
        return m3_error_set(error, M3_STATUS_CANCELLED,
                            "vocoder weight materialization was cancelled");
    }
    built = calloc(1U, sizeof(*built));
    if (built == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate vocoder runtime state");
    }
    built->backend = vocoder->backend;
    m3_runtime_workspace_init(&built->weights);
    (void)memset(&build, 0, sizeof(build));
    build.plan = plan;
    build.stage = vocoder;
    build.io = io;
    build.runtime = built;
    build.progress = progress;
    build.progress_context = progress_context;
    status = m3_vocoder_allocate_runtime(
        built, vocoder->backend, plan, error);
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_runtime_bind(built, plan, error);
    }
    if (status == M3_STATUS_OK) {
        build.row = malloc(M3_VOCODER_MAXIMUM_ROW_BYTES);
        if (maximum_rows > SIZE_MAX / sizeof(*build.gains)) {
            status = m3_error_set(error, M3_STATUS_OVERFLOW,
                                  "vocoder gain scratch size overflows");
        } else {
            build.gains = malloc(maximum_rows * sizeof(*build.gains));
            build.gain_capacity = maximum_rows;
        }
        if (status == M3_STATUS_OK &&
            (build.row == NULL || build.gains == NULL)) {
            status = m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                  "cannot allocate vocoder row scratch");
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_vocoder_build_weights(&build, error);
    }
    free(build.gains);
    free(build.row);
    if (status != M3_STATUS_OK) {
        m3_vocoder_runtime_free(built);
        return status;
    }
    *runtime = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

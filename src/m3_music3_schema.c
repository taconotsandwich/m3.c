/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_schema_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static const m3_music3_schema_summary m3_music3_summaries[] = {
    {399U, UINT64_C(8584475648), UINT64_C(17168951296), M3_DTYPE_BF16},
    {47U, UINT64_C(646025216), UINT64_C(1292050432), M3_DTYPE_BF16},
    {4U, UINT64_C(25167881), UINT64_C(100671524), M3_DTYPE_F32},
    {441U, UINT64_C(2431905920), UINT64_C(9727623680), M3_DTYPE_F32},
    {121U, UINT64_C(54170722), UINT64_C(216682888), M3_DTYPE_F32}
};

m3_status m3_music3_schema_emit(m3_music3_schema_emitter *emitter,
                                const char *name, m3_dtype dtype,
                                uint8_t rank, const uint64_t *shape)
{
    m3_tensor_metadata tensor;
    m3_status status;

    status = m3_tensor_metadata_init(&tensor, dtype, rank, shape,
                                     emitter->error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    return emitter->visitor(name, &tensor, emitter->context, emitter->error);
}

m3_status m3_music3_schema_emit1(m3_music3_schema_emitter *emitter,
                                 const char *name, uint64_t first)
{
    const uint64_t shape[] = {first};

    return m3_music3_schema_emit(emitter, name, M3_DTYPE_F32, 1U, shape);
}

m3_status m3_music3_schema_emit2(m3_music3_schema_emitter *emitter,
                                 const char *name, uint64_t first,
                                 uint64_t second)
{
    const uint64_t shape[] = {first, second};

    return m3_music3_schema_emit(emitter, name, M3_DTYPE_F32, 2U, shape);
}

m3_status m3_music3_schema_emit3(m3_music3_schema_emitter *emitter,
                                 const char *name, uint64_t first,
                                 uint64_t second, uint64_t third)
{
    const uint64_t shape[] = {first, second, third};

    return m3_music3_schema_emit(emitter, name, M3_DTYPE_F32, 3U, shape);
}

m3_status m3_music3_schema_format(char *name, size_t capacity,
                                  m3_error *error, const char *format, ...)
{
    va_list arguments;
    int count;

    va_start(arguments, format);
    count = vsnprintf(name, capacity, format, arguments);
    va_end(arguments);
    if (count < 0 || (size_t)count >= capacity) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Music3 tensor name overflows");
    }
    return M3_STATUS_OK;
}

m3_status m3_music3_schema_visit(m3_component_id id,
                                 m3_music3_schema_visitor visitor,
                                 void *context, m3_error *error)
{
    m3_music3_schema_emitter emitter;
    m3_status status;

    if (!m3_component_contains_weights(id) || visitor == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 schema visitor argument is invalid");
    }
    emitter.visitor = visitor;
    emitter.context = context;
    emitter.error = error;
    switch (id) {
    case M3_COMPONENT_LANGUAGE_MODEL:
        status = m3_music3_schema_generate_lm(&emitter);
        break;
    case M3_COMPONENT_RVQ_DEPTH_DECODER:
        status = m3_music3_schema_generate_rvq(&emitter);
        break;
    case M3_COMPONENT_CONDITION_ENCODER:
        status = m3_music3_schema_generate_condition(&emitter);
        break;
    case M3_COMPONENT_TRANSFORMER:
        status = m3_music3_schema_generate_flow(&emitter);
        break;
    case M3_COMPONENT_VOCODER:
        status = m3_music3_schema_generate_vocoder(&emitter);
        break;
    default:
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "component has no Music3 tensor schema");
    }
    if (status == M3_STATUS_OK) {
        m3_error_reset(error);
    }
    return status;
}

m3_status m3_music3_schema_expected_summary(
    m3_component_id id, m3_music3_schema_summary *summary, m3_error *error)
{
    if (!m3_component_contains_weights(id) || summary == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 schema summary argument is invalid");
    }
    *summary = m3_music3_summaries[(size_t)id];
    m3_error_reset(error);
    return M3_STATUS_OK;
}

typedef struct {
    const char *name;
    const m3_tensor_metadata *actual;
    bool found;
    bool matches;
} m3_schema_find_context;

static bool m3_schema_tensor_equal(const m3_tensor_metadata *left,
                                   const m3_tensor_metadata *right)
{
    uint8_t index;

    if (left->dtype != right->dtype || left->rank != right->rank ||
        left->element_count != right->element_count ||
        left->byte_count != right->byte_count) {
        return false;
    }
    for (index = 0U; index < left->rank; ++index) {
        if (left->shape[index] != right->shape[index]) {
            return false;
        }
    }
    return true;
}

static m3_status m3_schema_find_visitor(
    const char *name, const m3_tensor_metadata *tensor, void *context,
    m3_error *error)
{
    m3_schema_find_context *find = context;

    (void)error;
    if (strcmp(name, find->name) == 0) {
        find->found = true;
        find->matches = m3_schema_tensor_equal(tensor, find->actual);
    }
    return M3_STATUS_OK;
}

typedef struct {
    size_t count;
    uint64_t elements;
    uint64_t bytes;
    m3_dtype dtype;
    bool one_dtype;
} m3_schema_generated_summary;

static m3_status m3_schema_summary_visitor(
    const char *name, const m3_tensor_metadata *tensor, void *context,
    m3_error *error)
{
    m3_schema_generated_summary *summary = context;

    (void)name;
    if (summary->count == SIZE_MAX ||
        (uint64_t)tensor->element_count > UINT64_MAX - summary->elements ||
        (uint64_t)tensor->byte_count > UINT64_MAX - summary->bytes) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Music3 schema aggregate overflows");
    }
    if (summary->count == 0U) {
        summary->dtype = tensor->dtype;
    } else if (summary->dtype != tensor->dtype) {
        summary->one_dtype = false;
    }
    summary->count += 1U;
    summary->elements += (uint64_t)tensor->element_count;
    summary->bytes += (uint64_t)tensor->byte_count;
    return M3_STATUS_OK;
}

static m3_status m3_schema_verify_generated(m3_component_id id,
                                             m3_error *error)
{
    m3_schema_generated_summary generated = {0U, 0U, 0U, M3_DTYPE_F32,
                                              true};
    const m3_music3_schema_summary *expected =
        &m3_music3_summaries[(size_t)id];
    m3_status status = m3_music3_schema_visit(
        id, m3_schema_summary_visitor, &generated, error);

    if (status == M3_STATUS_OK &&
        (generated.count != expected->tensor_count ||
         generated.elements != expected->element_count ||
         generated.bytes != expected->payload_bytes ||
         !generated.one_dtype || generated.dtype != expected->dtype)) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "internal Music3 schema aggregate mismatch");
    }
    return status;
}

m3_status m3_music3_schema_validate_inventory(
    m3_component_id id, const m3_safetensors_tensor *tensors,
    size_t tensor_count, m3_error *error)
{
    const m3_music3_schema_summary *expected;
    uint64_t elements = 0U;
    uint64_t bytes = 0U;
    size_t index;
    m3_status status;

    if (!m3_component_contains_weights(id) ||
        (tensor_count != 0U && tensors == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Music3 inventory argument is invalid");
    }
    expected = &m3_music3_summaries[(size_t)id];
    if (tensor_count != expected->tensor_count) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "component %s has %zu tensors, expected %zu",
                            m3_component_directory(id), tensor_count,
                            expected->tensor_count);
    }
    status = m3_schema_verify_generated(id, error);
    for (index = 0U; index < tensor_count && status == M3_STATUS_OK; ++index) {
        m3_schema_find_context find;
        size_t duplicate_index;

        if (tensors[index].name == NULL || tensors[index].name[0] == '\0') {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "Music3 inventory has an empty tensor name");
            break;
        }
        for (duplicate_index = 0U; duplicate_index < index;
             ++duplicate_index) {
            if (strcmp(tensors[index].name,
                       tensors[duplicate_index].name) == 0) {
                status = m3_error_set(
                    error, M3_STATUS_INVALID_FORMAT,
                    "Music3 inventory duplicates tensor '%s'",
                    tensors[index].name);
                break;
            }
        }
        if (status != M3_STATUS_OK) {
            break;
        }
        find.name = tensors[index].name;
        find.actual = &tensors[index].tensor;
        find.found = false;
        find.matches = false;
        status = m3_music3_schema_visit(id, m3_schema_find_visitor, &find,
                                        error);
        if (status == M3_STATUS_OK && (!find.found || !find.matches)) {
            status = m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "tensor '%s' does not match the official Music3 schema",
                tensors[index].name);
        }
        if (status == M3_STATUS_OK &&
            ((uint64_t)tensors[index].tensor.element_count >
                 UINT64_MAX - elements ||
             (uint64_t)tensors[index].tensor.byte_count >
                 UINT64_MAX - bytes)) {
            status = m3_error_set(error, M3_STATUS_OVERFLOW,
                                  "Music3 inventory aggregate overflows");
        }
        if (status == M3_STATUS_OK) {
            elements += (uint64_t)tensors[index].tensor.element_count;
            bytes += (uint64_t)tensors[index].tensor.byte_count;
        }
    }
    if (status == M3_STATUS_OK &&
        (elements != expected->element_count ||
         bytes != expected->payload_bytes)) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "component %s tensor aggregate is incorrect",
                              m3_component_directory(id));
    }
    if (status == M3_STATUS_OK) {
        m3_error_reset(error);
    }
    return status;
}

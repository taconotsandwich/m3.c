/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_music3_schema.h"
#include "m3_vocoder_internal.h"

#include "m3_test.h"

#include <stdbool.h>
#include <string.h>

typedef struct {
    m3_test_context *test;
    const m3_vocoder_plan *plan;
    size_t entry;
    size_t source;
    size_t visited;
} m3_vocoder_plan_visit;

static bool m3_vocoder_plan_metadata_equal(
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

static m3_status m3_vocoder_plan_visit_source(
    const char *name, const m3_tensor_metadata *tensor, void *context,
    m3_error *error)
{
    m3_vocoder_plan_visit *visit = context;
    const m3_vocoder_plan_entry *entry;
    size_t source_count;

    (void)error;
    M3_TEST_EXPECT(visit->test, visit->entry < visit->plan->entry_count,
                   "official schema does not exceed plan");
    if (visit->entry >= visit->plan->entry_count) {
        return M3_STATUS_INTERNAL;
    }
    entry = &visit->plan->entries[visit->entry];
    source_count = entry->kind == M3_VOCODER_MATERIAL_COPY ? 1U : 2U;
    M3_TEST_EXPECT(visit->test,
                   strcmp(name, entry->source_names[visit->source]) == 0,
                   "official source name preserves emission order");
    M3_TEST_EXPECT(
        visit->test,
        m3_vocoder_plan_metadata_equal(
            tensor, &entry->source_metadata[visit->source]),
        "official source tensor preserves shape and dtype");
    ++visit->visited;
    ++visit->source;
    if (visit->source == source_count) {
        visit->source = 0U;
        ++visit->entry;
    }
    return M3_STATUS_OK;
}

void test_vocoder_official_plan(m3_test_context *test)
{
    m3_vocoder_plan_config config;
    m3_vocoder_plan plan;
    m3_vocoder_plan_visit visit;
    m3_error error;
    size_t normalized = 0U;
    size_t alphas = 0U;
    size_t biases = 0U;
    size_t source_bytes = 0U;
    size_t owned_bytes = 0U;
    size_t maximum_row = 0U;
    size_t maximum_buffer = 0U;
    size_t entry;

    m3_error_reset(&error);
    m3_vocoder_plan_init(&plan);
    m3_vocoder_plan_official_config(&config);
    M3_TEST_EXPECT(test,
                   m3_vocoder_plan_build(&config, &plan, &error) ==
                       M3_STATUS_OK,
                   "build official vocoder materialization plan");
    M3_TEST_EXPECT(test,
                   plan.entry_count == M3_VOCODER_RUNTIME_WEIGHT_COUNT,
                   "official plan owns 91 tensors");
    M3_TEST_EXPECT(test,
                   plan.source_count == M3_VOCODER_SOURCE_WEIGHT_COUNT,
                   "official plan consumes 121 tensors");
    (void)memset(&visit, 0, sizeof(visit));
    visit.test = test;
    visit.plan = &plan;
    M3_TEST_EXPECT(test,
                   m3_music3_schema_visit(
                       M3_COMPONENT_VOCODER, m3_vocoder_plan_visit_source,
                       &visit, &error) == M3_STATUS_OK,
                   "visit official vocoder schema");
    M3_TEST_EXPECT(test, visit.visited == M3_VOCODER_SOURCE_WEIGHT_COUNT &&
                             visit.entry == plan.entry_count &&
                             visit.source == 0U,
                   "official plan covers every source exactly once");
    for (entry = 0U; entry < plan.entry_count; ++entry) {
        const m3_vocoder_plan_entry *spec = &plan.entries[entry];
        size_t source_count = spec->kind == M3_VOCODER_MATERIAL_COPY
                                  ? 1U
                                  : 2U;
        size_t source;
        size_t rows = (size_t)spec->output_metadata.shape[0];
        size_t row_bytes = spec->output_metadata.byte_count / rows;

        owned_bytes += spec->output_metadata.byte_count;
        if (spec->output_metadata.byte_count > maximum_buffer) {
            maximum_buffer = spec->output_metadata.byte_count;
        }
        if (row_bytes > maximum_row) {
            maximum_row = row_bytes;
        }
        if (spec->kind == M3_VOCODER_MATERIAL_WEIGHT_NORM) {
            ++normalized;
            M3_TEST_EXPECT(test,
                           strstr(spec->output_name, "weight_g") == NULL &&
                               strstr(spec->output_name, "weight_v") == NULL,
                           "normalized output has one current weight name");
        } else if (strstr(spec->output_name, ".alpha") != NULL) {
            ++alphas;
        } else if (strstr(spec->output_name, ".bias") != NULL ||
                   strcmp(spec->output_name, "dec_in_proj.bias") == 0) {
            ++biases;
        }
        for (source = 0U; source < source_count; ++source) {
            source_bytes += spec->source_metadata[source].byte_count;
        }
    }
    M3_TEST_EXPECT(test, normalized == 30U,
                   "thirty weight-norm pairs become thirty weights");
    M3_TEST_EXPECT(test, alphas == 29U,
                   "official plan copies twenty-nine Snake alphas");
    M3_TEST_EXPECT(test, biases == 31U,
                   "official plan copies decoder and convolution biases");
    M3_TEST_EXPECT(test, source_bytes == 216682888U,
                   "official source payload byte count is exact");
    M3_TEST_EXPECT(test, owned_bytes == 216630660U,
                   "owned payload excludes thirty gain tensors");
    M3_TEST_EXPECT(test, source_bytes - owned_bytes == 52228U,
                   "weight_g disappearance removes exactly 13057 floats");
    M3_TEST_EXPECT(test, maximum_row == M3_VOCODER_MAXIMUM_ROW_BYTES,
                   "official maximum streamed row is 49152 bytes");
    M3_TEST_EXPECT(test, maximum_buffer == 75497472U,
                   "official largest owned backend buffer is exact");
    m3_vocoder_plan_dispose(&plan);
}

void test_vocoder_plan_rejects_invalid_config(m3_test_context *test)
{
    m3_vocoder_plan_config config;
    m3_vocoder_plan plan;
    m3_error error;

    m3_error_reset(&error);
    m3_vocoder_plan_init(&plan);
    m3_vocoder_plan_official_config(&config);
    config.initial_channels = 1535U;
    M3_TEST_EXPECT(test,
                   m3_vocoder_plan_build(&config, &plan, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject odd channel halving");
    config.initial_channels = 1536U;
    config.strides[0] = 0U;
    M3_TEST_EXPECT(test,
                   m3_vocoder_plan_build(&config, &plan, NULL) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject zero stride with null error sink");
    m3_vocoder_plan_official_config(&config);
    config.decoder_input_channels = UINT64_MAX;
    config.decoder_output_channels = UINT64_MAX;
    M3_TEST_EXPECT(test,
                   m3_vocoder_plan_build(&config, &plan, &error) ==
                       M3_STATUS_OVERFLOW,
                   "reject overflowing vocoder tensor size");
    M3_TEST_EXPECT(test, plan.entries == NULL,
                   "invalid plan leaves output unchanged");
    m3_vocoder_plan_dispose(&plan);
}

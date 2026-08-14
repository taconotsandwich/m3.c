/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3.h"
#include "m3_model.h"

#include <stdint.h>
#include <string.h>

void m3_test_model_contract(m3_test_context *test)
{
    static const char *const directories[M3_COMPONENT_COUNT] = {
        "language_model", "rvq_depth_decoder", "condition_encoder",
        "transformer", "vocoder", "tokenizer", "scheduler"
    };
    m3_model_metadata metadata;
    m3_component_metadata *component;
    m3_error error;
    size_t index;

    m3_model_metadata_init(&metadata);
    M3_TEST_EXPECT(test, metadata.present_component_count == 0U,
                   "initial present component count");
    for (index = 0U; index < M3_COMPONENT_COUNT; ++index) {
        m3_component_id id = (m3_component_id)index;
        const m3_component_metadata *entry =
            m3_model_metadata_component(&metadata, id);

        M3_TEST_EXPECT(test, m3_component_id_is_valid(id),
                       "valid component identity");
        M3_TEST_EXPECT(test,
                       strcmp(m3_component_directory(id),
                              directories[index]) == 0,
                       "official component directory");
        M3_TEST_EXPECT(test,
                       m3_component_contains_weights(id) ==
                           (id <= M3_COMPONENT_VOCODER),
                       "component weight role");
        M3_TEST_EXPECT(test,
                       entry != NULL && entry->id == id && !entry->present,
                       "initialized component metadata");
    }
    M3_TEST_EXPECT(test,
                   !m3_component_id_is_valid((m3_component_id)-1),
                   "reject negative component identity");
    M3_TEST_EXPECT(test, m3_component_directory(M3_COMPONENT_COUNT) == NULL,
                   "reject sentinel component directory");
    M3_TEST_EXPECT(test,
                   m3_model_metadata_component(
                       NULL, M3_COMPONENT_LANGUAGE_MODEL) == NULL,
                   "reject null aggregate lookup");

    M3_TEST_EXPECT(test,
                   m3_model_metadata_add_file(&metadata,
                                              M3_COMPONENT_TOKENIZER,
                                              &error) == M3_STATUS_OK,
                   "record tokenizer file");
    M3_TEST_EXPECT(test,
                   m3_model_metadata_add_tensor(
                       &metadata, M3_COMPONENT_LANGUAGE_MODEL, 128U,
                       &error) == M3_STATUS_OK,
                   "record language model tensor");
    M3_TEST_EXPECT(test, metadata.present_component_count == 2U,
                   "aggregate present component count");
    M3_TEST_EXPECT(test,
                   metadata.file_count == 1U && metadata.tensor_count == 1U &&
                       metadata.tensor_bytes == 128U,
                   "aggregate model totals");
    M3_TEST_EXPECT(
        test,
        metadata.components[M3_COMPONENT_LANGUAGE_MODEL].tensor_bytes == 128U,
        "component tensor totals");
    M3_TEST_EXPECT(test,
                   m3_model_metadata_add_tensor(
                       &metadata, M3_COMPONENT_SCHEDULER, 4U,
                       &error) == M3_STATUS_INVALID_ARGUMENT,
                   "reject tensors in resource component");

    m3_model_metadata_init(&metadata);
    component = &metadata.components[M3_COMPONENT_TRANSFORMER];
    component->tensor_bytes = SIZE_MAX;
    metadata.tensor_bytes = SIZE_MAX;
    M3_TEST_EXPECT(test,
                   m3_model_metadata_add_tensor(
                       &metadata, M3_COMPONENT_TRANSFORMER, 1U,
                       &error) == M3_STATUS_OVERFLOW,
                   "reject aggregate tensor overflow");
    M3_TEST_EXPECT(test,
                   component->tensor_count == 0U && !component->present,
                   "overflow leaves component counts unchanged");

    m3_model_metadata_init(&metadata);
    component = &metadata.components[M3_COMPONENT_VOCODER];
    component->file_count = SIZE_MAX;
    M3_TEST_EXPECT(test,
                   m3_model_metadata_add_file(&metadata,
                                              M3_COMPONENT_VOCODER,
                                              &error) == M3_STATUS_OVERFLOW,
                   "reject component file overflow");
    M3_TEST_EXPECT(test, !component->present,
                   "file overflow leaves presence unchanged");

    M3_TEST_EXPECT(test,
                   m3_model_metadata_add_file(NULL, M3_COMPONENT_VOCODER,
                                              &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject null model metadata");
    M3_TEST_EXPECT(test,
                   m3_model_metadata_add_file(&metadata, M3_COMPONENT_COUNT,
                                              &error) ==
                       M3_STATUS_OUT_OF_RANGE,
                   "reject invalid model component");
}

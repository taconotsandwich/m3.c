/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3.h"
#include "m3_error.h"
#include "m3_model.h"
#include "m3_tensor.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(bool condition, const char *message)
{
    if (!condition) {
        (void)fprintf(stderr, "FAIL: %s\n", message);
        failures += 1;
    }
}

static void test_public_contract(void)
{
    const char *version = m3_version();

    check(strcmp(version, M3_VERSION_STRING) == 0, "version string");
    check(strcmp(m3_status_string(M3_STATUS_OK), "ok") == 0,
          "success status string");
    check(strcmp(m3_status_string(M3_STATUS_INVALID_FORMAT),
                 "invalid format") == 0,
          "format status string");
    check(strcmp(m3_status_string(M3_STATUS_CANCELLED), "cancelled") == 0,
          "cancelled status string");
    check(strcmp(m3_status_string((m3_status)99), "unknown status") == 0,
          "unknown status string");
}

static void test_error_contract(void)
{
    char long_message[512];
    m3_error error;
    m3_status status;

    m3_error_reset(&error);
    check(error.status == M3_STATUS_OK, "reset error status");
    check(strcmp(m3_error_message(&error), "") == 0,
          "reset error message");

    status = m3_error_set(&error, M3_STATUS_IO, "cannot read shard %u", 3U);
    check(status == M3_STATUS_IO, "formatted error return status");
    check(error.status == M3_STATUS_IO, "formatted error stored status");
    check(strcmp(error.message, "cannot read shard 3") == 0,
          "formatted error message");

    status = m3_error_set(&error, M3_STATUS_OVERFLOW, NULL);
    check(status == M3_STATUS_OVERFLOW, "default error return status");
    check(strcmp(error.message, "overflow") == 0,
          "default error message");

    (void)memset(long_message, 'x', sizeof(long_message));
    long_message[sizeof(long_message) - 1U] = '\0';
    (void)m3_error_set(&error, M3_STATUS_INTERNAL, "%s", long_message);
    check(strlen(error.message) == M3_ERROR_MESSAGE_CAPACITY - 1U,
          "long error truncation");
    check(error.message[M3_ERROR_MESSAGE_CAPACITY - 1U] == '\0',
          "long error termination");

    status = m3_error_set(NULL, M3_STATUS_IO, "ignored");
    check(status == M3_STATUS_IO, "null error storage return status");
    check(strcmp(m3_error_message(NULL), "") == 0,
          "null error message");

    (void)m3_error_set(&error, M3_STATUS_OK, "ignored");
    check(error.status == M3_STATUS_OK && error.message[0] == '\0',
          "success clears error");
}

static void test_tensor_contract(void)
{
    const uint64_t shape[] = {2U, 3U, 4U};
    const uint64_t zero_shape[] = {7U, 0U, UINT64_MAX};
    const uint64_t element_overflow_shape[] = {(uint64_t)SIZE_MAX, 2U};
    const uint64_t byte_overflow_shape[] = {
        (uint64_t)(SIZE_MAX / 4U) + 1U
    };
    uint64_t maximum_rank_shape[M3_TENSOR_MAX_RANK];
    m3_tensor_metadata metadata;
    m3_error error;
    size_t index;

    for (index = 0U; index < M3_TENSOR_MAX_RANK; ++index) {
        maximum_rank_shape[index] = 1U;
    }

    check(m3_dtype_size(M3_DTYPE_F32) == 4U, "F32 byte size");
    check(m3_dtype_size(M3_DTYPE_F16) == 2U, "F16 byte size");
    check(m3_dtype_size(M3_DTYPE_BF16) == 2U, "BF16 byte size");
    check(strcmp(m3_dtype_name(M3_DTYPE_BF16), "BF16") == 0,
          "BF16 dtype name");
    check(m3_dtype_size((m3_dtype)99) == 0U, "unknown dtype size");

    check(m3_tensor_metadata_init(&metadata, M3_DTYPE_F32, 3U, shape,
                                  &error) == M3_STATUS_OK,
          "initialize F32 tensor metadata");
    check(metadata.rank == 3U, "tensor rank");
    check(metadata.element_count == 24U, "tensor element count");
    check(metadata.byte_count == 96U, "tensor byte count");
    check(metadata.shape[0] == 2U && metadata.shape[2] == 4U,
          "tensor shape copy");
    check(error.status == M3_STATUS_OK, "tensor success clears error");

    check(m3_tensor_metadata_init(&metadata, M3_DTYPE_BF16, 0U, NULL,
                                  &error) == M3_STATUS_OK,
          "initialize scalar tensor metadata");
    check(metadata.element_count == 1U && metadata.byte_count == 2U,
          "scalar tensor counts");

    check(m3_tensor_metadata_init(&metadata, M3_DTYPE_F16,
                                  M3_TENSOR_MAX_RANK, maximum_rank_shape,
                                  &error) == M3_STATUS_OK,
          "initialize maximum-rank metadata");
    check(metadata.element_count == 1U, "maximum-rank element count");

    check(m3_tensor_metadata_init(&metadata, M3_DTYPE_F16, 3U, zero_shape,
                                  &error) == M3_STATUS_OK,
          "initialize empty tensor metadata");
    check(metadata.element_count == 0U && metadata.byte_count == 0U,
          "empty tensor counts");

    check(m3_tensor_metadata_init(&metadata, M3_DTYPE_F16,
                                  M3_TENSOR_MAX_RANK + 1U, shape,
                                  &error) == M3_STATUS_OUT_OF_RANGE,
          "reject excessive tensor rank");
    check(metadata.element_count == 0U, "clear rejected tensor output");
    check(m3_tensor_metadata_init(&metadata, M3_DTYPE_F16, 1U, NULL,
                                  &error) == M3_STATUS_INVALID_ARGUMENT,
          "reject missing tensor shape");
    check(m3_tensor_metadata_init(&metadata, (m3_dtype)99, 0U, NULL,
                                  &error) == M3_STATUS_UNSUPPORTED,
          "reject unknown tensor dtype");
    check(m3_tensor_metadata_init(&metadata, M3_DTYPE_F16, 2U,
                                  element_overflow_shape,
                                  &error) == M3_STATUS_OVERFLOW,
          "reject tensor element overflow");
    check(m3_tensor_metadata_init(&metadata, M3_DTYPE_F32, 1U,
                                  byte_overflow_shape,
                                  &error) == M3_STATUS_OVERFLOW,
          "reject tensor byte overflow");
    check(m3_tensor_metadata_init(NULL, M3_DTYPE_F32, 0U, NULL,
                                  &error) == M3_STATUS_INVALID_ARGUMENT,
          "reject null tensor metadata output");
}

static void test_model_contract(void)
{
    static const char *const directories[M3_COMPONENT_COUNT] = {
        "language_model",
        "rvq_depth_decoder",
        "condition_encoder",
        "transformer",
        "vocoder",
        "tokenizer",
        "scheduler"
    };
    m3_model_metadata metadata;
    m3_component_metadata *component;
    m3_error error;
    size_t index;

    m3_model_metadata_init(&metadata);
    check(metadata.present_component_count == 0U,
          "initial present component count");
    for (index = 0U; index < M3_COMPONENT_COUNT; ++index) {
        m3_component_id id = (m3_component_id)index;
        const m3_component_metadata *entry =
            m3_model_metadata_component(&metadata, id);

        check(m3_component_id_is_valid(id), "valid component identity");
        check(strcmp(m3_component_directory(id), directories[index]) == 0,
              "official component directory");
        check(m3_component_contains_weights(id) ==
                  (id <= M3_COMPONENT_VOCODER),
              "component weight role");
        check(entry != NULL && entry->id == id && !entry->present,
              "initialized component metadata");
    }
    check(!m3_component_id_is_valid((m3_component_id)-1),
          "reject negative component identity");
    check(m3_component_directory(M3_COMPONENT_COUNT) == NULL,
          "reject sentinel component directory");
    check(m3_model_metadata_component(NULL,
                                      M3_COMPONENT_LANGUAGE_MODEL) == NULL,
          "reject null aggregate lookup");

    check(m3_model_metadata_add_file(&metadata, M3_COMPONENT_TOKENIZER,
                                     &error) == M3_STATUS_OK,
          "record tokenizer file");
    check(m3_model_metadata_add_tensor(&metadata,
                                       M3_COMPONENT_LANGUAGE_MODEL, 128U,
                                       &error) == M3_STATUS_OK,
          "record language model tensor");
    check(metadata.present_component_count == 2U,
          "aggregate present component count");
    check(metadata.file_count == 1U && metadata.tensor_count == 1U &&
              metadata.tensor_bytes == 128U,
          "aggregate model totals");
    check(metadata.components[M3_COMPONENT_LANGUAGE_MODEL].tensor_bytes ==
              128U,
          "component tensor totals");
    check(m3_model_metadata_add_tensor(&metadata, M3_COMPONENT_SCHEDULER,
                                       4U, &error) ==
              M3_STATUS_INVALID_ARGUMENT,
          "reject tensors in resource component");

    m3_model_metadata_init(&metadata);
    component = &metadata.components[M3_COMPONENT_TRANSFORMER];
    component->tensor_bytes = SIZE_MAX;
    metadata.tensor_bytes = SIZE_MAX;
    check(m3_model_metadata_add_tensor(&metadata, M3_COMPONENT_TRANSFORMER,
                                       1U, &error) == M3_STATUS_OVERFLOW,
          "reject aggregate tensor overflow");
    check(component->tensor_count == 0U && !component->present,
          "overflow leaves component counts unchanged");

    m3_model_metadata_init(&metadata);
    component = &metadata.components[M3_COMPONENT_VOCODER];
    component->file_count = SIZE_MAX;
    check(m3_model_metadata_add_file(&metadata, M3_COMPONENT_VOCODER,
                                     &error) == M3_STATUS_OVERFLOW,
          "reject component file overflow");
    check(!component->present, "file overflow leaves presence unchanged");

    check(m3_model_metadata_add_file(NULL, M3_COMPONENT_VOCODER,
                                     &error) == M3_STATUS_INVALID_ARGUMENT,
          "reject null model metadata");
    check(m3_model_metadata_add_file(&metadata, M3_COMPONENT_COUNT,
                                     &error) == M3_STATUS_OUT_OF_RANGE,
          "reject invalid model component");
}

int main(void)
{
    test_public_contract();
    test_error_contract();
    test_tensor_contract();
    test_model_contract();

    if (failures != 0) {
        (void)fprintf(stderr, "%d contract test(s) failed\n", failures);
        return 1;
    }

    (void)printf("m3 %s: contract tests passed\n", m3_version());
    return 0;
}

/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_backend_internal.h"
#include "m3_music3_internal.h"
#include "m3_music3_schema.h"
#include "tokenizer_test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M3_OPEN_TEST_PATH_CAPACITY 512U

typedef enum {
    M3_OPEN_TEST_SUCCESS = 0,
    M3_OPEN_TEST_WRONG_TOTAL,
    M3_OPEN_TEST_MALFORMED_TABLE,
    M3_OPEN_TEST_INSPECT_FAILURE,
    M3_OPEN_TEST_TOKENIZER_FAILURE
} m3_open_test_failure;

typedef struct {
    const char *trusted_root;
    m3_open_test_failure failure;
    m3_component_id failure_component;
    size_t backend_creates;
    size_t backend_destroys;
    size_t model_inspects;
    size_t weight_inspects[M3_MUSIC3_WEIGHT_COMPONENT_COUNT];
    size_t tokenizer_loads;
    bool root_forwarded;
    uint64_t observed_payload[M3_MUSIC3_WEIGHT_COMPONENT_COUNT];
    m3_tokenizer_fixture tokenizer_data;
} m3_open_test_context;

typedef struct {
    m3_open_test_context *test;
} m3_open_backend_context;

static void m3_open_backend_destroy(void *pointer)
{
    m3_open_backend_context *context = pointer;

    ++context->test->backend_destroys;
    free(context);
}

static m3_status m3_open_backend_allocate(
    void *context, size_t byte_count, size_t alignment, void **handle,
    void **data, m3_error *error)
{
    size_t allocation_size = byte_count;
    size_t remainder;
    void *memory;

    (void)context;
    *handle = NULL;
    *data = NULL;
    if (byte_count == 0U) {
        return M3_STATUS_OK;
    }
    remainder = allocation_size & (alignment - 1U);
    if (remainder != 0U) {
        if (allocation_size > SIZE_MAX - (alignment - remainder)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "open test allocation overflows");
        }
        allocation_size += alignment - remainder;
    }
    memory = aligned_alloc(alignment, allocation_size);
    if (memory == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "open test allocation failed");
    }
    *handle = memory;
    *data = memory;
    return M3_STATUS_OK;
}

static void m3_open_backend_free(
    void *context, void *handle, void *data)
{
    (void)context;
    (void)data;
    free(handle);
}

static m3_status m3_open_backend_execute(
    void *context, const m3_command *commands, size_t command_count,
    m3_scratch_arena *scratch, m3_error *error)
{
    (void)context;
    (void)commands;
    (void)command_count;
    (void)scratch;
    return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                        "open test backend does not execute");
}

static m3_status m3_open_test_create_backend(
    void *pointer, m3_backend **backend, m3_error *error)
{
    static const m3_backend_vtable vtable = {
        m3_open_backend_destroy,
        m3_open_backend_allocate,
        m3_open_backend_free,
        m3_open_backend_execute};
    m3_open_test_context *test = pointer;
    m3_open_backend_context *context = calloc(1U, sizeof(*context));
    m3_backend_info info;
    m3_status status;

    if (context == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate open test backend");
    }
    context->test = test;
    (void)memset(&info, 0, sizeof(info));
    (void)memcpy(info.name, "Music3 open test Metal",
                 sizeof("Music3 open test Metal"));
    info.kind = M3_BACKEND_METAL;
    info.unified_memory = true;
    info.maximum_storage_bytes = UINT64_MAX;
    info.recommended_working_set_bytes = UINT64_MAX;
    status = m3_backend_create_internal(
        &vtable, context, &info, backend, error);
    if (status != M3_STATUS_OK) {
        free(context);
        return status;
    }
    ++test->backend_creates;
    return M3_STATUS_OK;
}

static bool m3_open_test_root(
    m3_open_test_context *test, const char *root)
{
    bool exact = root == test->trusted_root &&
                 strcmp(root, test->trusted_root) == 0;

    test->root_forwarded = test->root_forwarded && exact;
    return exact;
}

static m3_status m3_open_test_inspect_model(
    void *pointer, const char *root, m3_model_metadata *metadata,
    m3_error *error)
{
    m3_open_test_context *test = pointer;

    ++test->model_inspects;
    if (!m3_open_test_root(test, root)) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "trusted root was not forwarded");
    }
    m3_model_metadata_init(metadata);
    return M3_STATUS_OK;
}

static char *m3_open_test_copy(const char *text)
{
    size_t length = strlen(text);
    char *copy = malloc(length + 1U);

    if (copy != NULL) {
        (void)memcpy(copy, text, length + 1U);
    }
    return copy;
}

static m3_status m3_open_test_inspect_weights(
    void *pointer, const char *root, m3_component_id component,
    m3_weight_table *table, m3_error *error)
{
    m3_open_test_context *test = pointer;
    m3_music3_schema_summary expected;
    size_t index = (size_t)component;
    m3_status status;

    ++test->weight_inspects[index];
    if (!m3_open_test_root(test, root)) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "trusted root was not forwarded");
    }
    if (test->failure == M3_OPEN_TEST_INSPECT_FAILURE &&
        component == test->failure_component) {
        return m3_error_set(error, M3_STATUS_IO,
                            "injected component inspection failure");
    }
    status = m3_music3_schema_expected_summary(
        component, &expected, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    test->observed_payload[index] = expected.payload_bytes;
    if (test->failure == M3_OPEN_TEST_MALFORMED_TABLE &&
        component == test->failure_component) {
        table->aggregate_payload_bytes = expected.payload_bytes;
        return M3_STATUS_OK;
    }
    table->shards = calloc(1U, sizeof(*table->shards));
    if (table->shards == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate open test table");
    }
    table->shard_count = 1U;
    table->shards[0].path = m3_open_test_copy("trusted-sparse-shard");
    if (table->shards[0].path == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot name open test shard");
    }
    table->shards[0].payload_bytes = expected.payload_bytes;
    if (test->failure == M3_OPEN_TEST_WRONG_TOTAL &&
        component == test->failure_component) {
        --table->shards[0].payload_bytes;
    }
    table->aggregate_payload_bytes = table->shards[0].payload_bytes;
    return M3_STATUS_OK;
}

static m3_status m3_open_test_load_tokenizer(
    void *pointer, m3_tokenizer *tokenizer, const char *path,
    m3_error *error)
{
    m3_open_test_context *test = pointer;
    char expected[M3_OPEN_TEST_PATH_CAPACITY];
    int length;

    ++test->tokenizer_loads;
    length = snprintf(
        expected, sizeof(expected), "%s/tokenizer/tokenizer.json",
        test->trusted_root);
    if (length <= 0 || (size_t)length >= sizeof(expected) ||
        strcmp(path, expected) != 0) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "exact tokenizer path was not used");
    }
    if (test->failure == M3_OPEN_TEST_TOKENIZER_FAILURE) {
        return m3_error_set(error, M3_STATUS_IO,
                            "injected tokenizer load failure");
    }
    return m3_test_tokenizer_fixture_load(
               tokenizer, &test->tokenizer_data,
               M3_TOKENIZER_FIXTURE_VALID, error)
               ? M3_STATUS_OK
               : (error == NULL ? M3_STATUS_INTERNAL : error->status);
}

static void m3_open_test_context_init(
    m3_open_test_context *context, const char *root,
    m3_open_test_failure failure, m3_component_id component)
{
    (void)memset(context, 0, sizeof(*context));
    context->trusted_root = root;
    context->failure = failure;
    context->failure_component = component;
    context->root_forwarded = true;
}

static void m3_open_test_operations(
    m3_open_test_context *context, m3_music3_operations *operations)
{
    *operations = *m3_music3_production_operations();
    operations->context = context;
    operations->create_backend = m3_open_test_create_backend;
    operations->inspect_model = m3_open_test_inspect_model;
    operations->inspect_weights = m3_open_test_inspect_weights;
    operations->load_tokenizer = m3_open_test_load_tokenizer;
}

static bool m3_open_test_exact_payloads(
    const m3_open_test_context *context, const m3_music3_engine *engine)
{
    size_t index;

    for (index = 0U; index < M3_MUSIC3_WEIGHT_COMPONENT_COUNT; ++index) {
        m3_music3_schema_summary expected;
        m3_error error;

        if (m3_music3_schema_expected_summary(
                (m3_component_id)index, &expected, &error) !=
                M3_STATUS_OK ||
            context->observed_payload[index] != expected.payload_bytes ||
            engine->component_payload_bytes[index] !=
                expected.payload_bytes) {
            return false;
        }
    }
    return true;
}

void m3_test_music3_open_atomicity(m3_test_context *test)
{
    static const char root[] = "/caller/trusted/music3";
    static const struct {
        m3_open_test_failure failure;
        m3_component_id component;
        m3_status status;
        size_t tokenizer_loads;
    } failures[] = {
        {M3_OPEN_TEST_WRONG_TOTAL, M3_COMPONENT_LANGUAGE_MODEL,
         M3_STATUS_INVALID_FORMAT, 0U},
        {M3_OPEN_TEST_WRONG_TOTAL, M3_COMPONENT_RVQ_DEPTH_DECODER,
         M3_STATUS_INVALID_FORMAT, 0U},
        {M3_OPEN_TEST_WRONG_TOTAL, M3_COMPONENT_CONDITION_ENCODER,
         M3_STATUS_INVALID_FORMAT, 0U},
        {M3_OPEN_TEST_WRONG_TOTAL, M3_COMPONENT_TRANSFORMER,
         M3_STATUS_INVALID_FORMAT, 0U},
        {M3_OPEN_TEST_WRONG_TOTAL, M3_COMPONENT_VOCODER,
         M3_STATUS_INVALID_FORMAT, 0U},
        {M3_OPEN_TEST_MALFORMED_TABLE, M3_COMPONENT_TRANSFORMER,
         M3_STATUS_INVALID_FORMAT, 0U},
        {M3_OPEN_TEST_INSPECT_FAILURE, M3_COMPONENT_VOCODER,
         M3_STATUS_IO, 0U},
        {M3_OPEN_TEST_TOKENIZER_FAILURE, M3_COMPONENT_VOCODER,
         M3_STATUS_IO, 1U},
    };
    m3_open_test_context context;
    m3_music3_operations operations;
    m3_music3_engine *engine = NULL;
    m3_error error;
    m3_status status;
    size_t index;

    m3_open_test_context_init(
        &context, root, M3_OPEN_TEST_SUCCESS,
        M3_COMPONENT_LANGUAGE_MODEL);
    m3_open_test_operations(&context, &operations);
    status = m3_music3_engine_open_core(
        &engine, root, &operations, &error);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_OK && engine != NULL &&
            context.root_forwarded && context.backend_creates == 1U &&
            context.backend_destroys == 0U &&
            context.model_inspects == 1U && context.tokenizer_loads == 1U &&
            m3_open_test_exact_payloads(&context, engine),
        "private open publishes one backend with exact five official totals");
    m3_music3_engine_free(engine);
    engine = NULL;
    M3_TEST_EXPECT(test, context.backend_destroys == 1U,
                   "successful open releases its single backend once");
    m3_tokenizer_fixture_dispose(&context.tokenizer_data);

    for (index = 0U; index < sizeof(failures) / sizeof(failures[0]);
         ++index) {
        m3_open_test_context_init(
            &context, root, failures[index].failure,
            failures[index].component);
        m3_open_test_operations(&context, &operations);
        status = m3_music3_engine_open_core(
            &engine, root, &operations, &error);
        M3_TEST_EXPECT(
            test,
            status == failures[index].status && engine == NULL &&
                context.root_forwarded && context.backend_creates == 1U &&
                context.backend_destroys == 1U &&
                context.model_inspects == 1U &&
                context.tokenizer_loads == failures[index].tokenizer_loads,
            "open failure publishes nothing and cleans every partial owner");
        m3_tokenizer_fixture_dispose(&context.tokenizer_data);
    }
}

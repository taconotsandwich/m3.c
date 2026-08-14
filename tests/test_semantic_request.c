/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "semantic_runtime_test.h"

#include "m3_backend.h"
#include "m3_tokenizer.h"

#include <stdint.h>
#include <string.h>

static bool m3_semantic_request_unchanged(
    const m3_semantic_output *output, const m3_storage *storage,
    const m3_rng *rng, const m3_rng *initial)
{
    uint16_t value = 0U;

    return output->storage == storage &&
           m3_semantic_test_output_bits(output, &value, 1U) &&
           value == UINT16_C(0x7171) &&
           memcmp(rng, initial, sizeof(*rng)) == 0;
}

void m3_test_semantic_request_contract(m3_test_context *test)
{
    const int32_t valid_prompt[] = {
        M3_TOKEN_IM_START, M3_TOKEN_CAPTION_START, M3_TOKEN_CAPTION_END,
        M3_TOKEN_IM_END, M3_TOKEN_AUDIO_START,
        M3_TOKEN_IM_START, M3_TOKEN_AUDIO_CFG, M3_TOKEN_AUDIO_CFG,
        M3_TOKEN_IM_END, M3_TOKEN_AUDIO_START
    };
    const int32_t swapped_prompt[] = {
        M3_TOKEN_IM_START, M3_TOKEN_AUDIO_CFG, M3_TOKEN_AUDIO_CFG,
        M3_TOKEN_IM_END, M3_TOKEN_AUDIO_START,
        M3_TOKEN_IM_START, M3_TOKEN_CAPTION_START, M3_TOKEN_CAPTION_END,
        M3_TOKEN_IM_END, M3_TOKEN_AUDIO_START
    };
    m3_semantic_test_fixture fixture;
    m3_semantic_operations incomplete;
    m3_semantic_output output;
    m3_storage *original_storage;
    m3_backend *other_backend = NULL;
    m3_backend *backend = NULL;
    m3_weight_stage language_stage = {0};
    m3_weight_stage rvq_stage = {0};
    m3_tensor_view strided_prompt;
    int32_t invalid_id = (int32_t)M3_TOKENIZER_ID_COUNT;
    int32_t invalid_boundary = (int32_t)M3_TOKEN_AUDIO_START;
    m3_rng initial;
    m3_rng invalid_rng;
    m3_rng rng;
    m3_error error;
    bool ready;

    m3_error_reset(&error);
    ready = m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
            m3_backend_create_host(&other_backend, &error) ==
                M3_STATUS_OK &&
            m3_semantic_test_fixture_init(&fixture, backend) &&
            m3_rng_seed(&rng, UINT64_C(0x99887766),
                        UINT64_C(0x44332211), &error) == M3_STATUS_OK &&
            m3_semantic_test_seed_output(
                backend, UINT16_C(0x7171), &output);
    M3_TEST_EXPECT(test, ready, "create semantic request fixture");
    if (!ready) {
        m3_backend_free(other_backend);
        m3_backend_free(backend);
        return;
    }
    original_storage = output.storage;
    initial = rng;
    M3_TEST_EXPECT(
        test,
        m3_storage_write(
            fixture.prompt_storage, 0U, swapped_prompt,
            sizeof(swapped_prompt), &error) == M3_STATUS_OK &&
            m3_semantic_generate_core(
                &fixture.operations, &fixture.prompt, 1U, &rng, NULL,
                NULL, &output, &error) == M3_STATUS_INVALID_FORMAT &&
            fixture.start_calls == 0U &&
            m3_semantic_request_unchanged(
                &output, original_storage, &rng, &initial),
        "require conditional row zero and CFG row one before startup");
    M3_TEST_EXPECT(
        test,
        m3_storage_write(
            fixture.prompt_storage, 0U, valid_prompt,
            sizeof(valid_prompt), &error) == M3_STATUS_OK,
        "restore valid semantic prompt fixture");
    M3_TEST_EXPECT(
        test,
        m3_storage_write(
            fixture.prompt_storage, sizeof(int32_t), &invalid_id,
            sizeof(invalid_id), &error) == M3_STATUS_OK &&
            m3_semantic_generate_core(
                &fixture.operations, &fixture.prompt, 1U, &rng, NULL,
                NULL, &output, NULL) == M3_STATUS_OUT_OF_RANGE &&
            m3_storage_write(
                fixture.prompt_storage, 0U, &invalid_boundary,
                sizeof(invalid_boundary), &error) == M3_STATUS_OK &&
            m3_semantic_generate_core(
                &fixture.operations, &fixture.prompt, 1U, &rng, NULL,
                NULL, &output, NULL) == M3_STATUS_INVALID_FORMAT &&
            m3_storage_write(
                fixture.prompt_storage, 0U, valid_prompt,
                sizeof(valid_prompt), &error) == M3_STATUS_OK &&
            m3_semantic_request_unchanged(
                &output, original_storage, &rng, &initial),
        "validate conditional tokenizer IDs and exact prompt boundaries");
    strided_prompt = fixture.prompt;
    strided_prompt.byte_strides[0] += sizeof(int32_t);
    M3_TEST_EXPECT(
        test,
        m3_semantic_generate_core(
            &fixture.operations, &strided_prompt, 1U, &rng, NULL,
            NULL, &output, NULL) == M3_STATUS_INVALID_ARGUMENT &&
            m3_semantic_generate_core(
                &fixture.operations, &fixture.prompt, 0U, &rng, NULL,
                NULL, &output, NULL) == M3_STATUS_OUT_OF_RANGE &&
            m3_semantic_generate_core(
                &fixture.operations, &fixture.prompt, 9001U, &rng,
                NULL, NULL, &output, NULL) == M3_STATUS_OUT_OF_RANGE &&
            m3_semantic_request_unchanged(
                &output, original_storage, &rng, &initial),
        "reject noncontiguous prompts and every frame-limit boundary atomically");
    invalid_rng = rng;
    invalid_rng.increment &= ~UINT64_C(1);
    incomplete = fixture.operations;
    incomplete.advance = NULL;
    M3_TEST_EXPECT(
        test,
        m3_semantic_generate_core(
            &fixture.operations, &fixture.prompt, 1U, &invalid_rng,
            NULL, NULL, &output, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            m3_semantic_generate_core(
                &incomplete, &fixture.prompt, 1U, &rng, NULL, NULL,
                &output, NULL) == M3_STATUS_INVALID_ARGUMENT &&
            m3_semantic_generate_core(
                &fixture.operations, &fixture.prompt, 1U, &rng, NULL,
                NULL, NULL, NULL) == M3_STATUS_INVALID_ARGUMENT &&
            fixture.start_calls == 0U,
        "reject invalid RNG, output, and operation ownership before startup");
    language_stage.backend = backend;
    rvq_stage.backend = other_backend;
    M3_TEST_EXPECT(
        test,
        m3_semantic_generate(
            &language_stage, &rvq_stage, &fixture.prompt, 1U, &rng,
            NULL, NULL, &output, &error) ==
                M3_STATUS_INVALID_ARGUMENT &&
            m3_semantic_request_unchanged(
                &output, original_storage, &rng, &initial),
        "reject split production backends before component binding");
    m3_semantic_output_dispose(&output);
    m3_semantic_test_fixture_dispose(&fixture);
    m3_backend_free(other_backend);
    m3_backend_free(backend);
}

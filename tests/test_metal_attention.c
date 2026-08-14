/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_backend_metal_helpers.h"
#include "m3_op_test.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    m3_tensor_view query;
    m3_tensor_view key;
    m3_tensor_view value;
    m3_tensor_view mask;
    m3_tensor_view output;
} m3_test_attention_views;

static bool m3_test_attention_metal_fixture(
    m3_test_context *test, m3_op_test_fixture *fixture)
{
    m3_error error;
    m3_status status;

    (void)memset(fixture, 0, sizeof(*fixture));
    status = m3_backend_create_metal(&fixture->backend, &error);
    if (status == M3_STATUS_UNSUPPORTED) {
        M3_TEST_SKIP(test, m3_error_message(&error));
        return false;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create Metal attention fixture");
    return status == M3_STATUS_OK;
}

static void m3_test_attention_command(
    m3_command *command, m3_test_attention_views *views, bool with_mask,
    bool causal, int64_t causal_offset)
{
    (void)memset(command, 0, sizeof(*command));
    command->kind = M3_OP_ATTENTION;
    command->descriptor.attention.query = &views->query;
    command->descriptor.attention.key = &views->key;
    command->descriptor.attention.value = &views->value;
    command->descriptor.attention.mask = with_mask ? &views->mask : NULL;
    command->descriptor.attention.output = &views->output;
    command->descriptor.attention.scale = 0.5F;
    command->descriptor.attention.causal_offset = causal_offset;
    command->descriptor.attention.causal = causal;
}

static bool m3_test_attention_strided_f32(
    m3_op_test_fixture *fixture, m3_tensor_view *view,
    const uint64_t shape[4], const float *values)
{
    m3_tensor_metadata metadata;
    m3_storage *storage = NULL;
    size_t strides[4];
    size_t byte_count;
    size_t flat;
    uint8_t axis;
    m3_error error;
    uint8_t *data;

    m3_tensor_view_init(view);
    if (m3_tensor_metadata_init(&metadata, M3_DTYPE_F32, 4U, shape,
                                &error) != M3_STATUS_OK ||
        metadata.element_count == 0U) {
        return false;
    }
    strides[3] = 2U * sizeof(float);
    for (axis = 3U; axis > 0U; --axis) {
        strides[axis - 1U] =
            strides[axis] * (size_t)shape[axis] + sizeof(float);
    }
    byte_count = sizeof(float) + sizeof(float);
    for (axis = 0U; axis < 4U; ++axis) {
        byte_count += ((size_t)shape[axis] - 1U) * strides[axis];
    }
    if (!m3_op_test_storage(fixture, byte_count, &storage)) {
        return false;
    }
    data = m3_storage_data(storage);
    (void)memset(data, 0xa5, byte_count);
    for (flat = 0U; flat < metadata.element_count; ++flat) {
        size_t remainder = flat;
        size_t byte_offset = sizeof(float);

        for (axis = 4U; axis > 0U; --axis) {
            size_t dimension = (size_t)shape[axis - 1U];
            size_t coordinate = remainder % dimension;

            remainder /= dimension;
            byte_offset += coordinate * strides[axis - 1U];
        }
        (void)memcpy(data + byte_offset, &values[flat], sizeof(float));
    }
    return m3_tensor_view_strided(view, storage, M3_DTYPE_F32, 4U, shape,
                                  strides, sizeof(float), &error) ==
           M3_STATUS_OK;
}

static bool m3_test_attention_outputs_close(
    m3_tensor_view *left, m3_tensor_view *right, float tolerance)
{
    const float *left_values = m3_op_test_f32(left);
    const float *right_values = m3_op_test_f32(right);
    size_t count = left->metadata.element_count;
    size_t index;

    for (index = 0U; index < count; ++index) {
        if (fabsf(left_values[index] - right_values[index]) > tolerance) {
            return false;
        }
    }
    return true;
}

static uint32_t m3_test_attention_f32_bits(float value)
{
    uint32_t bits;

    (void)memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static bool m3_test_attention_precision_oracle(m3_test_context *test)
{
    const uint64_t shape[] = {2U};
    const float scores[] = {0.0F, 1.0F};
    const uint16_t zeros[] = {0U, 0U};
    m3_op_test_fixture fixture;
    m3_tensor_view score_view;
    m3_tensor_view bf16_probabilities;
    m3_tensor_view f16_probabilities;
    m3_command command;
    m3_error error;
    bool ok;

    if (!m3_op_test_fixture_init(&fixture)) {
        M3_TEST_EXPECT(test, false, "create host attention oracle");
        return false;
    }
    ok = m3_op_test_tensor(&fixture, &score_view, M3_DTYPE_F32, 1U,
                           shape, scores) &&
         m3_op_test_tensor(&fixture, &bf16_probabilities, M3_DTYPE_BF16,
                           1U, shape, zeros) &&
         m3_op_test_tensor(&fixture, &f16_probabilities, M3_DTYPE_F16, 1U,
                           shape, zeros);
    M3_TEST_EXPECT(test, ok, "create exact attention probability oracle");
    if (!ok) {
        m3_op_test_fixture_dispose(&fixture);
        return false;
    }
    command.kind = M3_OP_SOFTMAX;
    command.descriptor.softmax.input = &score_view;
    command.descriptor.softmax.output = &bf16_probabilities;
    ok = m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
         M3_STATUS_OK;
    command.descriptor.softmax.output = &f16_probabilities;
    ok = ok && m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                   M3_STATUS_OK;
    M3_TEST_EXPECT(
        test,
        ok && m3_op_test_u16(&bf16_probabilities)[0] == 0x3e8aU &&
            m3_op_test_u16(&bf16_probabilities)[1] == 0x3f3bU &&
            m3_op_test_u16(&f16_probabilities)[0] == 0x344eU &&
            m3_op_test_u16(&f16_probabilities)[1] == 0x39d9U,
        "lock exact BF16 and F16 attention probability bits");
    m3_op_test_fixture_dispose(&fixture);
    return ok;
}

void m3_test_metal_attention_precision(m3_test_context *test)
{
    const uint64_t query_shape[] = {1U, 1U, 1U, 1U};
    const uint64_t key_shape[] = {1U, 1U, 2U, 1U};
    const uint16_t bf16_query[] = {0x3f80U};
    const uint16_t bf16_key[] = {0x0000U, 0x3f80U};
    const uint16_t bf16_value[] = {0xc280U, 0xc180U};
    const uint16_t f16_query[] = {0x3c00U};
    const uint16_t f16_key[] = {0x0000U, 0x3c00U};
    const uint16_t f16_value[] = {0xd400U, 0xd240U};
    const uint16_t zero16[] = {0U};
    const float f32_query[] = {1.0F};
    const float f32_key[] = {0.0F, 1.0F};
    const float f32_value[] = {-64.0F, -16.0F};
    const float zero32[] = {0.0F};
    m3_op_test_fixture fixture;
    m3_test_attention_views bf16;
    m3_test_attention_views f16;
    m3_test_attention_views f32;
    m3_command command;
    m3_error error;
    bool ok;

    if (!m3_test_attention_precision_oracle(test) ||
        !m3_test_attention_metal_fixture(test, &fixture)) {
        return;
    }
    ok = m3_op_test_tensor(&fixture, &bf16.query, M3_DTYPE_BF16, 4U,
                           query_shape, bf16_query) &&
         m3_op_test_tensor(&fixture, &bf16.key, M3_DTYPE_BF16, 4U,
                           key_shape, bf16_key) &&
         m3_op_test_tensor(&fixture, &bf16.value, M3_DTYPE_BF16, 4U,
                           key_shape, bf16_value) &&
         m3_op_test_tensor(&fixture, &bf16.output, M3_DTYPE_BF16, 4U,
                           query_shape, zero16) &&
         m3_op_test_tensor(&fixture, &f16.query, M3_DTYPE_F16, 4U,
                           query_shape, f16_query) &&
         m3_op_test_tensor(&fixture, &f16.key, M3_DTYPE_F16, 4U, key_shape,
                           f16_key) &&
         m3_op_test_tensor(&fixture, &f16.value, M3_DTYPE_F16, 4U,
                           key_shape, f16_value) &&
         m3_op_test_tensor(&fixture, &f16.output, M3_DTYPE_F16, 4U,
                           query_shape, zero16) &&
         m3_op_test_tensor(&fixture, &f32.query, M3_DTYPE_F32, 4U,
                           query_shape, f32_query) &&
         m3_op_test_tensor(&fixture, &f32.key, M3_DTYPE_F32, 4U, key_shape,
                           f32_key) &&
         m3_op_test_tensor(&fixture, &f32.value, M3_DTYPE_F32, 4U,
                           key_shape, f32_value) &&
         m3_op_test_tensor(&fixture, &f32.output, M3_DTYPE_F32, 4U,
                           query_shape, zero32);
    M3_TEST_EXPECT(test, ok, "create exact Metal attention tensors");
    if (!ok) {
        m3_op_test_fixture_dispose(&fixture);
        return;
    }
    m3_test_attention_command(&command, &bf16, false, false, 0);
    command.descriptor.attention.scale = 1.0F;
    ok = m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
         M3_STATUS_OK;
    m3_test_attention_command(&command, &f16, false, false, 0);
    command.descriptor.attention.scale = 1.0F;
    ok = ok && m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                   M3_STATUS_OK;
    m3_test_attention_command(&command, &f32, false, false, 0);
    command.descriptor.attention.scale = 1.0F;
    ok = ok && m3_op_test_execute(&fixture, &command, 1U, NULL, &error) ==
                   M3_STATUS_OK;
    M3_TEST_EXPECT(
        test,
        ok && m3_op_test_u16(&bf16.output)[0] == 0xc1e8U &&
            m3_op_test_u16(&f16.output)[0] == 0xd2b9U &&
            m3_test_attention_f32_bits(m3_op_test_f32(&f32.output)[0]) ==
                UINT32_C(0xc1e74605),
        "Metal attention preserves exact BF16, F16, and F32 eager bits");
    m3_op_test_fixture_dispose(&fixture);
}

static bool m3_test_attention_case_init(
    m3_op_test_fixture *fixture, m3_test_attention_views *views,
    const float *query, const float *key, const float *value,
    const float *mask, const float *zeros)
{
    const uint64_t query_shape[] = {2U, 4U, 2U, 2U};
    const uint64_t key_shape[] = {2U, 2U, 3U, 2U};
    const uint64_t mask_shape[] = {1U, 1U, 2U, 3U};

    return m3_test_attention_strided_f32(
               fixture, &views->query, query_shape, query) &&
           m3_test_attention_strided_f32(
               fixture, &views->key, key_shape, key) &&
           m3_test_attention_strided_f32(
               fixture, &views->value, key_shape, value) &&
           m3_test_attention_strided_f32(
               fixture, &views->mask, mask_shape, mask) &&
           m3_op_test_tensor(fixture, &views->output, M3_DTYPE_F32, 4U,
                             query_shape, zeros);
}

static bool m3_test_attention_compare_mode(
    m3_op_test_fixture *host, m3_test_attention_views *host_views,
    m3_op_test_fixture *metal, m3_test_attention_views *metal_views,
    bool with_mask, bool causal, int64_t offset)
{
    m3_command host_command;
    m3_command metal_command;
    m3_error error;

    m3_test_attention_command(&host_command, host_views, with_mask, causal,
                              offset);
    m3_test_attention_command(&metal_command, metal_views, with_mask, causal,
                              offset);
    return m3_op_test_execute(host, &host_command, 1U, NULL, &error) ==
               M3_STATUS_OK &&
           m3_op_test_execute(metal, &metal_command, 1U, NULL, &error) ==
               M3_STATUS_OK &&
           m3_test_attention_outputs_close(
               &host_views->output, &metal_views->output, 2.0e-5F);
}

static bool m3_test_attention_special_rows(
    m3_op_test_fixture *fixture, m3_error *error)
{
    const uint64_t query_shape[] = {1U, 1U, 2U, 1U};
    const uint64_t key_shape[] = {1U, 1U, 2U, 1U};
    const uint64_t mask_shape[] = {1U, 1U, 2U, 2U};
    const float query[] = {1.0F, FLT_MAX};
    const float key[] = {2.0F, 2.0F};
    const float value[] = {2.0F, 4.0F};
    const float mask[] = {-INFINITY, -INFINITY, 0.0F, 0.0F};
    const float zeros[] = {-9.0F, -9.0F};
    m3_test_attention_views views;
    m3_command command;

    if (!m3_op_test_tensor(fixture, &views.query, M3_DTYPE_F32, 4U,
                           query_shape, query) ||
        !m3_op_test_tensor(fixture, &views.key, M3_DTYPE_F32, 4U,
                           key_shape, key) ||
        !m3_op_test_tensor(fixture, &views.value, M3_DTYPE_F32, 4U,
                           key_shape, value) ||
        !m3_op_test_tensor(fixture, &views.mask, M3_DTYPE_F32, 4U,
                           mask_shape, mask) ||
        !m3_op_test_tensor(fixture, &views.output, M3_DTYPE_F32, 4U,
                           query_shape, zeros)) {
        return false;
    }
    m3_test_attention_command(&command, &views, true, false, 0);
    command.descriptor.attention.scale = 1.0F;
    return m3_op_test_execute(fixture, &command, 1U, NULL, error) ==
               M3_STATUS_OK &&
           m3_op_test_f32(&views.output)[0] == 0.0F &&
           m3_op_test_f32(&views.output)[1] == 3.0F;
}

static bool m3_test_attention_zero_shapes(
    m3_op_test_fixture *fixture, m3_error *error)
{
    const uint64_t zero_batch[] = {0U, 1U, 1U, 1U};
    const uint64_t zero_query[] = {1U, 1U, 0U, 1U};
    const uint64_t one[] = {1U, 1U, 1U, 1U};
    const float value[] = {1.0F};
    m3_test_attention_views batch_views;
    m3_test_attention_views query_views;
    m3_command command;

    if (!m3_op_test_tensor(fixture, &batch_views.query, M3_DTYPE_F32, 4U,
                           zero_batch, NULL) ||
        !m3_op_test_tensor(fixture, &batch_views.key, M3_DTYPE_F32, 4U,
                           zero_batch, NULL) ||
        !m3_op_test_tensor(fixture, &batch_views.value, M3_DTYPE_F32, 4U,
                           zero_batch, NULL) ||
        !m3_op_test_tensor(fixture, &batch_views.output, M3_DTYPE_F32, 4U,
                           zero_batch, NULL) ||
        !m3_op_test_tensor(fixture, &query_views.query, M3_DTYPE_F32, 4U,
                           zero_query, NULL) ||
        !m3_op_test_tensor(fixture, &query_views.key, M3_DTYPE_F32, 4U, one,
                           value) ||
        !m3_op_test_tensor(fixture, &query_views.value, M3_DTYPE_F32, 4U,
                           one, value) ||
        !m3_op_test_tensor(fixture, &query_views.output, M3_DTYPE_F32, 4U,
                           zero_query, NULL)) {
        return false;
    }
    m3_test_attention_command(&command, &batch_views, false, false, 0);
    if (m3_op_test_execute(fixture, &command, 1U, NULL, error) !=
        M3_STATUS_OK) {
        return false;
    }
    m3_test_attention_command(&command, &query_views, false, true,
                              INT64_MIN);
    return m3_op_test_execute(fixture, &command, 1U, NULL, error) ==
           M3_STATUS_OK;
}

void m3_test_metal_attention_oracle(m3_test_context *test)
{
    float query[32];
    float key[24];
    float value[24];
    const float mask[] = {0.0F, -INFINITY, -0.25F,
                          -0.5F, 0.0F, -INFINITY};
    float zeros[32] = {0.0F};
    m3_op_test_fixture host;
    m3_op_test_fixture metal;
    m3_test_attention_views host_views;
    m3_test_attention_views metal_views;
    m3_backend_allocation_stats stats;
    m3_error error;
    size_t index;
    bool ok;

    if (!m3_test_attention_metal_fixture(test, &metal)) {
        return;
    }
    if (!m3_op_test_fixture_init(&host)) {
        M3_TEST_EXPECT(test, false, "create host attention comparison");
        m3_op_test_fixture_dispose(&metal);
        return;
    }
    for (index = 0U; index < 32U; ++index) {
        query[index] = (float)((int)(index % 7U) - 3) * 0.25F;
    }
    for (index = 0U; index < 24U; ++index) {
        key[index] = (float)((int)(index % 5U) - 2) * 0.2F;
        value[index] = (float)((int)(index % 9U) - 4) * 0.125F;
    }
    ok = m3_test_attention_case_init(
             &host, &host_views, query, key, value, mask, zeros) &&
         m3_test_attention_case_init(
             &metal, &metal_views, query, key, value, mask, zeros);
    M3_TEST_EXPECT(test, ok,
                   "create strided broadcast GQA attention tensors");
    if (!ok) {
        m3_op_test_fixture_dispose(&host);
        m3_op_test_fixture_dispose(&metal);
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_test_attention_compare_mode(
            &host, &host_views, &metal, &metal_views, true, true, 1),
        "Metal matches host causal GQA with strided broadcast mask");
    M3_TEST_EXPECT(
        test,
        m3_test_attention_compare_mode(
            &host, &host_views, &metal, &metal_views, false, false, 0),
        "Metal matches host noncausal strided GQA");
    M3_TEST_EXPECT(
        test,
        m3_test_attention_compare_mode(
            &host, &host_views, &metal, &metal_views, false, true,
            INT64_MAX),
        "Metal causal offset handles positive overflow without wrap");
    M3_TEST_EXPECT(
        test,
        m3_test_attention_compare_mode(
            &host, &host_views, &metal, &metal_views, false, true,
            INT64_MIN),
        "Metal causal offset handles negative magnitude without wrap");
    M3_TEST_EXPECT(test, m3_test_attention_special_rows(&metal, &error),
                   "Metal maps all-negative-infinity rows to zero and "
                   "splits positive-infinity maxima");
    M3_TEST_EXPECT(test, m3_test_attention_zero_shapes(&metal, &error),
                   "Metal accepts zero batch and zero query dispatches");
    M3_TEST_EXPECT(
        test,
        m3_backend_get_allocation_stats(metal.backend, &stats, &error) ==
                M3_STATUS_OK &&
            stats.live_storage_count == metal.storage_count &&
            stats.live_allocated_bytes != 0U,
        "Metal attention storage remains live and accounted");
    m3_op_test_fixture_dispose(&host);
    m3_op_test_fixture_dispose(&metal);
}

static bool m3_test_attention_write_scalar(m3_tensor_view *view,
                                            float value)
{
    m3_error error;

    return m3_storage_write(view->storage, view->byte_offset, &value,
                            sizeof(value), &error) == M3_STATUS_OK;
}

void m3_test_metal_attention_preflight(m3_test_context *test)
{
    const uint64_t shape[] = {1U, 1U, 1U, 1U};
    const float one[] = {1.0F};
    const float two[] = {2.0F};
    const float zero[] = {0.0F};
    const float sentinel[] = {-7.0F};
    const float source_value[] = {9.0F};
    const char *messages[] = {
        "Metal attention query depends on an earlier output",
        "Metal attention key depends on an earlier output",
        "Metal attention value depends on an earlier output",
        "Metal attention mask depends on an earlier output"
    };
    m3_op_test_fixture fixture;
    m3_test_attention_views views;
    m3_tensor_view source;
    m3_tensor_view *targets[4];
    m3_command commands[2];
    m3_command writer;
    m3_error error;
    size_t index;
    bool ok;

    if (!m3_test_attention_metal_fixture(test, &fixture)) {
        return;
    }
    ok = m3_op_test_tensor(&fixture, &views.query, M3_DTYPE_F32, 4U,
                           shape, one) &&
         m3_op_test_tensor(&fixture, &views.key, M3_DTYPE_F32, 4U, shape,
                           one) &&
         m3_op_test_tensor(&fixture, &views.value, M3_DTYPE_F32, 4U, shape,
                           two) &&
         m3_op_test_tensor(&fixture, &views.mask, M3_DTYPE_F32, 4U, shape,
                           zero) &&
         m3_op_test_tensor(&fixture, &views.output, M3_DTYPE_F32, 4U, shape,
                           sentinel) &&
         m3_op_test_tensor(&fixture, &source, M3_DTYPE_F32, 4U, shape,
                           source_value);
    M3_TEST_EXPECT(test, ok, "create Metal attention dependency tensors");
    if (!ok) {
        m3_op_test_fixture_dispose(&fixture);
        return;
    }
    m3_test_attention_command(&writer, &views, true, false, 0);
    M3_TEST_EXPECT(
        test,
        m3_metal_has_prior_writer(&writer, 1U, views.output.storage) &&
            !m3_metal_has_prior_writer(&writer, 1U,
                                       views.query.storage),
        "Metal attention writer hook owns only output storage");
    targets[0] = &views.query;
    targets[1] = &views.key;
    targets[2] = &views.value;
    targets[3] = &views.mask;
    for (index = 0U; index < 4U; ++index) {
        float prior = index == 0U ? NAN
                     : index == 2U ? 2.0F
                                   : index == 3U ? 0.0F : 1.0F;
        m3_status status;

        ok = m3_test_attention_write_scalar(&views.query, 1.0F) &&
             m3_test_attention_write_scalar(&views.key, 1.0F) &&
             m3_test_attention_write_scalar(&views.value, 2.0F) &&
             m3_test_attention_write_scalar(&views.mask, 0.0F) &&
             m3_test_attention_write_scalar(&views.output, -7.0F) &&
             m3_test_attention_write_scalar(targets[index], prior);
        commands[0].kind = M3_OP_COPY;
        commands[0].descriptor.copy.input = &source;
        commands[0].descriptor.copy.output = targets[index];
        m3_test_attention_command(&commands[1], &views, true, false, 0);
        status = ok ? m3_op_test_execute(
                          &fixture, commands, 2U, NULL, &error)
                    : M3_STATUS_INTERNAL;
        M3_TEST_EXPECT(
            test,
            status == M3_STATUS_UNSUPPORTED &&
                strcmp(m3_error_message(&error), messages[index]) == 0 &&
                ((index == 0U && isnan(m3_op_test_f32(targets[index])[0])) ||
                 (index != 0U &&
                  m3_op_test_f32(targets[index])[0] == prior)) &&
                m3_op_test_f32(&views.output)[0] == -7.0F,
            "Metal rejects every earlier attention input writer atomically");
    }
    M3_TEST_EXPECT(
        test,
        m3_op_test_execute(&fixture, commands, 2U, NULL, NULL) ==
                M3_STATUS_UNSUPPORTED &&
            m3_op_test_f32(&views.output)[0] == -7.0F,
        "Metal attention dependency accepts a null error sink");
    ok = m3_test_attention_write_scalar(&views.query, NAN) &&
         m3_test_attention_write_scalar(&views.output, -7.0F);
    commands[0].kind = M3_OP_COPY;
    commands[0].descriptor.copy.input = &source;
    commands[0].descriptor.copy.output = &views.output;
    m3_test_attention_command(&commands[1], &views, true, false, 0);
    M3_TEST_EXPECT(
        test,
        ok && m3_op_test_execute(&fixture, commands, 2U, NULL, &error) ==
                  M3_STATUS_OUT_OF_RANGE &&
            strcmp(m3_error_message(&error),
                   "attention query contains an invalid non-finite value") ==
                0 &&
            m3_op_test_f32(&views.output)[0] == -7.0F,
        "Metal shared data preflight preserves whole-list atomicity");
    ok = m3_test_attention_write_scalar(&views.query, FLT_MAX) &&
         m3_test_attention_write_scalar(&views.key, 2.0F) &&
         m3_test_attention_write_scalar(&views.mask, -INFINITY) &&
         m3_test_attention_write_scalar(&views.output, -7.0F);
    m3_test_attention_command(&commands[0], &views, true, false, 0);
    commands[0].descriptor.attention.scale = 1.0F;
    M3_TEST_EXPECT(
        test,
        ok && m3_op_test_execute(&fixture, commands, 1U, NULL, &error) ==
                  M3_STATUS_OUT_OF_RANGE &&
            strcmp(m3_error_message(&error),
                   "attention score is NaN after accumulation") == 0 &&
            m3_op_test_f32(&views.output)[0] == -7.0F,
        "Metal rejects NaN attention scores before commit");
    m3_op_test_fixture_dispose(&fixture);
}

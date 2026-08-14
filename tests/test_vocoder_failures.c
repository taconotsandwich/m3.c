/* SPDX-License-Identifier: GPL-2.0-only */

#include "vocoder_runtime_test.h"

#include "m3_test.h"
#include "weight_stage_fixture.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    m3_vocoder_materialize_io default_io;
    size_t maximum_read;
    size_t maximum_write;
    size_t read_calls;
    size_t write_calls;
    size_t fail_read;
    size_t fail_write;
} m3_vocoder_io_probe;

typedef struct {
    size_t calls;
    size_t cancel_at;
    size_t total;
    bool ordered;
} m3_vocoder_cancel_probe;

static bool m3_vocoder_cancel_progress(void *context, uint64_t completed,
                                       uint64_t total)
{
    m3_vocoder_cancel_probe *probe = context;

    if (completed != (uint64_t)probe->calls ||
        total != (uint64_t)probe->total) {
        probe->ordered = false;
    }
    ++probe->calls;
    return completed != (uint64_t)probe->cancel_at;
}

static m3_status m3_vocoder_probe_read(
    void *context, const m3_storage *storage, size_t byte_offset,
    void *destination, size_t byte_count, m3_error *error)
{
    m3_vocoder_io_probe *probe = context;

    ++probe->read_calls;
    if (byte_count > probe->maximum_read) {
        probe->maximum_read = byte_count;
    }
    if (probe->read_calls == probe->fail_read) {
        return m3_error_set(error, M3_STATUS_IO,
                            "injected vocoder read failure");
    }
    return probe->default_io.read_storage(
        probe->default_io.context, storage, byte_offset, destination,
        byte_count, error);
}

static m3_status m3_vocoder_probe_write(
    void *context, m3_storage *storage, size_t byte_offset,
    const void *source, size_t byte_count, m3_error *error)
{
    m3_vocoder_io_probe *probe = context;

    ++probe->write_calls;
    if (byte_count > probe->maximum_write) {
        probe->maximum_write = byte_count;
    }
    if (probe->write_calls == probe->fail_write) {
        return m3_error_set(error, M3_STATUS_IO,
                            "injected vocoder write failure");
    }
    return probe->default_io.write_storage(
        probe->default_io.context, storage, byte_offset, source,
        byte_count, error);
}

static void m3_vocoder_probe_io(m3_vocoder_materialize_io *io,
                                m3_vocoder_io_probe *probe)
{
    (void)memset(probe, 0, sizeof(*probe));
    m3_vocoder_materialize_io_init(&probe->default_io);
    io->context = probe;
    io->read_storage = m3_vocoder_probe_read;
    io->write_storage = m3_vocoder_probe_write;
}

static bool m3_vocoder_fake_fixture(
    uint64_t recommended, m3_vocoder_test_fixture *fixture,
    m3_weight_stage_fake_context **context, m3_error *error)
{
    m3_backend *backend = NULL;

    (void)memset(fixture, 0, sizeof(*fixture));
    return m3_weight_stage_test_fake_backend_create(
               UINT64_MAX, recommended, SIZE_MAX, &backend, context,
               error) &&
           m3_vocoder_test_fixture_create(
               fixture, backend, true, error);
}

void test_vocoder_nonfinite_and_zero_norm(m3_test_context *test)
{
    m3_vocoder_test_fixture fixture = {0};
    m3_vocoder_materialize_io io;
    m3_vocoder_runtime *runtime = NULL;
    m3_backend_allocation_stats baseline;
    m3_backend_allocation_stats after;
    m3_tensor_view *gain_view;
    m3_tensor_view *value_view;
    m3_error error;
    float gain[1] = {1.0F};
    float values[14] = {0};
    float zero_output[14] = {0};
    m3_status status;

    m3_error_reset(&error);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&fixture.backend, &error) ==
                           M3_STATUS_OK &&
                       m3_vocoder_test_fixture_create(
                           &fixture, fixture.backend, true, &error),
                   "create invalid-norm fixture");
    if (fixture.backend == NULL) {
        return;
    }
    gain_view = m3_vocoder_test_source(&fixture, "conv_out.weight_g");
    value_view = m3_vocoder_test_source(&fixture, "conv_out.weight_v");
    m3_vocoder_materialize_io_init(&io);
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       fixture.backend, &baseline, &error) == M3_STATUS_OK,
                   "read invalid-norm baseline");
    M3_TEST_EXPECT(test,
                   m3_vocoder_test_write_values(
                       gain_view, gain, 1U, &error) &&
                       m3_vocoder_test_write_values(
                           value_view, values, 14U, &error),
                   "write zero norm source");
    status = m3_vocoder_runtime_create_core(
        &runtime, &fixture.stage, &fixture.plan, &io, NULL, NULL, &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_INVALID_FORMAT && runtime == NULL,
                   "reject positive gain with zero finite norm atomically");
    values[0] = NAN;
    M3_TEST_EXPECT(test,
                   m3_vocoder_test_write_values(
                       value_view, values, 14U, &error),
                   "write non-finite value source");
    status = m3_vocoder_runtime_create_core(
        &runtime, &fixture.stage, &fixture.plan, &io, NULL, NULL, NULL);
    M3_TEST_EXPECT(test, status == M3_STATUS_INVALID_FORMAT && runtime == NULL,
                   "reject non-finite value with null error sink");
    values[0] = 1.0F;
    gain[0] = INFINITY;
    M3_TEST_EXPECT(test,
                   m3_vocoder_test_write_values(
                       gain_view, gain, 1U, &error) &&
                       m3_vocoder_test_write_values(
                           value_view, values, 14U, &error),
                   "write non-finite gain source");
    status = m3_vocoder_runtime_create_core(
        &runtime, &fixture.stage, &fixture.plan, &io, NULL, NULL, &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_INVALID_FORMAT && runtime == NULL,
                   "reject non-finite gain atomically");
    gain[0] = 1.0F;
    values[0] = FLT_MAX;
    M3_TEST_EXPECT(test,
                   m3_vocoder_test_write_values(
                       gain_view, gain, 1U, &error) &&
                       m3_vocoder_test_write_values(
                           value_view, values, 14U, &error),
                   "write finite overflowing norm source");
    status = m3_vocoder_runtime_create_core(
        &runtime, &fixture.stage, &fixture.plan, &io, NULL, NULL, &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_INVALID_FORMAT && runtime == NULL,
                   "reject non-finite accumulated norm");
    gain[0] = FLT_MAX;
    values[0] = 1.0e-10F;
    M3_TEST_EXPECT(test,
                   m3_vocoder_test_write_values(
                       gain_view, gain, 1U, &error) &&
                       m3_vocoder_test_write_values(
                           value_view, values, 14U, &error),
                   "write finite scale-overflow source");
    status = m3_vocoder_runtime_create_core(
        &runtime, &fixture.stage, &fixture.plan, &io, NULL, NULL, NULL);
    M3_TEST_EXPECT(test, status == M3_STATUS_INVALID_FORMAT && runtime == NULL,
                   "reject non-finite scale with null error sink");
    gain[0] = 0.0F;
    values[0] = 1.0F;
    M3_TEST_EXPECT(test,
                   m3_vocoder_test_write_values(
                       gain_view, gain, 1U, &error) &&
                       m3_vocoder_test_write_values(
                           value_view, values, 14U, &error),
                   "write zero finite gain with positive norm");
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, &error) == M3_STATUS_OK &&
                       runtime != NULL,
                   "accept zero finite gain with positive finite norm");
    M3_TEST_EXPECT(test,
                   runtime != NULL &&
                       m3_vocoder_test_read_values(
                           m3_vocoder_runtime_weights(runtime)
                               ->convolution_output_weight,
                           zero_output, 14U, &error) &&
                       zero_output[0] == 0.0F,
                   "zero gain materializes a finite zero row");
    m3_vocoder_runtime_free(runtime);
    runtime = NULL;
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       fixture.backend, &after, &error) == M3_STATUS_OK &&
                       after.live_allocated_bytes ==
                           baseline.live_allocated_bytes &&
                       after.live_storage_count == baseline.live_storage_count,
                   "all invalid numeric paths roll back owned storage");
    m3_vocoder_test_fixture_dispose(&fixture);
}

void test_vocoder_nonfinite_direct_sources(m3_test_context *test)
{
    m3_vocoder_test_fixture fixture = {0};
    m3_vocoder_materialize_io io;
    m3_vocoder_runtime *runtime = NULL;
    m3_backend_allocation_stats baseline;
    m3_backend_allocation_stats after;
    m3_backend *backend = NULL;
    m3_tensor_view *view;
    m3_error error;
    m3_status status;

    m3_error_reset(&error);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
                       m3_vocoder_test_fixture_create(
                           &fixture, backend, true, &error),
                   "create direct finite-validation fixture");
    if (fixture.backend == NULL) {
        return;
    }
    m3_vocoder_materialize_io_init(&io);
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       fixture.backend, &baseline, &error) == M3_STATUS_OK,
                   "read direct finite-validation baseline");

    view = m3_vocoder_test_source(&fixture, "dec_in_proj.weight");
    M3_TEST_EXPECT(test, m3_vocoder_test_fill(view, NAN, &error),
                   "write non-finite direct decoder weight");
    status = m3_vocoder_runtime_create_core(
        &runtime, &fixture.stage, &fixture.plan, &io, NULL, NULL, &error);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_INVALID_FORMAT && runtime == NULL &&
                       strstr(m3_error_message(&error),
                              "dec_in_proj.weight") != NULL,
                   "reject non-finite direct weight with its diagnostic");
    M3_TEST_EXPECT(test, m3_vocoder_test_fill(view, 0.25F, &error),
                   "restore finite direct decoder weight");

    view = m3_vocoder_test_source(&fixture, "blocks.0.conv_t1.bias");
    M3_TEST_EXPECT(test, m3_vocoder_test_fill(view, INFINITY, &error),
                   "write non-finite direct convolution bias");
    status = m3_vocoder_runtime_create_core(
        &runtime, &fixture.stage, &fixture.plan, &io, NULL, NULL, NULL);
    M3_TEST_EXPECT(test, status == M3_STATUS_INVALID_FORMAT && runtime == NULL,
                   "reject non-finite bias with null error sink");
    M3_TEST_EXPECT(test, m3_vocoder_test_fill(view, 0.5F, &error),
                   "restore finite convolution bias");

    view = m3_vocoder_test_source(&fixture, "blocks.0.snake1.alpha");
    M3_TEST_EXPECT(test, m3_vocoder_test_fill(view, -INFINITY, &error),
                   "write non-finite direct Snake alpha");
    status = m3_vocoder_runtime_create_core(
        &runtime, &fixture.stage, &fixture.plan, &io, NULL, NULL, &error);
    M3_TEST_EXPECT(test,
                   status == M3_STATUS_INVALID_FORMAT && runtime == NULL &&
                       strstr(m3_error_message(&error),
                              "blocks.0.snake1.alpha") != NULL,
                   "reject non-finite alpha with its diagnostic");
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       fixture.backend, &after, &error) == M3_STATUS_OK &&
                       after.live_allocated_bytes ==
                           baseline.live_allocated_bytes &&
                       after.live_storage_count == baseline.live_storage_count,
                   "all non-finite direct rows roll back owned storage");
    m3_vocoder_test_fixture_dispose(&fixture);
}

void test_vocoder_missing_and_mismatched_sources(m3_test_context *test)
{
    m3_vocoder_test_fixture fixture = {0};
    m3_vocoder_materialize_io io;
    m3_vocoder_runtime *runtime = NULL;
    m3_vocoder_runtime *sentinel = (m3_vocoder_runtime *)(uintptr_t)1U;
    m3_tensor_view *view;
    m3_dtype saved_dtype;
    char saved_name[M3_VOCODER_NAME_CAPACITY];
    m3_error error;

    m3_error_reset(&error);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&fixture.backend, &error) ==
                           M3_STATUS_OK &&
                       m3_vocoder_test_fixture_create(
                           &fixture, fixture.backend, true, &error),
                   "create source-contract fixture");
    if (fixture.backend == NULL) {
        return;
    }
    m3_vocoder_materialize_io_init(&io);
    (void)memcpy(saved_name, fixture.plan.entries[0].source_names[0],
                 sizeof(saved_name));
    (void)memcpy(fixture.plan.entries[0].source_names[0], "missing.weight",
                 sizeof("missing.weight"));
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, &error) == M3_STATUS_INVALID_FORMAT &&
                       runtime == NULL,
                   "reject missing exact source name");
    (void)memcpy(fixture.plan.entries[0].source_names[0], saved_name,
                 sizeof(saved_name));
    view = m3_vocoder_test_source(&fixture, "dec_in_proj.weight");
    saved_dtype = view->metadata.dtype;
    view->metadata.dtype = M3_DTYPE_F16;
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, NULL) == M3_STATUS_INVALID_FORMAT &&
                       runtime == NULL,
                   "reject mismatched source dtype with null error sink");
    view->metadata.dtype = saved_dtype;
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &sentinel, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, &error) == M3_STATUS_INVALID_ARGUMENT &&
                       sentinel == (m3_vocoder_runtime *)(uintptr_t)1U,
                   "nonempty output pointer remains unchanged on error");
    --fixture.stage.view_count;
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, &error) == M3_STATUS_INVALID_FORMAT &&
                       runtime == NULL,
                   "reject non-exact source inventory");
    ++fixture.stage.view_count;
    m3_vocoder_test_fixture_dispose(&fixture);
}

void test_vocoder_backend_limits_and_rollback(m3_test_context *test)
{
    m3_vocoder_test_fixture fixture = {0};
    m3_weight_stage_fake_context *context = NULL;
    m3_vocoder_materialize_io io;
    m3_vocoder_runtime *runtime = NULL;
    m3_backend_allocation_stats baseline;
    m3_backend_allocation_stats after;
    m3_error error;
    size_t source_bytes;
    size_t runtime_bytes;

    m3_error_reset(&error);
    M3_TEST_EXPECT(test,
                   m3_vocoder_fake_fixture(
                       0U, &fixture, &context, &error),
                   "create allocation-fault backend fixture");
    if (fixture.backend == NULL) {
        return;
    }
    source_bytes = m3_vocoder_test_source_bytes(&fixture);
    runtime_bytes = m3_vocoder_test_runtime_bytes(&fixture);
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       fixture.backend, &baseline, &error) == M3_STATUS_OK,
                   "read allocation-fault baseline");
    context->fail_allocation_call = context->allocation_calls + 5U;
    m3_vocoder_materialize_io_init(&io);
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, &error) == M3_STATUS_OUT_OF_MEMORY &&
                       runtime == NULL,
                   "injected destination allocation failure is atomic");
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       fixture.backend, &after, &error) == M3_STATUS_OK &&
                       after.live_allocated_bytes == source_bytes &&
                       after.live_storage_count ==
                           fixture.plan.source_count &&
                       after.live_allocated_bytes ==
                           baseline.live_allocated_bytes &&
                       after.live_storage_count == baseline.live_storage_count,
                   "partial destination allocations roll back completely");
    context->fail_allocation_call = SIZE_MAX;
    m3_vocoder_test_fixture_dispose(&fixture);

    context = NULL;
    M3_TEST_EXPECT(test,
                   m3_vocoder_fake_fixture(
                       (uint64_t)(source_bytes + runtime_bytes - 1U),
                       &fixture, &context, &error),
                   "create working-set-limit fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, &error) == M3_STATUS_OUT_OF_MEMORY &&
                       runtime == NULL,
                   "reject live plus new recommended working-set overflow");
    m3_vocoder_test_fixture_dispose(&fixture);

    context = NULL;
    M3_TEST_EXPECT(test,
                   m3_vocoder_fake_fixture(
                       (uint64_t)(source_bytes + runtime_bytes),
                       &fixture, &context, &error),
                   "create exact working-set-limit fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, &error) == M3_STATUS_OK &&
                       runtime != NULL,
                   "accept exact live plus new working-set boundary");
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       fixture.backend, &after, &error) == M3_STATUS_OK &&
                       after.live_allocated_bytes ==
                           source_bytes + runtime_bytes &&
                       after.peak_allocated_bytes ==
                           source_bytes + runtime_bytes,
                   "exact working-set boundary locks live and peak bytes");
    m3_vocoder_runtime_free(runtime);
    m3_vocoder_test_fixture_dispose(&fixture);
}

void test_vocoder_cancellation_and_bounded_io(m3_test_context *test)
{
    m3_vocoder_test_fixture fixture = {0};
    m3_vocoder_materialize_io io;
    m3_vocoder_io_probe io_probe;
    m3_vocoder_cancel_probe cancel;
    m3_vocoder_runtime *runtime = NULL;
    m3_backend_allocation_stats baseline;
    m3_backend_allocation_stats after;
    m3_error error;
    size_t boundary;

    m3_error_reset(&error);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&fixture.backend, &error) ==
                           M3_STATUS_OK &&
                       m3_vocoder_test_fixture_create(
                           &fixture, fixture.backend, true, &error),
                   "create cancellation fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       fixture.backend, &baseline, &error) == M3_STATUS_OK,
                   "read cancellation baseline");
    for (boundary = 0U; boundary <= fixture.plan.source_count; ++boundary) {
        m3_vocoder_materialize_io_init(&io);
        cancel.calls = 0U;
        cancel.cancel_at = boundary;
        cancel.total = fixture.plan.source_count;
        cancel.ordered = true;
        M3_TEST_EXPECT(test,
                       m3_vocoder_runtime_create_core(
                           &runtime, &fixture.stage, &fixture.plan, &io,
                           m3_vocoder_cancel_progress, &cancel,
                           &error) == M3_STATUS_CANCELLED &&
                           runtime == NULL && cancel.ordered &&
                           cancel.calls == boundary + 1U,
                       "cancellation boundary is ordered and atomic");
        M3_TEST_EXPECT(test,
                       m3_backend_get_allocation_stats(
                           fixture.backend, &after, &error) == M3_STATUS_OK &&
                           after.live_allocated_bytes ==
                               baseline.live_allocated_bytes &&
                           after.live_storage_count ==
                               baseline.live_storage_count,
                       "cancellation frees every destination allocation");
    }
    m3_vocoder_probe_io(&io, &io_probe);
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, &error) == M3_STATUS_OK,
                   "materialize through bounded I/O seam");
    M3_TEST_EXPECT(test, io_probe.maximum_read <=
                                 M3_VOCODER_MAXIMUM_ROW_BYTES &&
                             io_probe.maximum_write <=
                                 M3_VOCODER_MAXIMUM_ROW_BYTES &&
                             io_probe.read_calls != 0U &&
                             io_probe.write_calls != 0U,
                   "all source reads and destination writes are row bounded");
    m3_vocoder_runtime_free(runtime);
    runtime = NULL;
    m3_vocoder_probe_io(&io, &io_probe);
    io_probe.fail_read = 3U;
    m3_error_reset(&error);
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, &error) == M3_STATUS_IO &&
                       runtime == NULL && error.status == M3_STATUS_IO &&
                       strcmp(m3_error_message(&error),
                              "injected vocoder read failure") == 0,
                   "bounded read failure preserves its first diagnostic");
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       fixture.backend, &after, &error) == M3_STATUS_OK &&
                       after.live_allocated_bytes ==
                           baseline.live_allocated_bytes &&
                       after.live_storage_count == baseline.live_storage_count,
                   "bounded read failure rolls back all destinations");
    m3_vocoder_probe_io(&io, &io_probe);
    io_probe.fail_read = 2U;
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, NULL) == M3_STATUS_IO && runtime == NULL,
                   "bounded read failure supports a null error sink");
    m3_vocoder_probe_io(&io, &io_probe);
    io_probe.fail_write = 3U;
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, NULL) == M3_STATUS_IO && runtime == NULL,
                   "injected bounded write failure is atomic with null error");
    m3_vocoder_test_fixture_dispose(&fixture);
}

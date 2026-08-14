/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_waveform_internal.h"

#include "waveform_test.h"
#include "weight_stage_fixture.h"

#include <stdint.h>
#include <string.h>

typedef struct {
    uint64_t cancel_at;
    uint64_t total;
    size_t calls;
    bool valid;
} m3_waveform_failure_progress;

typedef struct {
    size_t read_calls;
    size_t write_calls;
    size_t fail_read_call;
    size_t fail_write_call;
    size_t maximum_read;
    size_t maximum_write;
} m3_waveform_fault_io;

static void m3_waveform_failure_progress_init(
    m3_waveform_failure_progress *progress, uint64_t total,
    uint64_t cancel_at)
{
    (void)memset(progress, 0, sizeof(*progress));
    progress->cancel_at = cancel_at;
    progress->total = total;
    progress->valid = true;
}

static bool m3_waveform_failure_progress_call(
    void *context, uint64_t completed, uint64_t total)
{
    m3_waveform_failure_progress *progress = context;

    progress->valid = progress->valid && total == progress->total &&
                      completed == (uint64_t)progress->calls;
    ++progress->calls;
    return completed != progress->cancel_at;
}

static m3_status m3_waveform_fault_read(
    void *context, const m3_storage *storage, size_t byte_offset,
    void *destination, size_t byte_count, m3_error *error)
{
    m3_waveform_fault_io *io = context;

    ++io->read_calls;
    if (byte_count > io->maximum_read) {
        io->maximum_read = byte_count;
    }
    if (io->read_calls == io->fail_read_call) {
        return m3_error_set(error, M3_STATUS_IO,
                            "injected waveform read failure");
    }
    return m3_storage_read(
        storage, byte_offset, destination, byte_count, error);
}

static m3_status m3_waveform_fault_write(
    void *context, m3_storage *storage, size_t byte_offset,
    const void *source, size_t byte_count, m3_error *error)
{
    m3_waveform_fault_io *io = context;

    ++io->write_calls;
    if (byte_count > io->maximum_write) {
        io->maximum_write = byte_count;
    }
    if (io->write_calls == io->fail_write_call) {
        return m3_error_set(error, M3_STATUS_IO,
                            "injected waveform write failure");
    }
    return m3_storage_write(
        storage, byte_offset, source, byte_count, error);
}

static void m3_waveform_fault_io_init(
    m3_waveform_fault_io *context, m3_waveform_io *io,
    size_t fail_read, size_t fail_write)
{
    (void)memset(context, 0, sizeof(*context));
    context->fail_read_call = fail_read;
    context->fail_write_call = fail_write;
    io->context = context;
    io->read_storage = m3_waveform_fault_read;
    io->write_storage = m3_waveform_fault_write;
}

void test_waveform_assembly_io_faults(m3_test_context *test)
{
    const uint32_t sentinel_bits[] = {
        UINT32_C(0xc0e00000), UINT32_C(0x41300000)
    };
    m3_waveform_test_fixture fixture;
    m3_waveform_fault_io fault;
    m3_waveform_io io;
    m3_vocoder_output output;
    m3_backend *backend = NULL;
    m3_storage *published = NULL;
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    uint32_t sentinel_after[2U] = {0U};
    m3_error error;
    bool created;

    m3_waveform_test_fixture_init(&fixture);
    m3_vocoder_output_init(&output);
    created = m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
              m3_waveform_test_fixture_create(
                  &fixture, backend, 1U, &error) &&
              m3_waveform_test_output_create(
                  backend, sentinel_bits, &output, &error);
    M3_TEST_EXPECT(test, created, "create waveform fault fixture");
    if (!created) {
        m3_vocoder_output_dispose(&output);
        m3_waveform_test_fixture_dispose(&fixture);
        m3_backend_free(backend);
        return;
    }
    published = output.storage;
    m3_waveform_fault_io_init(&fault, &io, 3U, 0U);
    (void)m3_backend_get_allocation_stats(backend, &before, &error);
    M3_TEST_EXPECT(
        test,
        m3_waveform_assemble_io(
            fixture.chunks, fixture.chunk_count, 1U, &io, NULL, NULL,
            &output, &error) == M3_STATUS_IO &&
            fault.read_calls == 3U && fault.maximum_read == 1024U &&
            output.storage == published &&
            m3_backend_get_allocation_stats(
                backend, &after, &error) == M3_STATUS_OK &&
            before.live_allocated_bytes == after.live_allocated_bytes &&
            before.live_storage_count == after.live_storage_count,
        "read fault frees the private output and preserves publication");
    m3_waveform_fault_io_init(&fault, &io, 0U, 3U);
    (void)m3_backend_get_allocation_stats(backend, &before, &error);
    M3_TEST_EXPECT(
        test,
        m3_waveform_assemble_io(
            fixture.chunks, fixture.chunk_count, 1U, &io, NULL, NULL,
            &output, &error) == M3_STATUS_IO &&
            fault.write_calls == 3U && fault.maximum_write == 1024U &&
            output.storage == published &&
            m3_backend_get_allocation_stats(
                backend, &after, &error) == M3_STATUS_OK &&
            before.live_allocated_bytes == after.live_allocated_bytes &&
            before.live_storage_count == after.live_storage_count,
        "write fault frees the private output and preserves publication");
    M3_TEST_EXPECT(
        test,
        m3_storage_read(
            output.storage, 0U, sentinel_after, sizeof(sentinel_after),
            &error) == M3_STATUS_OK &&
            memcmp(sentinel_after, sentinel_bits,
                   sizeof(sentinel_bits)) == 0,
        "I/O faults preserve prior waveform bytes");
    m3_waveform_fault_io_init(&fault, &io, 0U, 0U);
    M3_TEST_EXPECT(
        test,
        m3_waveform_assemble_io(
            fixture.chunks, fixture.chunk_count, 1U, &io, NULL, NULL,
            &output, &error) == M3_STATUS_OK &&
            output.storage != published && fault.read_calls == 12U &&
            fault.write_calls == 12U && fault.maximum_read == 1024U &&
            fault.maximum_write == 1024U &&
            m3_waveform_test_output_matches(&fixture, &output, &error),
        "single scheduler keeps all storage transfers bounded");
    M3_TEST_EXPECT(
        test,
        m3_waveform_assemble_io(
            fixture.chunks, fixture.chunk_count, 1U, NULL, NULL, NULL,
            &output, NULL) == M3_STATUS_INVALID_ARGUMENT &&
            output.storage != NULL,
        "invalid internal I/O seam preserves published output");
    m3_vocoder_output_dispose(&output);
    m3_waveform_test_fixture_dispose(&fixture);
    m3_backend_free(backend);
}

void test_waveform_assembly_cancel_retry(m3_test_context *test)
{
    const uint32_t sentinel_bits[] = {
        UINT32_C(0xc0e00000), UINT32_C(0x41300000)
    };
    const uint64_t cancel_points[] = {0U, 1U, 2U};
    m3_waveform_test_fixture fixture;
    m3_waveform_failure_progress progress;
    m3_vocoder_output output;
    m3_backend *backend = NULL;
    m3_storage *published = NULL;
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    m3_error error;
    size_t index;
    bool created;

    m3_waveform_test_fixture_init(&fixture);
    m3_vocoder_output_init(&output);
    created = m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
              m3_waveform_test_fixture_create(
                  &fixture, backend, 1U, &error) &&
              m3_waveform_test_output_create(
                  backend, sentinel_bits, &output, &error);
    M3_TEST_EXPECT(test, created, "create waveform cancellation fixture");
    if (!created) {
        m3_vocoder_output_dispose(&output);
        m3_waveform_test_fixture_dispose(&fixture);
        m3_backend_free(backend);
        return;
    }
    published = output.storage;
    for (index = 0U; index < sizeof(cancel_points) /
                                     sizeof(cancel_points[0]); ++index) {
        m3_waveform_failure_progress_init(
            &progress, 2U, cancel_points[index]);
        (void)m3_backend_get_allocation_stats(backend, &before, &error);
        M3_TEST_EXPECT(
            test,
            m3_waveform_assemble(
                fixture.chunks, fixture.chunk_count, 1U,
                m3_waveform_failure_progress_call, &progress, &output,
                &error) == M3_STATUS_CANCELLED &&
                progress.valid &&
                progress.calls == (size_t)cancel_points[index] + 1U &&
                output.storage == published &&
                m3_backend_get_allocation_stats(
                    backend, &after, &error) == M3_STATUS_OK &&
                before.live_allocated_bytes == after.live_allocated_bytes &&
                before.live_storage_count == after.live_storage_count,
            "every channel boundary cancellation is output atomic");
    }
    M3_TEST_EXPECT(
        test,
        m3_waveform_assemble(
            fixture.chunks, 2U, 1U, NULL, NULL, &output, NULL) ==
                M3_STATUS_INVALID_ARGUMENT &&
            output.storage == published,
        "null-error topology failure preserves prior output");
    m3_waveform_failure_progress_init(&progress, 2U, UINT64_MAX);
    M3_TEST_EXPECT(
        test,
        m3_waveform_assemble(
            fixture.chunks, fixture.chunk_count, 1U,
            m3_waveform_failure_progress_call, &progress, &output,
            &error) == M3_STATUS_OK &&
            progress.valid && progress.calls == 3U &&
            output.storage != published &&
            m3_waveform_test_output_matches(&fixture, &output, &error),
        "successful retry replaces prior output after final checkpoint");
    m3_vocoder_output_dispose(&output);
    m3_waveform_test_fixture_dispose(&fixture);
    m3_backend_free(backend);
}

void test_waveform_assembly_validation_atomicity(m3_test_context *test)
{
    const uint32_t sentinel_bits[] = {
        UINT32_C(0xc0e00000), UINT32_C(0x41300000)
    };
    m3_weight_stage_fake_context *context = NULL;
    m3_waveform_test_fixture fixture;
    m3_waveform_failure_progress progress;
    m3_vocoder_output output;
    m3_backend *backend = NULL;
    m3_tensor_view malformed;
    m3_tensor_view saved_output_view;
    m3_storage *published = NULL;
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    size_t allocation_calls;
    size_t free_calls;
    m3_error error;
    bool created;

    m3_waveform_test_fixture_init(&fixture);
    m3_vocoder_output_init(&output);
    created = m3_weight_stage_test_fake_backend_create(
                  (uint64_t)SIZE_MAX, 0U, 0U, &backend, &context,
                  &error) &&
              m3_waveform_test_fixture_create(
                  &fixture, backend, 1U, &error) &&
              m3_waveform_test_output_create(
                  backend, sentinel_bits, &output, &error);
    M3_TEST_EXPECT(test, created, "create waveform validation fixture");
    if (!created) {
        m3_vocoder_output_dispose(&output);
        m3_waveform_test_fixture_dispose(&fixture);
        m3_backend_free(backend);
        return;
    }
    published = output.storage;
    malformed = fixture.chunks[0];
    ++malformed.metadata.element_count;
    allocation_calls = context->allocation_calls;
    free_calls = context->free_calls;
    m3_waveform_failure_progress_init(&progress, 2U, UINT64_MAX);
    (void)m3_backend_get_allocation_stats(backend, &before, &error);
    M3_TEST_EXPECT(
        test,
        m3_tensor_is_contiguous(&malformed) &&
            m3_waveform_assemble(
                &malformed, 1U, 1U,
                m3_waveform_failure_progress_call, &progress, &output,
                &error) == M3_STATUS_INVALID_ARGUMENT &&
            progress.calls == 0U &&
            context->allocation_calls == allocation_calls &&
            context->free_calls == free_calls &&
            output.storage == published &&
            m3_backend_get_allocation_stats(
                backend, &after, &error) == M3_STATUS_OK &&
            before.live_allocated_bytes == after.live_allocated_bytes &&
            before.live_storage_count == after.live_storage_count,
        "reject inconsistent contiguous metadata before progress or allocation");
    malformed = fixture.chunks[0];
    malformed.byte_offset = sizeof(float);
    allocation_calls = context->allocation_calls;
    free_calls = context->free_calls;
    m3_waveform_failure_progress_init(&progress, 2U, UINT64_MAX);
    M3_TEST_EXPECT(
        test,
        m3_tensor_is_contiguous(&malformed) &&
            m3_waveform_assemble(
                &malformed, 1U, 1U,
                m3_waveform_failure_progress_call, &progress, &output,
                NULL) == M3_STATUS_OUT_OF_RANGE &&
            progress.calls == 0U &&
            context->allocation_calls == allocation_calls &&
            context->free_calls == free_calls &&
            output.storage == published,
        "reject out-of-bounds contiguous view before all side effects");
    saved_output_view = output.waveform;
    output.waveform.storage = fixture.storages[0];
    allocation_calls = context->allocation_calls;
    free_calls = context->free_calls;
    m3_waveform_failure_progress_init(&progress, 2U, UINT64_MAX);
    M3_TEST_EXPECT(
        test,
        m3_waveform_assemble(
            fixture.chunks, 1U, 1U,
            m3_waveform_failure_progress_call, &progress, &output,
            &error) == M3_STATUS_INVALID_ARGUMENT &&
            progress.calls == 0U &&
            context->allocation_calls == allocation_calls &&
            context->free_calls == free_calls && output.storage == published,
        "reject inconsistent nonempty output ownership before side effects");
    output.waveform = saved_output_view;
    context->fail_allocation_call = context->allocation_calls + 1U;
    m3_waveform_failure_progress_init(&progress, 2U, UINT64_MAX);
    M3_TEST_EXPECT(
        test,
        m3_waveform_assemble(
            fixture.chunks, 1U, 1U,
            m3_waveform_failure_progress_call, &progress, &output,
            &error) == M3_STATUS_OUT_OF_MEMORY &&
            progress.calls == 1U && output.storage == published,
        "backend allocation fault preserves prior output after checkpoint zero");
    m3_vocoder_output_dispose(&output);
    m3_waveform_test_fixture_dispose(&fixture);
    m3_backend_free(backend);
    (void)context;
}

static void m3_waveform_maximum_case(
    m3_test_context *test, uint64_t maximum_storage,
    uint64_t recommended, m3_status expected, const char *description)
{
    const size_t input_samples = 352768U;
    const size_t input_bytes = 2822144U;
    const uint64_t shape[] = {1U, 2U, 352768U};
    m3_weight_stage_fake_context *context = NULL;
    m3_waveform_failure_progress progress;
    m3_vocoder_output output;
    m3_backend *backend = NULL;
    m3_storage *storage = NULL;
    m3_tensor_view chunks[89U];
    m3_error error;
    size_t allocation_calls = 0U;
    size_t free_calls = 0U;
    size_t index;
    bool created;

    (void)input_samples;
    m3_vocoder_output_init(&output);
    created = m3_weight_stage_test_fake_backend_create(
                  maximum_storage, recommended, 0U, &backend, &context,
                  &error) &&
              m3_storage_allocate(
                  backend, input_bytes, 64U, &storage, &error) ==
                  M3_STATUS_OK;
    for (index = 0U; index < 89U && created; ++index) {
        m3_tensor_view_init(&chunks[index]);
        created = m3_tensor_view_contiguous(
                      &chunks[index], storage, M3_DTYPE_F32, 3U, shape,
                      0U, &error) == M3_STATUS_OK;
    }
    M3_TEST_EXPECT(test, created, "create maximum waveform metadata fixture");
    if (!created) {
        m3_storage_free(storage);
        m3_backend_free(backend);
        return;
    }
    allocation_calls = context->allocation_calls;
    free_calls = context->free_calls;
    m3_waveform_failure_progress_init(&progress, 178U, 0U);
    M3_TEST_EXPECT(
        test,
        m3_waveform_test_output_samples(9000U) == 15897088U &&
            m3_waveform_assemble(
                chunks, 89U, 9000U,
                m3_waveform_failure_progress_call, &progress, &output,
                &error) == expected &&
            context->allocation_calls == allocation_calls &&
            context->free_calls == free_calls && output.storage == NULL &&
            ((expected == M3_STATUS_CANCELLED && progress.valid &&
              progress.calls == 1U) ||
             (expected != M3_STATUS_CANCELLED && progress.calls == 0U)),
        description);
    m3_storage_free(storage);
    m3_backend_free(backend);
    (void)context;
}

void test_waveform_assembly_maximum_preflight(m3_test_context *test)
{
    const uint64_t output_bytes = UINT64_C(127176704);
    const uint64_t input_bytes = UINT64_C(2822144);

    m3_waveform_maximum_case(
        test, output_bytes - 1U, 0U, M3_STATUS_OUT_OF_RANGE,
        "maximum metadata rejects output one byte above storage limit");
    m3_waveform_maximum_case(
        test, output_bytes, input_bytes + output_bytes - 1U,
        M3_STATUS_OUT_OF_MEMORY,
        "maximum metadata rejects live plus output one byte above working set");
    m3_waveform_maximum_case(
        test, output_bytes, input_bytes + output_bytes,
        M3_STATUS_CANCELLED,
        "maximum metadata locks 89 chunks and exact 127176704-byte output");
}

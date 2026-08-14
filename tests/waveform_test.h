/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_WAVEFORM_TEST_H
#define M3_WAVEFORM_TEST_H

#include "m3_vocoder_runtime.h"

#include <stdbool.h>
#include <stdint.h>

#define M3_WAVEFORM_TEST_MAX_CHUNKS 89U

typedef struct {
    m3_backend *backend;
    m3_storage *storages[M3_WAVEFORM_TEST_MAX_CHUNKS];
    m3_tensor_view chunks[M3_WAVEFORM_TEST_MAX_CHUNKS];
    size_t chunk_count;
    uint64_t frame_count;
} m3_waveform_test_fixture;

void m3_waveform_test_fixture_init(m3_waveform_test_fixture *fixture);
bool m3_waveform_test_fixture_create(
    m3_waveform_test_fixture *fixture, m3_backend *backend,
    uint64_t frame_count, m3_error *error);
void m3_waveform_test_fixture_dispose(m3_waveform_test_fixture *fixture);
uint32_t m3_waveform_test_pattern(
    size_t chunk, size_t channel, size_t sample);
size_t m3_waveform_test_output_samples(uint64_t frame_count);
bool m3_waveform_test_output_matches(
    const m3_waveform_test_fixture *fixture,
    const m3_vocoder_output *output, m3_error *error);
bool m3_waveform_test_output_create(
    m3_backend *backend, const uint32_t bits[2U],
    m3_vocoder_output *output, m3_error *error);

#endif

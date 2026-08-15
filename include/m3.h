/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_H
#define M3_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_VERSION_MAJOR 0
#define M3_VERSION_MINOR 1
#define M3_VERSION_PATCH 0
#define M3_VERSION_STRING "0.1.0"

typedef enum {
    M3_STATUS_OK = 0,
    M3_STATUS_INVALID_ARGUMENT = 1,
    M3_STATUS_OUT_OF_RANGE = 2,
    M3_STATUS_OVERFLOW = 3,
    M3_STATUS_OUT_OF_MEMORY = 4,
    M3_STATUS_IO = 5,
    M3_STATUS_INVALID_FORMAT = 6,
    M3_STATUS_UNSUPPORTED = 7,
    M3_STATUS_CANCELLED = 8,
    M3_STATUS_INTERNAL = 9
} m3_status;

#define M3_ERROR_MESSAGE_CAPACITY 256U

typedef struct {
    m3_status status;
    char message[M3_ERROR_MESSAGE_CAPACITY];
} m3_error;

typedef struct {
    const char *data;
    size_t length;
} m3_text_view;

typedef struct m3_music3_engine m3_music3_engine;
typedef struct m3_music3_output m3_music3_output;

typedef struct {
    m3_text_view caption;
    m3_text_view lyrics;
    uint64_t maximum_frames;
    uint64_t seed;
    uint64_t sequence;
} m3_music3_request;

typedef enum {
    M3_MUSIC3_PHASE_PREPARE = 0,
    M3_MUSIC3_PHASE_STAGE_SEMANTIC,
    M3_MUSIC3_PHASE_SEMANTIC,
    M3_MUSIC3_PHASE_STAGE_FLOW,
    M3_MUSIC3_PHASE_FLOW,
    M3_MUSIC3_PHASE_STAGE_VOCODER,
    M3_MUSIC3_PHASE_MATERIALIZE_VOCODER,
    M3_MUSIC3_PHASE_DECODE,
    M3_MUSIC3_PHASE_ASSEMBLE,
    M3_MUSIC3_PHASE_COUNT
} m3_music3_phase;

typedef bool (*m3_music3_progress_callback)(
    void *context, m3_music3_phase phase, uint64_t completed,
    uint64_t total);

typedef struct {
    uint32_t sample_rate;
    uint32_t channel_count;
    uint64_t samples_per_channel;
} m3_music3_output_info;

const char *m3_version(void);
const char *m3_status_string(m3_status status);
void m3_error_reset(m3_error *error);
const char *m3_error_message(const m3_error *error);

/* trusted_model_root is a caller-trusted local directory. Opening validates
 * the official Music3 structure and tensor contracts, but does not
 * authenticate a repository revision or file contents. engine and *engine
 * must be non-null and null respectively; failure leaves *engine null. */
m3_status m3_music3_engine_open(m3_music3_engine **engine,
                                const char *trusted_model_root,
                                m3_error *error);
void m3_music3_engine_free(m3_music3_engine *engine);

/* The engine is non-reentrant and not thread-safe. Success atomically replaces
 * a null or same-engine output. Failure and cancellation preserve it. While
 * generation or its progress callback is active, the caller must not mutate
 * or free *output, or free the engine. */
m3_status m3_music3_generate(
    m3_music3_engine *engine, const m3_music3_request *request,
    m3_music3_progress_callback progress, void *progress_context,
    m3_music3_output **output, m3_error *error);

/* Every output borrows its engine backend. All outputs must be freed before
 * their engine. */
void m3_music3_output_free(m3_music3_output *output);
m3_status m3_music3_output_get_info(
    const m3_music3_output *output, m3_music3_output_info *info,
    m3_error *error);
m3_status m3_music3_output_read_channel_f32(
    const m3_music3_output *output, uint32_t channel,
    uint64_t sample_offset, float *destination, size_t sample_count,
    m3_error *error);
m3_status m3_music3_output_write_wav(const m3_music3_output *output,
                                     const char *path,
                                     m3_error *error);

#endif

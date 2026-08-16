/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3.h"

#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const uint64_t fixed_phase_totals[M3_MUSIC3_PHASE_COUNT] = {
    UINT64_C(1),
    UINT64_C(18461001728),
    UINT64_C(0),
    UINT64_C(9828295204),
    UINT64_C(0),
    UINT64_C(216682888),
    UINT64_C(121),
    UINT64_C(0),
    UINT64_C(0),
};

static const char *const phase_names[M3_MUSIC3_PHASE_COUNT] = {
    "prepare",
    "stage_semantic",
    "semantic",
    "stage_flow",
    "flow",
    "stage_vocoder",
    "materialize_vocoder",
    "decode",
    "assemble",
};

typedef struct {
    int phase;
    uint64_t completed;
    uint64_t total;
    uint64_t maximum_frames;
    uint64_t chunk_count;
    bool invalid;
    bool phase_seen[M3_MUSIC3_PHASE_COUNT];
    double phase_start[M3_MUSIC3_PHASE_COUNT];
    double phase_end[M3_MUSIC3_PHASE_COUNT];
} progress_state;

static struct timespec benchmark_epoch;
static volatile sig_atomic_t stop_requested;

static double elapsed_seconds(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return -1.0;
    }
    return (double)(now.tv_sec - benchmark_epoch.tv_sec) +
           (double)(now.tv_nsec - benchmark_epoch.tv_nsec) / 1.0e9;
}

static void request_stop(int signal_number)
{
    (void)signal_number;
    stop_requested = 1;
}

static bool expected_phase_total(progress_state *state,
                                 m3_music3_phase phase, uint64_t total)
{
    uint64_t expected = fixed_phase_totals[phase];

    if (phase == M3_MUSIC3_PHASE_SEMANTIC) {
        if (state->maximum_frames > (UINT64_MAX - UINT64_C(48)) /
                                        UINT64_C(49)) {
            return false;
        }
        expected = UINT64_C(49) * state->maximum_frames + UINT64_C(48);
    } else if (phase == M3_MUSIC3_PHASE_FLOW) {
        if (total == 0U || total % UINT64_C(30) != 0U) {
            return false;
        }
        state->chunk_count = total / UINT64_C(30);
        return true;
    } else if (phase == M3_MUSIC3_PHASE_DECODE) {
        expected = UINT64_C(73) * state->chunk_count;
    } else if (phase == M3_MUSIC3_PHASE_ASSEMBLE) {
        expected = UINT64_C(2) * state->chunk_count;
    }
    return total == expected;
}

static bool report_progress(void *context, m3_music3_phase phase,
                            uint64_t completed, uint64_t total)
{
    progress_state *state = context;
    int value = (int)phase;
    double elapsed = elapsed_seconds();
    bool valid = value >= 0 && value < (int)M3_MUSIC3_PHASE_COUNT &&
                 completed <= total &&
                 expected_phase_total(state, phase, total);

    if (valid && state->phase < 0) {
        valid = value == (int)M3_MUSIC3_PHASE_PREPARE && completed == 0U;
    } else if (valid && value == state->phase) {
        valid = total == state->total && completed > state->completed;
    } else if (valid) {
        valid = value == state->phase + 1 &&
                state->completed == state->total && completed == 0U;
    }
    if (!valid) {
        state->invalid = true;
        (void)fprintf(stderr,
                      "INVALID_PROGRESS phase=%d completed=%llu total=%llu\n",
                      value, (unsigned long long)completed,
                      (unsigned long long)total);
        return false;
    }
    state->phase = value;
    state->completed = completed;
    state->total = total;
    if (!state->phase_seen[value]) {
        state->phase_seen[value] = true;
        state->phase_start[value] = elapsed;
        (void)fprintf(stderr,
                      "PHASE_BEGIN phase=%s elapsed=%.6f total=%llu\n",
                      phase_names[value], elapsed,
                      (unsigned long long)total);
    }
    if (completed == total) {
        state->phase_end[value] = elapsed;
        (void)fprintf(stderr,
                      "PHASE_END phase=%s elapsed=%.6f duration=%.6f\n",
                      phase_names[value], elapsed,
                      elapsed - state->phase_start[value]);
    }
    (void)fflush(stderr);
    return stop_requested == 0;
}

static bool parse_u64(const char *text, uint64_t *value)
{
    char *end = NULL;
    unsigned long long parsed;

    if (text[0] == '-') {
        return false;
    }
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *value = (uint64_t)parsed;
    return true;
}

static uint16_t read_le16(const unsigned char *bytes)
{
    return (uint16_t)((uint16_t)bytes[0] |
                      ((uint16_t)bytes[1] << 8U));
}

static uint32_t read_le32(const unsigned char *bytes)
{
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8U) |
           ((uint32_t)bytes[2] << 16U) | ((uint32_t)bytes[3] << 24U);
}

static uint64_t fnv1a(const void *data, size_t size)
{
    const unsigned char *bytes = data;
    uint64_t hash = UINT64_C(14695981039346656037);
    size_t index;

    for (index = 0U; index < size; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static bool validate_samples(const float *samples, size_t count,
                             float *minimum, float *maximum,
                             double *energy)
{
    size_t index;

    if (count == 0U) {
        return false;
    }
    *minimum = samples[0];
    *maximum = samples[0];
    *energy = 0.0;
    for (index = 0U; index < count; ++index) {
        float value = samples[index];

        if (!isfinite(value) || value < -1.0F || value > 1.0F) {
            return false;
        }
        if (value < *minimum) {
            *minimum = value;
        }
        if (value > *maximum) {
            *maximum = value;
        }
        *energy += (double)value * (double)value;
    }
    return true;
}

static bool validate_wave(const char *path, const float *left,
                          const float *right, size_t count)
{
    size_t expected_size;
    unsigned char *bytes = NULL;
    FILE *file = NULL;
    long length;
    size_t index;
    bool valid = false;

    if (count > (SIZE_MAX - 44U) / (2U * sizeof(float))) {
        return false;
    }
    expected_size = 44U + 2U * count * sizeof(float);
    if (expected_size > UINT32_MAX) {
        return false;
    }
    file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0L, SEEK_END) != 0) {
        goto done;
    }
    length = ftell(file);
    if (length < 0L || (size_t)length != expected_size ||
        fseek(file, 0L, SEEK_SET) != 0) {
        goto done;
    }
    bytes = malloc(expected_size);
    if (bytes == NULL ||
        fread(bytes, 1U, expected_size, file) != expected_size) {
        goto done;
    }
    valid = memcmp(bytes, "RIFF", 4U) == 0 &&
            read_le32(bytes + 4U) == (uint32_t)(expected_size - 8U) &&
            memcmp(bytes + 8U, "WAVEfmt ", 8U) == 0 &&
            read_le32(bytes + 16U) == 16U && read_le16(bytes + 20U) == 3U &&
            read_le16(bytes + 22U) == 2U &&
            read_le32(bytes + 24U) == 44100U &&
            read_le32(bytes + 28U) == 352800U &&
            read_le16(bytes + 32U) == 8U && read_le16(bytes + 34U) == 32U &&
            memcmp(bytes + 36U, "data", 4U) == 0 &&
            read_le32(bytes + 40U) ==
                (uint32_t)(2U * count * sizeof(float));
    for (index = 0U; valid && index < count; ++index) {
        uint32_t left_bits;
        uint32_t right_bits;
        size_t offset = 44U + index * 2U * sizeof(float);

        (void)memcpy(&left_bits, &left[index], sizeof(left_bits));
        (void)memcpy(&right_bits, &right[index], sizeof(right_bits));
        valid = read_le32(bytes + offset) == left_bits &&
                read_le32(bytes + offset + sizeof(float)) == right_bits;
    }

done:
    free(bytes);
    if (file != NULL) {
        (void)fclose(file);
    }
    return valid;
}

static void print_usage(const char *program)
{
    (void)fprintf(stderr,
                  "usage: %s MODEL_ROOT WAVE_PATH [MAX_FRAMES [SEED "
                  "[SEQUENCE [CAPTION [LYRICS]]]]]\n",
                  program);
}

int main(int argc, char **argv)
{
    static const char default_caption[] = "A gentle piano melody.";
    static const char default_lyrics[] = "[Verse]\nHello world.";
    const char *model_root;
    const char *wave_path;
    const char *caption = default_caption;
    const char *lyrics = default_lyrics;
    m3_music3_request request;
    m3_music3_output_info info = {0};
    m3_music3_engine *engine = NULL;
    m3_music3_output *output = NULL;
    progress_state progress = {.phase = -1};
    float *left = NULL;
    float *right = NULL;
    float left_min = 0.0F;
    float left_max = 0.0F;
    float right_min = 0.0F;
    float right_max = 0.0F;
    double left_energy = 0.0;
    double right_energy = 0.0;
    size_t sample_count = 0U;
    size_t byte_count = 0U;
    m3_error error;
    m3_status status = M3_STATUS_INVALID_ARGUMENT;
    double open_start;
    double open_end;
    double generate_start;
    double generate_end;
    double validation_end;
    int phase_index;
    int result = 1;

    if (argc < 3 || argc > 8) {
        print_usage(argv[0]);
        return 2;
    }
    model_root = argv[1];
    wave_path = argv[2];
    request.maximum_frames = 1U;
    request.seed = UINT64_C(20260815);
    request.sequence = 0U;
    if ((argc >= 4 && !parse_u64(argv[3], &request.maximum_frames)) ||
        (argc >= 5 && !parse_u64(argv[4], &request.seed)) ||
        (argc >= 6 && !parse_u64(argv[5], &request.sequence))) {
        print_usage(argv[0]);
        return 2;
    }
    if (argc >= 7) {
        caption = argv[6];
    }
    if (argc >= 8) {
        lyrics = argv[7];
    }
    progress.maximum_frames = request.maximum_frames;
    request.caption.data = caption;
    request.caption.length = strlen(caption);
    request.lyrics.data = lyrics;
    request.lyrics.length = strlen(lyrics);
    (void)setvbuf(stderr, NULL, _IOLBF, 0U);
    if (clock_gettime(CLOCK_MONOTONIC, &benchmark_epoch) != 0 ||
        signal(SIGINT, request_stop) == SIG_ERR ||
        signal(SIGTERM, request_stop) == SIG_ERR) {
        (void)fprintf(stderr, "BENCHMARK_SETUP_FAILED\n");
        goto done;
    }
    m3_error_reset(&error);
    (void)fprintf(stderr,
                  "OPEN root=%s frames=%llu seed=%llu sequence=%llu\n",
                  model_root,
                  (unsigned long long)request.maximum_frames,
                  (unsigned long long)request.seed,
                  (unsigned long long)request.sequence);
    open_start = elapsed_seconds();
    status = m3_music3_engine_open(&engine, model_root, &error);
    open_end = elapsed_seconds();
    if (status != M3_STATUS_OK) {
        (void)fprintf(stderr, "OPEN_FAILED status=%d message=%s\n",
                      (int)status, m3_error_message(&error));
        goto done;
    }
    generate_start = elapsed_seconds();
    status = m3_music3_generate(engine, &request, report_progress, &progress,
                                &output, &error);
    generate_end = elapsed_seconds();
    if (status != M3_STATUS_OK || output == NULL || progress.invalid ||
        progress.phase != (int)M3_MUSIC3_PHASE_ASSEMBLE ||
        progress.completed != progress.total) {
        (void)fprintf(stderr,
                      "GENERATE_FAILED status=%d message=%s phase=%d "
                      "completed=%llu signal=%d\n",
                      (int)status, m3_error_message(&error), progress.phase,
                      (unsigned long long)progress.completed,
                      (int)stop_requested);
        goto done;
    }
    status = m3_music3_output_get_info(output, &info, &error);
    if (status != M3_STATUS_OK || info.sample_rate != 44100U ||
        info.channel_count != 2U || info.samples_per_channel == 0U ||
        info.samples_per_channel > SIZE_MAX) {
        (void)fprintf(stderr, "INFO_FAILED status=%d message=%s\n",
                      (int)status, m3_error_message(&error));
        goto done;
    }
    sample_count = (size_t)info.samples_per_channel;
    if (sample_count > SIZE_MAX / sizeof(float)) {
        (void)fprintf(stderr, "OUTPUT_SIZE_OVERFLOW\n");
        goto done;
    }
    byte_count = sample_count * sizeof(float);
    left = malloc(byte_count);
    right = malloc(byte_count);
    if (left == NULL || right == NULL) {
        (void)fprintf(stderr, "OUTPUT_ALLOCATION_FAILED\n");
        goto done;
    }
    status = m3_music3_output_read_channel_f32(
        output, 0U, 0U, left, sample_count, &error);
    if (status == M3_STATUS_OK) {
        status = m3_music3_output_read_channel_f32(
            output, 1U, 0U, right, sample_count, &error);
    }
    if (status != M3_STATUS_OK ||
        !validate_samples(left, sample_count, &left_min, &left_max,
                          &left_energy) ||
        !validate_samples(right, sample_count, &right_min, &right_max,
                          &right_energy) ||
        left_energy + right_energy == 0.0) {
        (void)fprintf(stderr, "SAMPLES_FAILED status=%d message=%s\n",
                      (int)status, m3_error_message(&error));
        goto done;
    }
    status = m3_music3_output_write_wav(output, wave_path, &error);
    if (status != M3_STATUS_OK ||
        !validate_wave(wave_path, left, right, sample_count)) {
        (void)fprintf(stderr, "WAVE_FAILED status=%d message=%s\n",
                      (int)status, m3_error_message(&error));
        goto done;
    }
    validation_end = elapsed_seconds();
    (void)fprintf(stderr,
                  "BENCHMARK open=%.6f generate=%.6f "
                  "output_validation=%.6f total=%.6f\n",
                  open_end - open_start, generate_end - generate_start,
                  validation_end - generate_end, validation_end);
    for (phase_index = 0; phase_index < (int)M3_MUSIC3_PHASE_COUNT;
         ++phase_index) {
        if (progress.phase_seen[phase_index]) {
            (void)fprintf(stderr,
                          "PHASE_TIME phase=%s start=%.6f end=%.6f "
                          "duration=%.6f\n",
                          phase_names[phase_index],
                          progress.phase_start[phase_index],
                          progress.phase_end[phase_index],
                          progress.phase_end[phase_index] -
                              progress.phase_start[phase_index]);
        }
    }
    (void)fprintf(stderr,
                  "E2E_OK samples=%zu left_min=%.9g left_max=%.9g "
                  "left_energy=%.17g left_fnv=%016llx right_min=%.9g "
                  "right_max=%.9g right_energy=%.17g right_fnv=%016llx\n",
                  sample_count, (double)left_min, (double)left_max,
                  left_energy, (unsigned long long)fnv1a(left, byte_count),
                  (double)right_min, (double)right_max, right_energy,
                  (unsigned long long)fnv1a(right, byte_count));
    result = 0;

done:
    free(right);
    free(left);
    m3_music3_output_free(output);
    m3_music3_engine_free(engine);
    (void)fprintf(stderr, "CLEANUP result=%d\n", result);
    return result;
}

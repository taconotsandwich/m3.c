/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_VOCODER_INTERNAL_H
#define M3_VOCODER_INTERNAL_H

#include "m3_runtime_workspace.h"
#include "m3_vocoder_runtime.h"

#include <stddef.h>
#include <stdint.h>

#define M3_VOCODER_SOURCE_WEIGHT_COUNT 121U
#define M3_VOCODER_RUNTIME_WEIGHT_COUNT 91U
#define M3_VOCODER_BLOCK_COUNT 4U
#define M3_VOCODER_RESIDUAL_COUNT 3U
#define M3_VOCODER_DECODE_OPERATION_COUNT 73U
#define M3_VOCODER_MAXIMUM_LATENT_LENGTH 689U
#define M3_VOCODER_DECODE_BUFFER_COUNT 3U
#define M3_VOCODER_MAXIMUM_ROW_BYTES 49152U
#define M3_VOCODER_NAME_CAPACITY 128U

typedef enum {
    M3_VOCODER_MATERIAL_COPY = 0,
    M3_VOCODER_MATERIAL_WEIGHT_NORM
} m3_vocoder_material_kind;

typedef struct {
    uint64_t latent_channels;
    uint64_t maximum_latent_length;
    uint64_t decoder_input_channels;
    uint64_t decoder_output_channels;
    uint64_t initial_channels;
    size_t block_count;
    size_t residual_count;
    uint64_t strides[M3_VOCODER_BLOCK_COUNT];
} m3_vocoder_plan_config;

typedef struct {
    m3_vocoder_material_kind kind;
    char output_name[M3_VOCODER_NAME_CAPACITY];
    char source_names[2U][M3_VOCODER_NAME_CAPACITY];
    m3_tensor_metadata source_metadata[2U];
    m3_tensor_metadata output_metadata;
} m3_vocoder_plan_entry;

typedef struct {
    m3_vocoder_plan_entry *entries;
    size_t entry_count;
    size_t source_count;
    size_t block_count;
    size_t residual_count;
    m3_vocoder_plan_config config;
} m3_vocoder_plan;

typedef struct {
    size_t buffer_bytes[M3_VOCODER_DECODE_BUFFER_COUNT];
    size_t workspace_bytes;
    size_t output_bytes;
    uint64_t output_length;
} m3_vocoder_decode_measurement;

typedef struct {
    const m3_tensor_view *snake1_alpha;
    const m3_tensor_view *conv1_weight;
    const m3_tensor_view *conv1_bias;
    const m3_tensor_view *snake2_alpha;
    const m3_tensor_view *conv2_weight;
    const m3_tensor_view *conv2_bias;
} m3_vocoder_residual_weights;

typedef struct {
    const m3_tensor_view *snake_alpha;
    const m3_tensor_view *transpose_weight;
    const m3_tensor_view *transpose_bias;
    m3_vocoder_residual_weights residuals[M3_VOCODER_RESIDUAL_COUNT];
} m3_vocoder_block_weights;

typedef struct {
    const m3_tensor_view *decoder_input_weight;
    const m3_tensor_view *decoder_input_bias;
    const m3_tensor_view *convolution_input_weight;
    const m3_tensor_view *convolution_input_bias;
    m3_vocoder_block_weights blocks[M3_VOCODER_BLOCK_COUNT];
    const m3_tensor_view *snake_output_alpha;
    const m3_tensor_view *convolution_output_weight;
    const m3_tensor_view *convolution_output_bias;
    size_t block_count;
    size_t residual_count;
} m3_vocoder_weights;

typedef m3_status (*m3_vocoder_storage_read_fn)(
    void *context, const m3_storage *storage, size_t byte_offset,
    void *destination, size_t byte_count, m3_error *error);
typedef m3_status (*m3_vocoder_storage_write_fn)(
    void *context, m3_storage *storage, size_t byte_offset,
    const void *source, size_t byte_count, m3_error *error);

typedef struct {
    void *context;
    m3_vocoder_storage_read_fn read_storage;
    m3_vocoder_storage_write_fn write_storage;
} m3_vocoder_materialize_io;

struct m3_vocoder_runtime {
    m3_backend *backend;
    m3_runtime_workspace weights;
    m3_vocoder_weights bound;
    m3_vocoder_plan_config config;
};

void m3_vocoder_plan_init(m3_vocoder_plan *plan);
void m3_vocoder_plan_dispose(m3_vocoder_plan *plan);
void m3_vocoder_plan_official_config(m3_vocoder_plan_config *config);
m3_status m3_vocoder_plan_config_validate(
    const m3_vocoder_plan_config *config, m3_error *error);
m3_status m3_vocoder_plan_build(
    const m3_vocoder_plan_config *config, m3_vocoder_plan *plan,
    m3_error *error);

void m3_vocoder_materialize_io_init(m3_vocoder_materialize_io *io);
m3_status m3_vocoder_runtime_create_core(
    m3_vocoder_runtime **runtime, const m3_weight_stage *vocoder,
    const m3_vocoder_plan *plan, const m3_vocoder_materialize_io *io,
    m3_progress_callback progress, void *progress_context,
    m3_error *error);

m3_status m3_vocoder_runtime_bind(
    m3_vocoder_runtime *runtime, const m3_vocoder_plan *plan,
    m3_error *error);
const m3_vocoder_weights *m3_vocoder_runtime_weights(
    const m3_vocoder_runtime *runtime);
m3_status m3_vocoder_decode_measure(
    const m3_vocoder_plan_config *config, uint64_t latent_length,
    m3_vocoder_decode_measurement *measurement, m3_error *error);
m3_status m3_vocoder_decode_prepare(
    m3_vocoder_runtime *runtime, const m3_tensor_view *latents,
    m3_vocoder_output *output,
    m3_vocoder_decode_measurement *measurement, m3_error *error);

#endif

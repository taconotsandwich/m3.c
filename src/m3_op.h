/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_OP_H
#define M3_OP_H

#include "m3_backend.h"
#include "m3_scratch.h"
#include "m3_tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    M3_OP_COPY = 0,
    M3_OP_CAST,
    M3_OP_ADD,
    M3_OP_MUL,
    M3_OP_EMBEDDING,
    M3_OP_MATMUL,
    M3_OP_LINEAR,
    M3_OP_RMS_NORM,
    M3_OP_LAYER_NORM,
    M3_OP_ROPE,
    M3_OP_ATTENTION,
    M3_OP_GATED_SILU,
    M3_OP_SOFTMAX,
    M3_OP_TOP_K,
    M3_OP_CATEGORICAL,
    M3_OP_CONV1D,
    M3_OP_CONV_TRANSPOSE1D,
    M3_OP_NEAREST_RESIZE1D
} m3_op_kind;

typedef enum {
    M3_ROPE_HALF_SPLIT = 0,
    M3_ROPE_INTERLEAVED
} m3_rope_mode;

typedef struct {
    const m3_tensor_view *input;
    m3_tensor_view *output;
} m3_op_unary;

typedef struct {
    const m3_tensor_view *left;
    const m3_tensor_view *right;
    m3_tensor_view *output;
} m3_op_binary;

typedef struct {
    const m3_tensor_view *ids;
    const m3_tensor_view *table;
    m3_tensor_view *output;
} m3_op_embedding;

typedef struct {
    const m3_tensor_view *left;
    const m3_tensor_view *right;
    m3_tensor_view *output;
} m3_op_matmul;

typedef struct {
    const m3_tensor_view *input;
    const m3_tensor_view *weight;
    const m3_tensor_view *bias;
    m3_tensor_view *output;
} m3_op_linear;

typedef struct {
    const m3_tensor_view *input;
    const m3_tensor_view *scale;
    m3_tensor_view *output;
    float epsilon;
} m3_op_rms_norm;

typedef struct {
    const m3_tensor_view *input;
    const m3_tensor_view *scale;
    const m3_tensor_view *bias;
    m3_tensor_view *output;
    float epsilon;
} m3_op_layer_norm;

typedef struct {
    const m3_tensor_view *input;
    const m3_tensor_view *cosines;
    const m3_tensor_view *sines;
    m3_tensor_view *output;
    uint64_t position_offset;
    uint32_t rotary_dimension;
    m3_rope_mode mode;
} m3_op_rope;

typedef struct {
    const m3_tensor_view *query;
    const m3_tensor_view *key;
    const m3_tensor_view *value;
    const m3_tensor_view *mask;
    m3_tensor_view *output;
    float scale;
    int64_t causal_offset;
    bool causal;
} m3_op_attention;

typedef struct {
    const m3_tensor_view *logits;
    m3_tensor_view *values;
    m3_tensor_view *indices;
    uint64_t k;
} m3_op_top_k;

typedef struct {
    const m3_tensor_view *probabilities;
    const m3_tensor_view *uniforms;
    m3_tensor_view *output;
} m3_op_categorical;

typedef struct {
    const m3_tensor_view *input;
    const m3_tensor_view *weight;
    const m3_tensor_view *bias;
    m3_tensor_view *output;
    uint64_t groups;
    uint64_t stride;
    uint64_t dilation;
    uint64_t pad_left;
    uint64_t pad_right;
} m3_op_conv1d;

typedef struct {
    const m3_tensor_view *input;
    const m3_tensor_view *weight;
    const m3_tensor_view *bias;
    m3_tensor_view *output;
    uint64_t groups;
    uint64_t stride;
    uint64_t dilation;
    uint64_t pad_left;
    uint64_t pad_right;
    uint64_t output_padding;
} m3_op_conv_transpose1d;

typedef struct {
    m3_op_kind kind;
    union {
        m3_op_unary copy;
        m3_op_unary cast;
        m3_op_binary add;
        m3_op_binary mul;
        m3_op_embedding embedding;
        m3_op_matmul matmul;
        m3_op_linear linear;
        m3_op_rms_norm rms_norm;
        m3_op_layer_norm layer_norm;
        m3_op_rope rope;
        m3_op_attention attention;
        m3_op_binary gated_silu;
        m3_op_unary softmax;
        m3_op_top_k top_k;
        m3_op_categorical categorical;
        m3_op_conv1d conv1d;
        m3_op_conv_transpose1d conv_transpose1d;
        m3_op_unary nearest_resize1d;
    } descriptor;
} m3_command;

/* Commands borrow their views. Inputs may be strided; outputs are contiguous.
 * Host softmax and attention map an all -INFINITY row to zeros and divide
 * probability equally among only the +INFINITY maxima when any are present. */

m3_status m3_command_validate(const m3_backend *backend,
                              const m3_command *command,
                              m3_error *error);
m3_status m3_commands_scratch_bytes(const m3_backend *backend,
                                    const m3_command *commands,
                                    size_t command_count,
                                    size_t *scratch_bytes,
                                    m3_error *error);
m3_status m3_backend_execute(m3_backend *backend,
                             const m3_command *commands,
                             size_t command_count,
                             m3_scratch_arena *scratch,
                             m3_error *error);

#endif

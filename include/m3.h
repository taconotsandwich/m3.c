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

#define M3_TENSOR_MAX_RANK 8U
#define M3_GRAPH_NODE_INPUT_CAPACITY 5U
#define M3_GRAPH_NODE_OUTPUT_CAPACITY 2U
#define M3_GRAPH_VALUE_NONE UINT32_MAX

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

typedef enum {
    M3_DTYPE_F32 = 0,
    M3_DTYPE_F16,
    M3_DTYPE_BF16,
    M3_DTYPE_I32
} m3_dtype;

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
    M3_OP_NEAREST_RESIZE1D,
    M3_OP_SNAKE1D,
    M3_OP_TANH
} m3_op_kind;

typedef enum {
    M3_ROPE_HALF_SPLIT = 0,
    M3_ROPE_INTERLEAVED
} m3_rope_mode;

typedef enum {
    M3_PROVIDER_NEURAL_ENGINE = 0,
    M3_PROVIDER_METAL,
    M3_PROVIDER_CPU
} m3_provider_kind;

typedef enum {
    M3_GRAPH_INPUT = 0,
    M3_GRAPH_CONSTANT,
    M3_GRAPH_TEMPORARY,
    M3_GRAPH_OUTPUT
} m3_graph_value_role;

typedef uint32_t m3_graph_value_id;

typedef struct {
    m3_dtype dtype;
    uint8_t rank;
    uint64_t shape[M3_TENSOR_MAX_RANK];
    m3_graph_value_role role;
} m3_graph_value_desc;

typedef struct {
    m3_op_kind kind;
    m3_graph_value_id inputs[M3_GRAPH_NODE_INPUT_CAPACITY];
    m3_graph_value_id outputs[M3_GRAPH_NODE_OUTPUT_CAPACITY];
    union {
        struct {
            float epsilon;
        } norm;
        struct {
            uint64_t position_offset;
            uint32_t rotary_dimension;
            m3_rope_mode mode;
        } rope;
        struct {
            float scale;
            int64_t causal_offset;
            bool causal;
        } attention;
        struct {
            uint64_t k;
        } top_k;
        struct {
            uint64_t groups;
            uint64_t stride;
            uint64_t dilation;
            uint64_t pad_left;
            uint64_t pad_right;
            uint64_t output_padding;
        } convolution;
        struct {
            uint64_t output_length;
        } resize;
    } parameters;
} m3_graph_node_desc;

typedef struct {
    const char *compiled_model_path;
    const m3_graph_value_id *inputs;
    const char *const *input_names;
    size_t input_count;
    const m3_graph_value_id *outputs;
    const char *const *output_names;
    size_t output_count;
} m3_coreml_partition_desc;

typedef struct {
    bool allow_neural_engine;
    bool allow_metal;
    bool allow_cpu;
} m3_session_options;

typedef struct {
    size_t value_count;
    size_t node_count;
    size_t storage_slot_count;
    size_t allocated_bytes;
} m3_session_plan_info;

typedef struct m3_graph m3_graph;
typedef struct m3_session m3_session;

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

void m3_graph_value_desc_init(m3_graph_value_desc *description,
                              m3_graph_value_role role, m3_dtype dtype,
                              uint8_t rank, const uint64_t *shape);
void m3_graph_node_desc_init(m3_graph_node_desc *description,
                             m3_op_kind kind);
void m3_session_options_init(m3_session_options *options);

m3_status m3_graph_create(m3_graph **graph, m3_error *error);
void m3_graph_free(m3_graph *graph);
m3_status m3_graph_add_value(m3_graph *graph,
                             const m3_graph_value_desc *description,
                             m3_graph_value_id *value, m3_error *error);
m3_status m3_graph_set_constant(m3_graph *graph,
                                m3_graph_value_id value,
                                const void *data, size_t byte_count,
                                m3_error *error);
m3_status m3_graph_add_inferred_value(
    m3_graph *graph, const m3_graph_node_desc *node,
    size_t output_index, m3_graph_value_role role, m3_dtype dtype,
    m3_graph_value_id *value, m3_error *error);

/* Regular nodes use the fixed input/output positions of their operation.
 * Optional bias and mask inputs use M3_GRAPH_VALUE_NONE. The graph copies the
 * descriptor and validates the complete topological data flow at session
 * creation. */
m3_status m3_graph_add_node(m3_graph *graph,
                            const m3_graph_node_desc *description,
                            m3_error *error);

/* Core ML partitions are precompiled model subgraphs. Feature names, value
 * identifiers, and the path are copied by the graph. Core ML is configured
 * for CPU and Neural Engine only; regular graph nodes use Metal, then CPU. */
m3_status m3_graph_add_coreml_partition(
    m3_graph *graph, const m3_coreml_partition_desc *description,
    m3_error *error);

/* A session owns its selected storage backend and compiled execution plan.
 * The graph can be freed after successful session creation. */
m3_status m3_session_create(m3_session **session, const m3_graph *graph,
                             const m3_session_options *options,
                             m3_error *error);
void m3_session_free(m3_session *session);
m3_status m3_session_write_input(m3_session *session,
                                  m3_graph_value_id value,
                                  const void *data, size_t byte_count,
                                  m3_error *error);
m3_status m3_session_read_output(const m3_session *session,
                                  m3_graph_value_id value,
                                  void *data, size_t byte_count,
                                  m3_error *error);
m3_status m3_session_run(m3_session *session, m3_error *error);
m3_status m3_session_get_plan_info(const m3_session *session,
                                    m3_session_plan_info *info,
                                    m3_error *error);
m3_status m3_session_get_node_provider(const m3_session *session,
                                        size_t node_index,
                                        m3_provider_kind *provider,
                                        m3_error *error);

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

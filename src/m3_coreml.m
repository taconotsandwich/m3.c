/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif

#include "m3_graph_internal.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <stdlib.h>
#include <string.h>

struct m3_coreml_partition {
    CFTypeRef model;
    m3_graph_value_id *inputs;
    CFTypeRef *input_names;
    size_t input_count;
    m3_graph_value_id *outputs;
    CFTypeRef *output_names;
    size_t output_count;
};

static m3_status m3_coreml_error(m3_error *error, m3_status status,
                                  const char *operation,
                                  NSError *foundation_error)
{
    const char *message = foundation_error == nil
        ? "unknown Core ML error"
        : foundation_error.localizedDescription.UTF8String;

    return m3_error_set(error, status, "%s: %s", operation,
                        message == NULL ? "unavailable diagnostic" : message);
}

static bool m3_coreml_dtype(m3_dtype dtype,
                            MLMultiArrayDataType *coreml_dtype)
{
    switch (dtype) {
    case M3_DTYPE_F32:
        *coreml_dtype = MLMultiArrayDataTypeFloat32;
        return true;
    case M3_DTYPE_F16:
        *coreml_dtype = MLMultiArrayDataTypeFloat16;
        return true;
    case M3_DTYPE_I32:
        *coreml_dtype = MLMultiArrayDataTypeInt32;
        return true;
    case M3_DTYPE_BF16:
        return false;
    }
    return false;
}

static NSArray<NSNumber *> *m3_coreml_shape(
    const m3_tensor_view *view)
{
    NSMutableArray<NSNumber *> *shape =
        [NSMutableArray arrayWithCapacity:view->metadata.rank];
    uint8_t axis;

    for (axis = 0U; axis < view->metadata.rank; ++axis) {
        [shape addObject:@(view->metadata.shape[axis])];
    }
    return shape;
}

static NSArray<NSNumber *> *m3_coreml_strides(
    const m3_tensor_view *view)
{
    NSMutableArray<NSNumber *> *strides =
        [NSMutableArray arrayWithCapacity:view->metadata.rank];
    size_t element_size = m3_dtype_size(view->metadata.dtype);
    uint8_t axis;

    for (axis = 0U; axis < view->metadata.rank; ++axis) {
        [strides addObject:@(view->byte_strides[axis] / element_size)];
    }
    return strides;
}

static m3_status m3_coreml_array(m3_tensor_view *view,
                                  MLMultiArray **array_output,
                                  m3_error *error)
{
    MLMultiArrayDataType data_type;
    NSError *foundation_error = nil;
    uint8_t *data;
    MLMultiArray *array;

    *array_output = nil;
    if (!m3_coreml_dtype(view->metadata.dtype, &data_type)) {
        return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                            "Core ML does not support %s tensors",
                            m3_dtype_name(view->metadata.dtype));
    }
    data = m3_storage_data(view->storage);
    if (data == NULL) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Core ML tensor storage is not host visible");
    }
    array = [[MLMultiArray alloc]
        initWithDataPointer:data + view->byte_offset
                      shape:m3_coreml_shape(view)
                   dataType:data_type
                    strides:m3_coreml_strides(view)
                deallocator:nil
                      error:&foundation_error];
    if (array == nil) {
        return m3_coreml_error(error, M3_STATUS_INVALID_ARGUMENT,
                               "cannot bind Core ML tensor",
                               foundation_error);
    }
    *array_output = array;
    return M3_STATUS_OK;
}

static bool m3_coreml_description_matches(
    MLFeatureDescription *feature, const m3_tensor_view *view)
{
    MLMultiArrayDataType dtype;
    MLMultiArrayConstraint *constraint;
    NSArray<NSNumber *> *shape;

    if (!m3_coreml_dtype(view->metadata.dtype, &dtype) ||
        feature.type != MLFeatureTypeMultiArray) {
        return false;
    }
    constraint = feature.multiArrayConstraint;
    shape = m3_coreml_shape(view);
    return constraint != nil && constraint.dataType == dtype &&
           [constraint.shape isEqualToArray:shape];
}

static m3_status m3_coreml_copy_bindings(
    const m3_graph_value_id *values, char *const *names, size_t count,
    m3_graph_value_id **copied_values, CFTypeRef **copied_names,
    m3_error *error)
{
    size_t index;

    *copied_values = calloc(count, sizeof(**copied_values));
    *copied_names = calloc(count, sizeof(**copied_names));
    if (*copied_values == NULL || *copied_names == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate Core ML partition bindings");
    }
    for (index = 0U; index < count; ++index) {
        NSString *name = [NSString stringWithUTF8String:names[index]];

        if (name == nil) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "Core ML feature name is not UTF-8");
        }
        (*copied_values)[index] = values[index];
        (*copied_names)[index] = CFBridgingRetain(name);
    }
    return M3_STATUS_OK;
}

static void m3_coreml_release_names(CFTypeRef *names, size_t count)
{
    size_t index;

    if (names == NULL) {
        return;
    }
    for (index = 0U; index < count; ++index) {
        if (names[index] != NULL) {
            CFRelease(names[index]);
        }
    }
    free(names);
}

void m3_coreml_partition_free(m3_coreml_partition *partition)
{
    if (partition == NULL) {
        return;
    }
    if (partition->model != NULL) {
        CFRelease(partition->model);
    }
    m3_coreml_release_names(partition->input_names,
                            partition->input_count);
    m3_coreml_release_names(partition->output_names,
                            partition->output_count);
    free(partition->inputs);
    free(partition->outputs);
    free(partition);
}

static m3_status m3_coreml_load(const char *path, MLModel **model,
                                m3_error *error)
{
    NSString *model_path = [NSString stringWithUTF8String:path];
    NSURL *source;
    NSURL *compiled;
    NSError *foundation_error = nil;
    MLModelConfiguration *configuration;
    bool temporary = false;

    if (model_path == nil) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Core ML model path is not UTF-8");
    }
    source = [NSURL fileURLWithPath:model_path];
    compiled = source;
    if (![source.pathExtension isEqualToString:@"mlmodelc"]) {
        compiled = [MLModel compileModelAtURL:source
                                        error:&foundation_error];
        temporary = true;
        if (compiled == nil) {
            return m3_coreml_error(error, M3_STATUS_INVALID_FORMAT,
                                   "cannot compile Core ML model",
                                   foundation_error);
        }
    }
    configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    *model = [MLModel modelWithContentsOfURL:compiled
                               configuration:configuration
                                       error:&foundation_error];
    if (temporary) {
        (void)[[NSFileManager defaultManager]
            removeItemAtURL:compiled error:nil];
    }
    if (*model == nil) {
        return m3_coreml_error(error, M3_STATUS_INVALID_FORMAT,
                               "cannot load Core ML model",
                               foundation_error);
    }
    return M3_STATUS_OK;
}

static m3_status m3_coreml_validate_side(
    NSDictionary<NSString *, MLFeatureDescription *> *features,
    const m3_graph_value_id *values, CFTypeRef *names, size_t count,
    const m3_tensor_view *views, size_t value_count, const char *side,
    m3_error *error)
{
    size_t index;

    for (index = 0U; index < count; ++index) {
        NSString *name;
        MLFeatureDescription *feature;
        size_t other;

        if (names[index] == NULL) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "Core ML %s feature name is absent", side);
        }
        name = (__bridge NSString *)names[index];
        feature = features[name];

        if ((size_t)values[index] >= value_count || feature == nil ||
            !m3_coreml_description_matches(feature, &views[values[index]])) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "Core ML %s feature %s does not match graph",
                                side, name.UTF8String);
        }
        for (other = 0U; other < index; ++other) {
            if ([name isEqualToString:(__bridge NSString *)names[other]]) {
                return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                    "Core ML %s feature name is repeated",
                                    side);
            }
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_coreml_partition_create(
    const m3_graph_coreml *description, const m3_tensor_view *views,
    size_t value_count, m3_coreml_partition **partition_output,
    m3_error *error)
{
    m3_coreml_partition *partition;
    MLModel *model = nil;
    m3_status status;

    if (partition_output == NULL || *partition_output != NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Core ML partition output must be empty");
    }
    partition = calloc(1U, sizeof(*partition));
    if (partition == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate Core ML partition");
    }
    partition->input_count = description->input_count;
    partition->output_count = description->output_count;
    status = m3_coreml_copy_bindings(
        description->inputs, description->input_names,
        description->input_count, &partition->inputs,
        &partition->input_names, error);
    if (status == M3_STATUS_OK) {
        status = m3_coreml_copy_bindings(
            description->outputs, description->output_names,
            description->output_count, &partition->outputs,
            &partition->output_names, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_coreml_load(description->compiled_model_path, &model,
                                error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_coreml_validate_side(
            model.modelDescription.inputDescriptionsByName,
            partition->inputs, partition->input_names,
            partition->input_count, views, value_count, "input", error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_coreml_validate_side(
            model.modelDescription.outputDescriptionsByName,
            partition->outputs, partition->output_names,
            partition->output_count, views, value_count, "output", error);
    }
    if (status == M3_STATUS_OK) {
        partition->model = CFBridgingRetain(model);
        *partition_output = partition;
        m3_error_reset(error);
    } else {
        m3_coreml_partition_free(partition);
    }
    return status;
}

static m3_status m3_coreml_copy_output(MLMultiArray *source,
                                        MLMultiArray *backing,
                                        m3_tensor_view *destination,
                                        m3_error *error)
{
    __block m3_status copy_status = M3_STATUS_OK;
    size_t element_size = m3_dtype_size(destination->metadata.dtype);
    uint8_t *output = m3_storage_data(destination->storage);

    if (source == backing) {
        return M3_STATUS_OK;
    }
    if (output == NULL && destination->metadata.byte_count != 0U) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Core ML output storage is unavailable");
    }
    [source getBytesWithHandler:^(const void *bytes, NSInteger size) {
        size_t flat;

        if (bytes == NULL || size < 0) {
            copy_status = m3_error_set(
                error, M3_STATUS_INTERNAL,
                "Core ML output storage is unavailable");
            return;
        }
        for (flat = 0U; flat < destination->metadata.element_count; ++flat) {
            size_t remaining = flat;
            size_t source_offset = 0U;
            uint8_t axis = destination->metadata.rank;

            while (axis != 0U) {
                size_t coordinate;

                --axis;
                coordinate = remaining %
                    (size_t)destination->metadata.shape[axis];
                remaining /= (size_t)destination->metadata.shape[axis];
                source_offset += coordinate *
                    (size_t)source.strides[axis].unsignedLongLongValue;
            }
            if (source_offset > SIZE_MAX / element_size ||
                source_offset * element_size > (size_t)size ||
                element_size >
                    (size_t)size - source_offset * element_size) {
                copy_status = m3_error_set(
                    error, M3_STATUS_INTERNAL,
                    "Core ML output range is invalid");
                return;
            }
            (void)memcpy(output + destination->byte_offset +
                             flat * element_size,
                         (const uint8_t *)bytes +
                             source_offset * element_size,
                         element_size);
        }
    }];
    return copy_status;
}

m3_status m3_coreml_partition_execute(
    m3_coreml_partition *partition, m3_tensor_view *views,
    size_t value_count, m3_error *error)
{
    MLModel *model;
    NSMutableDictionary<NSString *, id> *inputs;
    NSMutableDictionary<NSString *, id> *backings;
    NSMutableArray<MLMultiArray *> *output_arrays;
    MLPredictionOptions *options;
    MLDictionaryFeatureProvider *provider;
    id<MLFeatureProvider> prediction;
    NSError *foundation_error = nil;
    size_t index;

    if (partition == NULL || views == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Core ML execution state is incomplete");
    }
    model = (__bridge MLModel *)partition->model;
    inputs = [NSMutableDictionary dictionaryWithCapacity:partition->input_count];
    backings = [NSMutableDictionary dictionaryWithCapacity:partition->output_count];
    output_arrays = [NSMutableArray arrayWithCapacity:partition->output_count];
    for (index = 0U; index < partition->input_count; ++index) {
        m3_graph_value_id value = partition->inputs[index];
        MLMultiArray *array;
        m3_status status;

        if ((size_t)value >= value_count) {
            return m3_error_set(error, M3_STATUS_INTERNAL,
                                "Core ML input binding is out of range");
        }
        status = m3_coreml_array(&views[value], &array, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        if (array == nil) {
            return m3_error_set(error, M3_STATUS_INTERNAL,
                                "Core ML input array was not created");
        }
        inputs[(__bridge NSString *)partition->input_names[index]] =
            [MLFeatureValue featureValueWithMultiArray:array];
    }
    for (index = 0U; index < partition->output_count; ++index) {
        m3_graph_value_id value = partition->outputs[index];
        MLMultiArray *array;
        m3_status status;

        if ((size_t)value >= value_count) {
            return m3_error_set(error, M3_STATUS_INTERNAL,
                                "Core ML output binding is out of range");
        }
        status = m3_coreml_array(&views[value], &array, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        if (array == nil) {
            return m3_error_set(error, M3_STATUS_INTERNAL,
                                "Core ML output array was not created");
        }
        [output_arrays addObject:array];
        backings[(__bridge NSString *)partition->output_names[index]] = array;
    }
    provider = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:inputs error:&foundation_error];
    if (provider == nil) {
        return m3_coreml_error(error, M3_STATUS_INVALID_ARGUMENT,
                               "cannot build Core ML inputs",
                               foundation_error);
    }
    options = [[MLPredictionOptions alloc] init];
    options.outputBackings = backings;
    prediction = [model predictionFromFeatures:provider
                                       options:options
                                         error:&foundation_error];
    if (prediction == nil) {
        return m3_coreml_error(error, M3_STATUS_INTERNAL,
                               "Core ML prediction failed",
                               foundation_error);
    }
    for (index = 0U; index < partition->output_count; ++index) {
        NSString *name = (__bridge NSString *)partition->output_names[index];
        MLMultiArray *array =
            [prediction featureValueForName:name].multiArrayValue;
        m3_tensor_view *destination = &views[partition->outputs[index]];
        MLMultiArray *backing = output_arrays[index];
        MLMultiArrayDataType dtype;
        m3_status status;

        if (array == nil ||
            !m3_coreml_dtype(destination->metadata.dtype, &dtype) ||
            array.dataType != dtype ||
            ![array.shape isEqualToArray:m3_coreml_shape(destination)]) {
            return m3_error_set(error, M3_STATUS_INTERNAL,
                                "Core ML output %s violates its contract",
                                name.UTF8String);
        }
        status = m3_coreml_copy_output(array, backing, destination, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

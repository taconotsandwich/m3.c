/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_metal_helpers.h"

#include <string.h>

_Static_assert(M3_TENSOR_MAX_RANK == 8,
               "Metal view rank ABI changed");
_Static_assert(sizeof(m3_metal_view_parameters) == 152U,
               "Metal view parameter size changed");
_Static_assert(_Alignof(m3_metal_view_parameters) == 8U,
               "Metal view parameter alignment changed");
_Static_assert(M3_DTYPE_F32 == 0 && M3_DTYPE_F16 == 1 &&
                   M3_DTYPE_BF16 == 2 && M3_DTYPE_I32 == 3,
               "Metal dtype ABI changed");

void m3_metal_view_parameters_init(
    m3_metal_view_parameters *parameters,
    const m3_tensor_view *view)
{
    uint8_t axis;

    (void)memset(parameters, 0, sizeof(*parameters));
    parameters->element_count =
        (uint64_t)view->metadata.element_count;
    parameters->byte_offset = (uint64_t)view->byte_offset;
    parameters->rank = (uint32_t)view->metadata.rank;
    parameters->dtype = (uint32_t)view->metadata.dtype;
    for (axis = 0U; axis < view->metadata.rank; ++axis) {
        parameters->shape[axis] = view->metadata.shape[axis];
        parameters->byte_strides[axis] =
            (uint64_t)view->byte_strides[axis];
    }
}

bool m3_metal_has_prior_writer(const m3_command *commands,
                               size_t command_index,
                               const m3_storage *storage)
{
    size_t index;

    if (commands == NULL || storage == NULL) {
        return false;
    }
    for (index = 0U; index < command_index; ++index) {
        if (m3_metal_command_writes_storage(&commands[index], storage)) {
            return true;
        }
    }
    return false;
}

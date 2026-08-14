/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_weight_stage_internal.h"

#include <stdlib.h>
#include <string.h>

void m3_weight_stage_init(m3_weight_stage *stage)
{
    if (stage != NULL) {
        (void)memset(stage, 0, sizeof(*stage));
    }
}

void m3_weight_stage_dispose(m3_weight_stage *stage)
{
    size_t index;

    if (stage == NULL) {
        return;
    }
    if (stage->storages != NULL) {
        for (index = 0U; index < stage->storage_count; ++index) {
            m3_storage_free(stage->storages[index]);
        }
    }
    free(stage->storages);
    free(stage->views);
    m3_weight_stage_init(stage);
}

m3_status m3_weight_stage_load(m3_weight_stage *stage,
                               const m3_weight_table *table,
                               m3_backend *backend,
                               m3_weight_stage_progress progress,
                               void *progress_context, m3_error *error)
{
    m3_weight_stage_io io;

    m3_weight_stage_io_init(&io);
    return m3_weight_stage_load_with_io(stage, table, backend, progress,
                                        progress_context, &io, error);
}

const m3_tensor_view *m3_weight_stage_find_view(
    const m3_weight_stage *stage, const char *name)
{
    const m3_weight_binding *binding;
    size_t index;

    if (stage == NULL || name == NULL || stage->table == NULL ||
        stage->view_count != stage->table->binding_count ||
        (stage->view_count != 0U && stage->views == NULL)) {
        return NULL;
    }
    binding = m3_weight_table_find(stage->table, name);
    if (binding == NULL) {
        return NULL;
    }
    index = (size_t)(binding - stage->table->bindings);
    return index < stage->view_count ? &stage->views[index] : NULL;
}

/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3.h"
#include "m3_test.h"
#include "test_cases.h"

#include <stddef.h>

int main(void)
{
    static const m3_test_case cases[] = {
        {"public and error contracts", m3_test_api_contract},
        {"tensor contract", m3_test_tensor_contract},
        {"model contract", m3_test_model_contract},
        {"fixture contract", m3_test_fixture_contract}
    };
    static const size_t case_count = sizeof(cases) / sizeof(cases[0]);

    return m3_test_run("m3 " M3_VERSION_STRING, cases, case_count);
}

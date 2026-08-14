/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_TOKENIZER_TEST_H
#define M3_TOKENIZER_TEST_H

#include "m3_test.h"
#include "m3_tokenizer.h"
#include "tokenizer_fixture.h"

#define M3_FIXTURE_AA 256U
#define M3_FIXTURE_AAA 257U
#define M3_FIXTURE_A1 258U
#define M3_FIXTURE_TWO_SPACES 259U
#define M3_FIXTURE_SPACE_H 260U
#define M3_FIXTURE_SPACE_BANG 261U
#define M3_FIXTURE_TAB_H 262U
#define M3_FIXTURE_TAB_BANG 263U
#define M3_FIXTURE_12 264U
#define M3_FIXTURE_CRLF 265U
#define M3_FIXTURE_E_ACUTE 266U
#define M3_FIXTURE_I_APOSTROPHE 267U
#define M3_FIXTURE_APOSTROPHE_C5 268U
#define M3_FIXTURE_APOSTROPHE_LONG_S 269U
#define M3_FIXTURE_LONG_S_A 270U
#define M3_FIXTURE_APOSTROPHE_M 271U
#define M3_FIXTURE_APOSTROPHE_M_X 272U
#define M3_FIXTURE_ARABIC_ONE 273U
#define M3_FIXTURE_ARABIC_TWO 274U
#define M3_FIXTURE_ARABIC_TWELVE 275U
#define M3_FIXTURE_NBSP 276U
#define M3_FIXTURE_NBSP_H 277U
#define M3_FIXTURE_NBSP_BANG 278U

bool m3_test_tokenizer_fixture_load(m3_tokenizer *tokenizer,
                                    m3_tokenizer_fixture *fixture,
                                    m3_tokenizer_fixture_kind kind,
                                    m3_error *error);
bool m3_test_tokenizer_encode_text(m3_tokenizer *tokenizer,
                                   const char *text, size_t length,
                                   m3_token_ids *ids, m3_error *error);
void m3_test_expect_ids(m3_test_context *test, const m3_token_ids *actual,
                        const uint32_t *expected, size_t expected_count,
                        const char *message);
bool m3_test_fixture_insert_before(m3_tokenizer_fixture *fixture,
                                   const char *needle, uint8_t byte);

#endif

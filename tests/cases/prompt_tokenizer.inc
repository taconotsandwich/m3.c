/* SPDX-License-Identifier: GPL-2.0-only */

M3_TEST_CASE(m3_test_prompt_exact_contract,
             "Music3 prompt markers and tokenizer byte view")
M3_TEST_CASE(m3_test_prompt_caption_normalization,
             "Music3 caption normalization")
M3_TEST_CASE(m3_test_prompt_lyrics_normalization,
             "Music3 lyrics normalization")
M3_TEST_CASE(m3_test_prompt_edge_behavior,
             "Music3 prompt edge behavior")
M3_TEST_CASE(m3_test_prompt_trailing_line_breaks,
             "Music3 caption and lyrics trailing line breaks")
M3_TEST_CASE(m3_test_prompt_invalid_inputs,
             "Music3 prompt input validation")
M3_TEST_CASE(m3_test_tokenizer_bpe_boundaries,
             "Music3 tokenizer BPE boundary semantics")
M3_TEST_CASE(m3_test_tokenizer_unicode_vectors,
             "Music3 tokenizer Unicode and NFC vectors")
M3_TEST_CASE(m3_test_tokenizer_whitespace_vectors,
             "Music3 tokenizer ordered whitespace alternatives")
M3_TEST_CASE(m3_test_tokenizer_added_and_decode,
             "Music3 tokenizer added tokens and byte decoding")
M3_TEST_CASE(m3_test_tokenizer_loader_rejection,
             "Music3 tokenizer strict loader rejection and atomicity")
M3_TEST_CASE(m3_test_tokenizer_cfg_row,
             "Music3 tokenizer classifier-free guidance row")

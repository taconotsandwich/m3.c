M3_TEST_CASE(m3_test_music3_scripted_pipeline,
             "Music3 orchestrates every phase and public output operation")
M3_TEST_CASE(m3_test_music3_reduced_metal_pipeline,
             "Music3 runs reduced real-Metal flow, vocoder, and assembly")
M3_TEST_CASE(m3_test_music3_result_validation,
             "Music3 withholds phase finals for malformed child results")
M3_TEST_CASE(m3_test_music3_cancellation_atomicity,
             "Music3 cancellation preserves output and releases live state")
M3_TEST_CASE(m3_test_music3_ownership_atomicity,
             "Music3 rejects cross-owner aliases without freeing old owners")
M3_TEST_CASE(m3_test_music3_open_atomicity,
             "Music3 open retains exact official component totals atomically")
M3_TEST_CASE(m3_test_music3_aggregate_preflights,
             "Music3 aggregate preflights include every live and future owner")
M3_TEST_CASE(m3_test_music3_reentrant_and_cross_engine,
             "Music3 rejects reentrancy and cross-engine output replacement")

/* SPDX-License-Identifier: GPL-2.0-only */

M3_TEST_CASE(test_vocoder_official_plan,
             "vocoder official materialization plan")
M3_TEST_CASE(test_vocoder_plan_rejects_invalid_config,
             "vocoder plan invalid configuration")
M3_TEST_CASE(test_vocoder_host_materialization,
             "vocoder host weight materialization")
M3_TEST_CASE(test_vocoder_dim0_normalization,
             "vocoder dim-zero weight normalization")
M3_TEST_CASE(test_vocoder_fma_sensitive_order,
             "vocoder FMA-sensitive normalization order")
M3_TEST_CASE(test_vocoder_nonfinite_and_zero_norm,
             "vocoder finite normalization invariant")
M3_TEST_CASE(test_vocoder_nonfinite_direct_sources,
             "vocoder finite direct-source invariant")
M3_TEST_CASE(test_vocoder_missing_and_mismatched_sources,
             "vocoder exact source contracts")
M3_TEST_CASE(test_vocoder_backend_limits_and_rollback,
             "vocoder backend limits and rollback")
M3_TEST_CASE(test_vocoder_cancellation_and_bounded_io,
             "vocoder cancellation and bounded I/O")
M3_TEST_CASE(test_vocoder_stage_lifetime_independence,
             "vocoder staged-weight lifetime independence")
M3_TEST_CASE(test_vocoder_real_metal_ownership,
             "vocoder real Metal destination ownership")
M3_TEST_CASE(test_vocoder_decode_plan_oracles,
             "vocoder decoder production memory plan")
M3_TEST_CASE(test_vocoder_decode_host_oracle,
             "vocoder decoder reduced host oracle")
M3_TEST_CASE(test_vocoder_decode_atomic_cancel,
             "vocoder decoder atomic cancellation")
M3_TEST_CASE(test_vocoder_decode_preflight_limits,
             "vocoder decoder preflight limits")
M3_TEST_CASE(test_vocoder_decode_rejects_malformed_input,
             "vocoder decoder malformed input allocation atomicity")
M3_TEST_CASE(test_metal_vocoder_decode_chunk,
             "vocoder decoder real Metal integration")
M3_TEST_CASE(test_vocoder_decode_rejects_partial_topology,
             "vocoder decoder rejects partial topology")
M3_TEST_CASE(test_waveform_assembly_boundaries,
             "waveform official crop and concatenate boundaries")
M3_TEST_CASE(test_waveform_assembly_clamp_bits,
             "waveform bit-exact post-tanh clamp")
M3_TEST_CASE(test_waveform_assembly_alias_lifetime,
             "waveform aliased input ownership lifetime")
M3_TEST_CASE(test_waveform_assembly_io_faults,
             "waveform bounded storage I/O faults")
M3_TEST_CASE(test_waveform_assembly_cancel_retry,
             "waveform channel cancellation and retry")
M3_TEST_CASE(test_waveform_assembly_validation_atomicity,
             "waveform validation side-effect ordering")
M3_TEST_CASE(test_waveform_assembly_maximum_preflight,
             "waveform maximum metadata preflight")
M3_TEST_CASE(test_metal_waveform_assembly,
             "waveform real Metal storage assembly")

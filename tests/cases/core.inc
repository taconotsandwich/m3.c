/* SPDX-License-Identifier: GPL-2.0-only */

M3_TEST_CASE(m3_test_api_contract, "public and error contracts")
M3_TEST_CASE(m3_test_tensor_contract, "tensor contract")
M3_TEST_CASE(m3_test_model_contract, "model contract")
M3_TEST_CASE(m3_test_fixture_contract, "fixture contract")
M3_TEST_CASE(m3_test_host_storage_lifetime, "host storage lifetime")
M3_TEST_CASE(m3_test_tensor_views, "tensor views")
M3_TEST_CASE(m3_test_tensor_view_rejections, "tensor view rejections")
M3_TEST_CASE(m3_test_scratch_arena, "scratch arena")
M3_TEST_CASE(m3_test_weight_bindings, "weight bindings")
M3_TEST_CASE(m3_test_weight_binding_rejections, "weight binding rejections")
M3_TEST_CASE(m3_test_weight_stage_load, "stage exact component weights")
M3_TEST_CASE(m3_test_weight_stage_snapshot_rejections,
             "reject changed staged-weight shards")
M3_TEST_CASE(m3_test_weight_stage_io_recovery,
             "recover exact staged-weight reads")
M3_TEST_CASE(m3_test_weight_stage_io_failures,
             "contain staged-weight I/O failures")
M3_TEST_CASE(m3_test_weight_stage_cancellation,
             "cancel staged-weight loading atomically")
M3_TEST_CASE(m3_test_weight_stage_preflight_limits,
             "preflight staged-weight backend limits")
M3_TEST_CASE(m3_test_weight_stage_allocation_failure,
             "contain staged-weight allocation failure")
M3_TEST_CASE(m3_test_weight_stage_contract_rejections,
             "reject invalid staged-weight contracts")
M3_TEST_CASE(m3_test_op_conversions, "operation numeric conversions")
M3_TEST_CASE(m3_test_op_broadcast_embedding, "broadcast and embedding operations")
M3_TEST_CASE(m3_test_op_dense_norms, "dense and normalization operations")
M3_TEST_CASE(m3_test_op_rope, "rotary position operations")
M3_TEST_CASE(m3_test_op_attention, "reference attention operation")
M3_TEST_CASE(m3_test_op_softmax, "stable softmax operation")
M3_TEST_CASE(m3_test_op_sampling, "top-k and categorical operations")
M3_TEST_CASE(m3_test_op_validation_atomicity, "command validation atomicity")
M3_TEST_CASE(m3_test_op_scratch_rejections, "operation scratch rejections")
M3_TEST_CASE(m3_test_op_sampling_shape_rejections, "sampling shape rejections")
M3_TEST_CASE(m3_test_op_metal_unsupported, "Metal operation rejection")
M3_TEST_CASE(m3_test_op_conv1d, "reference Conv1d operation")
M3_TEST_CASE(m3_test_op_conv_transpose1d, "reference ConvTranspose1d operation")
M3_TEST_CASE(m3_test_op_nearest_resize1d, "reference nearest resize operation")
M3_TEST_CASE(m3_test_op_convolution_validation, "convolution validation")
M3_TEST_CASE(m3_test_op_conv_transpose_validation, "transposed convolution validation")
M3_TEST_CASE(m3_test_op_resize_safety, "nearest resize safety")

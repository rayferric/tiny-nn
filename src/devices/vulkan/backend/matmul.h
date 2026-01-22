#pragma once

#include "../impl.h"

#include <vulkan_shaders/matmul.h>

static void matmul(
    tnn_device_t *dev,
    const void *a,
    const void *b,
    void *c,
    size_t m,
    size_t k,
    size_t n,
    bool tpose_a,
    bool tpose_b,
    bool accum
) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

	vk_buffer_t *bufs[3] = {
	    (vk_buffer_t *)a, (vk_buffer_t *)b, (vk_buffer_t *)c
	};
	struct {
		uint32_t M;
		uint32_t K;
		uint32_t N;
		uint32_t tpose_a;
		uint32_t tpose_b;
		uint32_t accum;
	} push_constants = {
	    .M = (uint32_t)m,
	    .K = (uint32_t)k,
	    .N = (uint32_t)n,
	    .tpose_a = tpose_a ? 1 : 0,
	    .tpose_b = tpose_b ? 1 : 0,
	    .accum = accum ? 1 : 0,
	};

	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	dispatch_kernel(
	    dev,
	    &ctx->kernels.matmul,
	    matmul_spv,
	    matmul_spv_len,
	    bufs,
	    sizeof(bufs) / sizeof(bufs[0]),
	    &push_constants,
	    sizeof(push_constants),
	    (n + 15) / 16,
	    (m + 15) / 16,
	    1,
	    // ^ matching local work group size in the shader
	    cmd
	);
}

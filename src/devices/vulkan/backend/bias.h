#pragma once

#include "../impl.h"

#include <vulkan_shaders/accum.h>
#include <vulkan_shaders/add.h>
#include <vulkan_shaders/sum.h>

static void
add(tnn_device_t *dev,
    const void *a,
    const void *b,
    void *out,
    size_t outer,
    size_t inner) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	vk_buffer_t *bufs[3] = {
	    (vk_buffer_t *)a, (vk_buffer_t *)b, (vk_buffer_t *)out
	};
	struct {
		uint32_t outer;
		uint32_t inner;
	} push_constants = {
	    .outer = (uint32_t)outer,
	    .inner = (uint32_t)inner,
	};
	dispatch_kernel(
	    dev,
	    &ctx->kernels.add,
	    add_spv,
	    add_spv_len,
	    bufs,
	    sizeof(bufs) / sizeof(bufs[0]),
	    &push_constants,
	    sizeof(push_constants),
	    (uint32_t)outer,
	    (uint32_t)inner,
	    1,
	    cmd
	);
}

static void accum(tnn_device_t *dev, const void *in, void *out, size_t n) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	vk_buffer_t *bufs[2] = {(vk_buffer_t *)in, (vk_buffer_t *)out};
	struct {
		uint32_t size;
	} push_constants = {
	    .size = (uint32_t)n,
	};
	dispatch_kernel(
	    dev,
	    &ctx->kernels.accum,
	    accum_spv,
	    accum_spv_len,
	    bufs,
	    sizeof(bufs) / sizeof(bufs[0]),
	    &push_constants,
	    sizeof(push_constants),
	    (uint32_t)((n + 255) / 256),
	    1,
	    1,
	    cmd
	);
}

static void
sum(tnn_device_t *dev,
    const void *in,
    void *out,
    size_t outer,
    size_t reduced,
    size_t inner,
    float scale,
    bool accum,
    bool reverse) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	vk_buffer_t *bufs[2] = {(vk_buffer_t *)in, (vk_buffer_t *)out};
	struct {
		uint32_t outer;
		uint32_t reduced;
		uint32_t inner;
		float scale;
		uint32_t accum;
		uint32_t reverse;
	} push_constants = {
	    .outer = (uint32_t)outer,
	    .reduced = (uint32_t)reduced,
	    .inner = (uint32_t)inner,
	    .scale = scale,
	    .accum = accum ? 1 : 0,
	    .reverse = reverse ? 1 : 0,
	};
	dispatch_kernel(
	    dev,
	    &ctx->kernels.sum,
	    sum_spv,
	    sum_spv_len,
	    bufs,
	    sizeof(bufs) / sizeof(bufs[0]),
	    &push_constants,
	    sizeof(push_constants),
	    outer * inner,
	    1,
	    1,
	    cmd
	);
}

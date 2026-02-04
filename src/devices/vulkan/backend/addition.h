#pragma once

#include <math.h>

#include "../impl.h"

#include <vulkan_shaders/accum.h>
#include <vulkan_shaders/add.h>
#include <vulkan_shaders/sum_broadcast.h>
#include <vulkan_shaders/sum_reduce.h>

static void
add(tnn_device_t *dev,
    const void *a,
    const void *b,
    void *out,
    size_t outer,
    size_t inner) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

	uint32_t grid_size = ceil(sqrt(outer * inner)) + 0.5f;

	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	vk_buffer_t *bufs[3] = {
	    (vk_buffer_t *)a, (vk_buffer_t *)b, (vk_buffer_t *)out
	};
	struct {
		uint32_t outer;
		uint32_t inner;
		uint32_t grid_size;
	} push_constants = {
	    .outer = outer,
	    .inner = inner,
	    .grid_size = grid_size,
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
	    grid_size,
	    grid_size,
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

static void sum_reduce(
    tnn_device_t *dev,
    const void *in,
    void *out,
    size_t outer,
    size_t reduced,
    size_t inner,
    float scale,
    bool accum
) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	vk_buffer_t *bufs[2] = {(vk_buffer_t *)in, (vk_buffer_t *)out};
	struct {
		uint32_t outer;
		uint32_t reduced;
		uint32_t inner;
		float scale;
		uint32_t accum;
	} push_constants = {
	    .outer = (uint32_t)outer,
	    .reduced = (uint32_t)reduced,
	    .inner = (uint32_t)inner,
	    .scale = scale,
	    .accum = accum ? 1 : 0,
	};
	dispatch_kernel(
	    dev,
	    &ctx->kernels.sum_reduce,
	    sum_reduce_spv,
	    sum_reduce_spv_len,
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

static void sum_broadcast(
    tnn_device_t *dev,
    const void *in,
    void *out,
    size_t outer,
    size_t reduced,
    size_t inner,
    float scale,
    bool accum
) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	vk_buffer_t *bufs[2] = {(vk_buffer_t *)in, (vk_buffer_t *)out};
	struct {
		uint32_t outer;
		uint32_t reduced;
		uint32_t inner;
		uint32_t accum;
		float scale;
	} push_constants = {
	    .outer = (uint32_t)outer,
	    .reduced = (uint32_t)reduced,
	    .inner = (uint32_t)inner,
	    .accum = accum ? 1 : 0,
	    .scale = scale,
	};
	dispatch_kernel(
	    dev,
	    &ctx->kernels.sum_broadcast,
	    sum_broadcast_spv,
	    sum_broadcast_spv_len,
	    bufs,
	    sizeof(bufs) / sizeof(bufs[0]),
	    &push_constants,
	    sizeof(push_constants),
	    (outer * reduced * inner + 255) / 256,
	    1,
	    1,
	    cmd
	);
}

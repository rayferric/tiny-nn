#pragma once

#include "../impl.h"

#include <vulkan_shaders/relu_bw.h>
#include <vulkan_shaders/relu_fw.h>

static void
relu_fw(tnn_device_t *dev, const void *input, void *output, size_t size) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	vk_buffer_t *bufs[2] = {(vk_buffer_t *)input, (vk_buffer_t *)output};
	struct {
		uint32_t size;
	} push_constants = {
	    .size = (uint32_t)size,
	};
	dispatch_kernel(
	    dev,
	    &ctx->kernels.relu_fw,
	    relu_fw_spv,
	    relu_fw_spv_len,
	    bufs,
	    sizeof(bufs) / sizeof(bufs[0]),
	    &push_constants,
	    sizeof(push_constants),
	    (size + 255) / 256,
	    1,
	    1,
	    cmd
	);
}

static void relu_bw(
    tnn_device_t *dev,
    const void *output,
    const void *output_grad,
    void *input_grad,
    size_t size
) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	vk_buffer_t *bufs[3] = {
	    (vk_buffer_t *)output,
	    (vk_buffer_t *)output_grad,
	    (vk_buffer_t *)input_grad
	};
	struct {
		uint32_t size;
	} push_constants = {
	    .size = (uint32_t)size,
	};
	dispatch_kernel(
	    dev,
	    &ctx->kernels.relu_bw,
	    relu_bw_spv,
	    relu_bw_spv_len,
	    bufs,
	    sizeof(bufs) / sizeof(bufs[0]),
	    &push_constants,
	    sizeof(push_constants),
	    (size + 255) / 256,
	    1,
	    1,
	    cmd
	);
}

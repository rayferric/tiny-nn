#pragma once

#include "../impl.h"

#include <vulkan_shaders/conv_bw_input.h>
#include <vulkan_shaders/conv_bw_weight.h>
#include <vulkan_shaders/conv_fw.h>

// input: [batch, h_in, w_in, c_in]
// weight: [c_out, k, k, c_in]
// output: [batch, h_out, w_out, c_out]
static void conv_fw(
    tnn_device_t *dev,
    const void *input,
    const void *weight,
    void *output,
    size_t batch,
    size_t h_in,
    size_t w_in,
    size_t c_in,
    size_t h_out,
    size_t w_out,
    size_t c_out,
    size_t kernel_size,
    size_t stride,
    size_t padding
) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	vk_buffer_t *bufs[3] = {
	    (vk_buffer_t *)input, (vk_buffer_t *)weight, (vk_buffer_t *)output
	};
	struct {
		uint32_t batch;
		uint32_t h_in;
		uint32_t w_in;
		uint32_t c_in;
		uint32_t h_out;
		uint32_t w_out;
		uint32_t c_out;
		uint32_t kernel_size;
		uint32_t stride;
		uint32_t padding;
	} push_constants = {
	    .batch = (uint32_t)batch,
	    .h_in = (uint32_t)h_in,
	    .w_in = (uint32_t)w_in,
	    .c_in = (uint32_t)c_in,
	    .h_out = (uint32_t)h_out,
	    .w_out = (uint32_t)w_out,
	    .c_out = (uint32_t)c_out,
	    .kernel_size = (uint32_t)kernel_size,
	    .stride = (uint32_t)stride,
	    .padding = (uint32_t)padding
	};
	dispatch_kernel(
	    dev,
	    &ctx->kernels.conv_fw,
	    conv_fw_spv,
	    conv_fw_spv_len,
	    bufs,
	    sizeof(bufs) / sizeof(bufs[0]),
	    &push_constants,
	    sizeof(push_constants),
	    (batch * h_out * w_out * c_out + 255) / 256,
	    1,
	    1,
	    cmd
	);
}

// out_grad: [batch, h_out, w_out, c_out]
// weight: [c_out, k, k, c_in]
// in_grad: [batch, h_in, w_in, c_in]
static void conv_bw_input(
    tnn_device_t *dev,
    const void *out_grad,
    const void *weight,
    void *in_grad,
    size_t batch,
    size_t h_in,
    size_t w_in,
    size_t c_in,
    size_t h_out,
    size_t w_out,
    size_t c_out,
    size_t kernel_size,
    size_t stride,
    size_t padding
) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	vk_buffer_t *bufs[3] = {
	    (vk_buffer_t *)out_grad, (vk_buffer_t *)weight, (vk_buffer_t *)in_grad
	};
	struct {
		uint32_t batch;
		uint32_t h_in;
		uint32_t w_in;
		uint32_t c_in;
		uint32_t h_out;
		uint32_t w_out;
		uint32_t c_out;
		uint32_t kernel_size;
		uint32_t stride;
		uint32_t padding;
	} push_constants = {
	    .batch = (uint32_t)batch,
	    .h_in = (uint32_t)h_in,
	    .w_in = (uint32_t)w_in,
	    .c_in = (uint32_t)c_in,
	    .h_out = (uint32_t)h_out,
	    .w_out = (uint32_t)w_out,
	    .c_out = (uint32_t)c_out,
	    .kernel_size = (uint32_t)kernel_size,
	    .stride = (uint32_t)stride,
	    .padding = (uint32_t)padding
	};
	dispatch_kernel(
	    dev,
	    &ctx->kernels.conv_bw_input,
	    conv_bw_input_spv,
	    conv_bw_input_spv_len,
	    bufs,
	    sizeof(bufs) / sizeof(bufs[0]),
	    &push_constants,
	    sizeof(push_constants),
	    (batch * h_in * w_in * c_in + 255) / 256,
	    1,
	    1,
	    cmd
	);
}

// input: [batch, h_in, w_in, c_in]
// out_grad: [batch, h_out, w_out, c_out]
// weight_grad: [c_out, k, k, c_in]
static void conv_bw_weight(
    tnn_device_t *dev,
    const void *input,
    const void *out_grad,
    void *weight_grad,
    size_t batch,
    size_t h_in,
    size_t w_in,
    size_t c_in,
    size_t h_out,
    size_t w_out,
    size_t c_out,
    size_t kernel_size,
    size_t stride,
    size_t padding
) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	vk_buffer_t *bufs[3] = {
	    (vk_buffer_t *)input,
	    (vk_buffer_t *)out_grad,
	    (vk_buffer_t *)weight_grad
	};
	struct {
		uint32_t batch;
		uint32_t h_in;
		uint32_t w_in;
		uint32_t c_in;
		uint32_t h_out;
		uint32_t w_out;
		uint32_t c_out;
		uint32_t kernel_size;
		uint32_t stride;
		uint32_t padding;
	} push_constants = {
	    .batch = (uint32_t)batch,
	    .h_in = (uint32_t)h_in,
	    .w_in = (uint32_t)w_in,
	    .c_in = (uint32_t)c_in,
	    .h_out = (uint32_t)h_out,
	    .w_out = (uint32_t)w_out,
	    .c_out = (uint32_t)c_out,
	    .kernel_size = (uint32_t)kernel_size,
	    .stride = (uint32_t)stride,
	    .padding = (uint32_t)padding
	};
	dispatch_kernel(
	    dev,
	    &ctx->kernels.conv_bw_weight,
	    conv_bw_weight_spv,
	    conv_bw_weight_spv_len,
	    bufs,
	    sizeof(bufs) / sizeof(bufs[0]),
	    &push_constants,
	    sizeof(push_constants),
	    (c_out * kernel_size * kernel_size * c_in + 255) / 256,
	    1,
	    1,
	    cmd
	);
}

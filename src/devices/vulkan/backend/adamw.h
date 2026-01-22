#pragma once

#include "../impl.h"

#include <vulkan_shaders/adamw.h>

static void adamw(
    tnn_device_t *dev,
    void *param_data,
    void *param_grad,
    void *m1_data,
    void *m2_data,
    size_t param_size,
    float t,
    float lr,
    float b1,
    float b2,
    float eps,
    float wd
) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

	vk_buffer_t *bufs[4] = {
	    (vk_buffer_t *)param_data,
	    (vk_buffer_t *)param_grad,
	    (vk_buffer_t *)m1_data,
	    (vk_buffer_t *)m2_data,
	};

	struct {
		uint32_t param_size;
		float t;
		float lr;
		float b1;
		float b2;
		float eps;
		float wd;
	} push_constants = {
	    .param_size = (uint32_t)param_size,
	    .t = t,
	    .lr = lr,
	    .b1 = b1,
	    .b2 = b2,
	    .eps = eps,
	    .wd = wd
	};

	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	dispatch_kernel(
	    dev,
	    &ctx->kernels.adamw,
	    adamw_spv,
	    adamw_spv_len,
	    bufs,
	    sizeof(bufs) / sizeof(bufs[0]),
	    &push_constants,
	    sizeof(push_constants),
	    (param_size + 255) / 256, // 256 local threads in shader
	    1,
	    1,
	    cmd
	);
}

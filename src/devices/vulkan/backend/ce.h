#pragma once

#include "../impl.h"

#include "./memory.h"

#include <vulkan_shaders/ce_bw.h>
#include <vulkan_shaders/ce_fw_1.h>
#include <vulkan_shaders/ce_fw_2.h>

static void ce_fw(
    tnn_device_t *dev,
    const void *pred,
    const void *tgt,
    void *out,
    size_t n,
    size_t c
) {
	TNN_TRACY_ZONE_START();

	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

	VkCommandBuffer cmd = ensure_ready_for_recording(dev);

	vk_buffer_t *per_batch_loss_tmp = buf_alloc(dev, n * sizeof(float));

	{
		vk_buffer_t *bufs[3] = {
		    (vk_buffer_t *)pred, (vk_buffer_t *)tgt, per_batch_loss_tmp
		};
		struct {
			uint32_t n;
			uint32_t c;
		} push_constants = {
		    .n = (uint32_t)n,
		    .c = (uint32_t)c,
		};
		dispatch_kernel(
		    dev,
		    &ctx->kernels.ce_fw_1,
		    ce_fw_1_spv,
		    ce_fw_1_spv_len,
		    bufs,
		    sizeof(bufs) / sizeof(bufs[0]),
		    &push_constants,
		    sizeof(push_constants),
		    n,
		    1,
		    1,
		    cmd
		);
	}
	// second kernel for final reduction instead of atomicAdd in
	// single kernel --- this needs the aux buffer allocation + free
	{
		vk_buffer_t *bufs[2] = {
		    per_batch_loss_tmp,
		    (vk_buffer_t *)out,
		};
		struct {
			uint32_t n;
		} push_constants = {
		    .n = (uint32_t)n,
		};
		dispatch_kernel(
		    dev,
		    &ctx->kernels.ce_fw_2,
		    ce_fw_2_spv,
		    ce_fw_2_spv_len,
		    bufs,
		    sizeof(bufs) / sizeof(bufs[0]),
		    &push_constants,
		    sizeof(push_constants),
		    1,
		    1,
		    1,
		    cmd
		);
	}

	buf_free(dev, per_batch_loss_tmp);

	TNN_TRACY_ZONE_END();
}

static void ce_bw(
    tnn_device_t *dev,
    const void *pred,
    const void *tgt,
    const void *out_grad,
    void *pred_grad,
    size_t n,
    size_t c
) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

	vk_buffer_t *bufs[4] = {
	    (vk_buffer_t *)pred,
	    (vk_buffer_t *)tgt,
	    (vk_buffer_t *)out_grad,
	    (vk_buffer_t *)pred_grad
	};

	struct {
		uint32_t n;
		uint32_t c;
	} push_constants = {
	    .n = (uint32_t)n,
	    .c = (uint32_t)c,
	};

	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	dispatch_kernel(
	    dev,
	    &ctx->kernels.ce_bw,
	    ce_bw_spv,
	    ce_bw_spv_len,
	    bufs,
	    sizeof(bufs) / sizeof(bufs[0]),
	    &push_constants,
	    sizeof(push_constants),
	    n,
	    1,
	    1,
	    cmd
	);
}

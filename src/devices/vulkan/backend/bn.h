#pragma once

#include "../impl.h"

#include "./memory.h"

#include <vulkan_shaders/bn_bw_test.h>
#include <vulkan_shaders/bn_bw_train.h>
#include <vulkan_shaders/bn_fw_1.h>
#include <vulkan_shaders/bn_fw_2.h>
#include <vulkan_shaders/bn_fw_3.h>

// input/output: [NHW, C]
// running_mean/running_var: [C]
// batch_mean/batch_var: [C] - written if !test
static void bn_fw(
    tnn_device_t *dev,
    const void *input,
    void *output,
    void *running_mean,
    void *running_var,
    void *batch_var,
    size_t NHW,
    size_t C,
    float momentum,
    bool test
) {
	TNN_TRACY_ZONE_START();

	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

	VkCommandBuffer cmd = ensure_ready_for_recording(dev);

	if (test) {
		vk_buffer_t *bufs_norm[6] = {
		    (vk_buffer_t *)input,
		    (vk_buffer_t *)output,
		    (vk_buffer_t *)running_mean,
		    (vk_buffer_t *)running_var,
		    (vk_buffer_t *)running_mean,
		    (vk_buffer_t *)running_var,
		};

		struct {
			uint32_t NHW;
			uint32_t C;
			float momentum;
			uint32_t test;
		} push_constants = {
		    .NHW = (uint32_t)NHW,
		    .C = (uint32_t)C,
		    .momentum = momentum,
		    .test = 1,
		};

		dispatch_kernel(
		    dev,
		    &ctx->kernels.bn_fw_3,
		    bn_fw_3_spv,
		    bn_fw_3_spv_len,
		    bufs_norm,
		    6,
		    &push_constants,
		    sizeof(push_constants),
		    C,
		    1,
		    1,
		    cmd
		);
	} else {
		vk_buffer_t *batch_mean_tmp = buf_alloc(dev, C * sizeof(float));

		// Step 1: Compute mean
		{
			vk_buffer_t *bufs[2] = {
			    (vk_buffer_t *)input,
			    batch_mean_tmp,
			};
			struct {
				uint32_t NHW;
				uint32_t C;
			} push_constants = {
			    .NHW = (uint32_t)NHW,
			    .C = (uint32_t)C,
			};
			dispatch_kernel(
			    dev,
			    &ctx->kernels.bn_fw_1,
			    bn_fw_1_spv,
			    bn_fw_1_spv_len,
			    bufs,
			    2,
			    &push_constants,
			    sizeof(push_constants),
			    C,
			    1,
			    1,
			    cmd
			);
		}

		// Step 2: Compute variance
		{
			vk_buffer_t *bufs[3] = {
			    (vk_buffer_t *)input,
			    batch_mean_tmp,
			    (vk_buffer_t *)batch_var,
			};
			struct {
				uint32_t NHW;
				uint32_t C;
			} push_constants = {
			    .NHW = (uint32_t)NHW,
			    .C = (uint32_t)C,
			};
			dispatch_kernel(
			    dev,
			    &ctx->kernels.bn_fw_2,
			    bn_fw_2_spv,
			    bn_fw_2_spv_len,
			    bufs,
			    3,
			    &push_constants,
			    sizeof(push_constants),
			    C,
			    1,
			    1,
			    cmd
			);
		}

		// Step 3: Normalize and update running stats
		{
			vk_buffer_t *bufs[6] = {
			    (vk_buffer_t *)input,
			    (vk_buffer_t *)output,
			    batch_mean_tmp,
			    (vk_buffer_t *)batch_var,
			    (vk_buffer_t *)running_mean,
			    (vk_buffer_t *)running_var,
			};
			struct {
				uint32_t NHW;
				uint32_t C;
				float momentum;
				uint32_t test;
			} push_constants = {
			    .NHW = (uint32_t)NHW,
			    .C = (uint32_t)C,
			    .momentum = momentum,
			    .test = 0,
			};
			dispatch_kernel(
			    dev,
			    &ctx->kernels.bn_fw_3,
			    bn_fw_3_spv,
			    bn_fw_3_spv_len,
			    bufs,
			    6,
			    &push_constants,
			    sizeof(push_constants),
			    C,
			    1,
			    1,
			    cmd
			);
		}

		buf_free(dev, batch_mean_tmp);
	}

	TNN_TRACY_ZONE_END();
}

static void bn_bw(
    tnn_device_t *dev,
    const void *out_grad,    // [NHW, C]
    const void *out_data,    // [NHW, C]
    void *in_grad,           // [NHW, C]
    const void *running_var, // [C]
    const void *batch_var,   // [C]
    size_t NHW,
    size_t C,
    bool test
) {
	TNN_TRACY_ZONE_START();

	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

	VkCommandBuffer cmd = ensure_ready_for_recording(dev);

	if (test) {
		// Test mode: simple gradient scaling
		vk_buffer_t *bufs[3] = {
		    (vk_buffer_t *)out_grad,
		    (vk_buffer_t *)in_grad,
		    (vk_buffer_t *)running_var,
		};
		struct {
			uint32_t NHW;
			uint32_t C;
		} push_constants = {
		    .NHW = (uint32_t)NHW,
		    .C = (uint32_t)C,
		};
		dispatch_kernel(
		    dev,
		    &ctx->kernels.bn_bw_test,
		    bn_bw_test_spv,
		    bn_bw_test_spv_len,
		    bufs,
		    3,
		    &push_constants,
		    sizeof(push_constants),
		    C,
		    1,
		    1,
		    cmd
		);
	} else {
		// Training mode: full backprop through normalization
		vk_buffer_t *bufs[4] = {
		    (vk_buffer_t *)out_grad,
		    (vk_buffer_t *)out_data,
		    (vk_buffer_t *)in_grad,
		    (vk_buffer_t *)batch_var,
		};
		struct {
			uint32_t NHW;
			uint32_t C;
		} push_constants = {
		    .NHW = (uint32_t)NHW,
		    .C = (uint32_t)C,
		};
		dispatch_kernel(
		    dev,
		    &ctx->kernels.bn_bw_train,
		    bn_bw_train_spv,
		    bn_bw_train_spv_len,
		    bufs,
		    4,
		    &push_constants,
		    sizeof(push_constants),
		    C,
		    1,
		    1,
		    cmd
		);
	}

	TNN_TRACY_ZONE_END();
}

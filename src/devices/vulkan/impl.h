#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <vulkan/vulkan.h>

#include <tnn/tnn.h>

#include "../../util/safe_malloc.h"

typedef struct {
	VkShaderModule shader_module;
	VkDescriptorSetLayout descriptor_set_layout;
	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;
} vk_kernel_t;

#define VK_MAX_CMD_BUF_LEN 30

typedef union {
	struct {
		vk_kernel_t matmul;
		vk_kernel_t add;
		vk_kernel_t accum;
		vk_kernel_t sum_reduce;
		vk_kernel_t sum_broadcast;
		vk_kernel_t relu_fw;
		vk_kernel_t relu_bw;
		vk_kernel_t ce_fw_1;
		vk_kernel_t ce_fw_2;
		vk_kernel_t ce_bw;
		vk_kernel_t conv_fw;
		vk_kernel_t conv_bw_input;
		vk_kernel_t conv_bw_weight;
		vk_kernel_t adamw;
		vk_kernel_t bn_fw_1;
		vk_kernel_t bn_fw_2;
		vk_kernel_t bn_fw_3;
		vk_kernel_t bn_bw_train;
		vk_kernel_t bn_bw_test;
	};
	vk_kernel_t array[19];
} vk_device_context_kernels_t;

typedef struct {
	// init during device enum
	VkPhysicalDevice physical_device;

	// init on first op
	VkDevice device;
	VkQueue compute_queue;
	uint32_t queue_family_index;
	VkCommandPool command_pool;

	// dev: new fields introduced recently
	vk_device_context_kernels_t kernels;
	VkDescriptorPool descriptor_pool;
	VkCommandBuffer *cmd_bufs;
	VkSemaphore *semaphores; // one less than cmd_bufs
	size_t cmd_bufs_len;
	size_t cmd_bufs_curr_idx; // tracks current buffer
	size_t curr_cmd_buf_num_recorded;
	// ^ new cmd_buf and semaphore every VK_MAX_CMD_BUF_LEN backend calls
	bool currently_recording;
} vk_device_context_t;

typedef struct {
	VkBuffer buffer;
	VkDeviceMemory memory;
	size_t size;
} vk_buffer_t;

void error_exit(tnn_device_t *dev, const char *msg) {
	fprintf(stderr, "%s [%s - %s]\n", msg, dev->name, dev->desc);
	exit(1);
}

static void init_kernel(
    tnn_device_t *dev,
    vk_kernel_t *kernel,
    uint32_t num_buffers,
    uint32_t push_const_sz,
    const uint8_t *spirv,
    size_t spirv_len
) {
	vk_device_context_t *ctx = dev->_ctx;
	VkResult res;

	// shader module
	VkShaderModuleCreateInfo shader_info = {
	    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
	    .codeSize = spirv_len,
	    .pCode = (uint32_t *)spirv
	};
	res = vkCreateShaderModule(
	    ctx->device, &shader_info, NULL, &kernel->shader_module
	);
	if (res != VK_SUCCESS) {
		error_exit(dev, "failed to create matmul shader");
	}

	// descriptor set layout (buffer bindings)
	VkDescriptorSetLayoutBinding *bindings =
	    safe_malloc(num_buffers * sizeof(VkDescriptorSetLayoutBinding));
	for (uint32_t i = 0; i < num_buffers; i++) {
		bindings[i] = (VkDescriptorSetLayoutBinding){
		    .binding = i,
		    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		    .descriptorCount = 1,
		    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
		};
	}
	VkDescriptorSetLayoutCreateInfo layout_info = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	    .bindingCount = num_buffers,
	    .pBindings = bindings
	};
	res = vkCreateDescriptorSetLayout(
	    ctx->device, &layout_info, NULL, &kernel->descriptor_set_layout
	);
	if (res != VK_SUCCESS) {
		error_exit(dev, "failed to create descriptor set layout");
	}
	free(bindings);

	// pipeline layout (push constants + buffer bindings)
	VkPushConstantRange push_constant_range = {
	    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
	    .offset = 0,
	    .size = push_const_sz
	};
	VkPipelineLayoutCreateInfo pipeline_layout_info = {
	    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
	    .setLayoutCount = 1,
	    .pSetLayouts = &kernel->descriptor_set_layout,
	    .pushConstantRangeCount = 1,
	    .pPushConstantRanges = &push_constant_range
	};

	res = vkCreatePipelineLayout(
	    ctx->device, &pipeline_layout_info, NULL, &kernel->pipeline_layout
	);
	if (res != VK_SUCCESS) {
		error_exit(dev, "failed to create pipeline layout");
	}

	// compute pipeline
	VkComputePipelineCreateInfo pipeline_info = {
	    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
	    .stage =
	        {
	            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
	            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
	            .module = kernel->shader_module,
	            .pName = "main" // entry point in shader
	        },
	    .layout = kernel->pipeline_layout
	};
	res = vkCreateComputePipelines(
	    ctx->device,
	    VK_NULL_HANDLE, // no pipeline cache for now
	    1,
	    &pipeline_info,
	    NULL,
	    &kernel->pipeline
	);
	if (res != VK_SUCCESS) {
		error_exit(dev, "failed to create compute pipeline");
	}
}

static void destroy_kernel(tnn_device_t *dev, vk_kernel_t *kernel) {
	if (kernel->shader_module == VK_NULL_HANDLE) {
		return; // not initialized
	}

	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	vkDestroyPipeline(ctx->device, kernel->pipeline, NULL);
	vkDestroyPipelineLayout(ctx->device, kernel->pipeline_layout, NULL);
	vkDestroyDescriptorSetLayout(
	    ctx->device, kernel->descriptor_set_layout, NULL
	);
	vkDestroyShaderModule(ctx->device, kernel->shader_module, NULL);
}

// initializes basic vulkan resources
static void ensure_device_initialized(tnn_device_t *dev) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	if (ctx->device != VK_NULL_HANDLE) {
		return;
	}

	// find compute queue family
	uint32_t num_queue_families = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(
	    ctx->physical_device, &num_queue_families, NULL
	);
	VkQueueFamilyProperties *queue_families =
	    safe_malloc(num_queue_families * sizeof(VkQueueFamilyProperties));
	vkGetPhysicalDeviceQueueFamilyProperties(
	    ctx->physical_device, &num_queue_families, queue_families
	);
	uint32_t compute_family_index = UINT32_MAX;
	for (uint32_t i = 0; i < num_queue_families; i++) {
		if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			compute_family_index = i;
			break;
		}
	}
	free(queue_families);
	if (compute_family_index == UINT32_MAX) {
		error_exit(dev, "no compute queue family found");
	}
	ctx->queue_family_index = compute_family_index;

	// create logical device
	float queue_priority = 1.0f;
	VkDeviceQueueCreateInfo queue_create_info = {
	    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
	    .queueFamilyIndex = compute_family_index,
	    .queueCount = 1,
	    .pQueuePriorities = &queue_priority,
	};
	VkDeviceCreateInfo device_create_info = {
	    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
	    .queueCreateInfoCount = 1,
	    .pQueueCreateInfos = &queue_create_info,
	};
	VkResult res = vkCreateDevice(
	    ctx->physical_device, &device_create_info, NULL, &ctx->device
	);
	if (res != VK_SUCCESS) {
		error_exit(dev, "failed to create logical device");
	}

	// get queue handle
	vkGetDeviceQueue(
	    ctx->device, ctx->queue_family_index, 0, &ctx->compute_queue
	);

	// create command pool
	VkCommandPoolCreateInfo pool_info = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
	    .queueFamilyIndex = ctx->queue_family_index,
	    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
	};
	res =
	    vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->command_pool);
	if (res != VK_SUCCESS) {
		error_exit(dev, "failed to create command pool");
	}

	// single large descriptor pool
	VkDescriptorPoolSize pool_size = {
	    .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	    .descriptorCount = 30000 // expect approx 3 buffers per kernel
	};
	VkDescriptorPoolCreateInfo desc_pool_info = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
	    .maxSets = 10000,
	    .poolSizeCount = 1,
	    .pPoolSizes = &pool_size
	};
	res = vkCreateDescriptorPool(
	    ctx->device, &desc_pool_info, NULL, &ctx->descriptor_pool
	);
}

// ensures at least 1 extra command buffer is available
static void _ensure_enough_cmd_bufs(tnn_device_t *dev) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	if (ctx->cmd_bufs_curr_idx + 1 < ctx->cmd_bufs_len) {
		return; // +1 for one extra
	}

	size_t new_cap = ctx->cmd_bufs_len == 0
	                   ? VK_MAX_CMD_BUF_LEN
	                   : ctx->cmd_bufs_len + VK_MAX_CMD_BUF_LEN;

	ctx->cmd_bufs =
	    safe_realloc(ctx->cmd_bufs, new_cap * sizeof(VkCommandBuffer));
	ctx->semaphores =
	    safe_realloc(ctx->semaphores, (new_cap - 1) * sizeof(VkSemaphore));

	// alloc new cmd bufs
	size_t num_new = new_cap - ctx->cmd_bufs_len;
	VkCommandBufferAllocateInfo alloc_info = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	    .commandPool = ctx->command_pool,
	    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	    .commandBufferCount = (uint32_t)num_new
	};
	VkResult res = vkAllocateCommandBuffers(
	    ctx->device, &alloc_info, &ctx->cmd_bufs[ctx->cmd_bufs_len]
	);
	if (res != VK_SUCCESS) {
		error_exit(dev, "failed to allocate command buffers");
	}

	// create new semaphores (one less than cmd_bufs)
	for (size_t i = ctx->cmd_bufs_len; i < new_cap - 1; ++i) {
		VkSemaphoreCreateInfo sem_info = {
		    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
		};
		res = vkCreateSemaphore(
		    ctx->device, &sem_info, NULL, &ctx->semaphores[i]
		);
		if (res != VK_SUCCESS) {
			error_exit(dev, "failed to create semaphore");
		}
	}

	ctx->cmd_bufs_len = new_cap;
}

// if recording then ends recording, if anything recorded, then submits
// if submitted, advances to the next command buffer without starting recording
static void
_end_submit_current_buffer(tnn_device_t *dev, bool dont_signal_next) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

	if (!ctx->currently_recording) {
		return;
	}

	VkCommandBuffer current_cmd = ctx->cmd_bufs[ctx->cmd_bufs_curr_idx];

	// end current command buffer
	VkResult res = vkEndCommandBuffer(current_cmd);
	if (res != VK_SUCCESS) {
		error_exit(dev, "failed to end command buffer");
	}
	ctx->currently_recording = false;

	if (ctx->curr_cmd_buf_num_recorded == 0) {
		return; // nothing to submit
	}

	// setup submit info
	VkSubmitInfo submit_info = {
	    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
	    .commandBufferCount = 1,
	    .pCommandBuffers = &current_cmd
	};

	// wait on previous semaphore if not first buffer
	VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	if (ctx->cmd_bufs_curr_idx > 0) {
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores =
		    &ctx->semaphores[ctx->cmd_bufs_curr_idx - 1];
		submit_info.pWaitDstStageMask = &wait_stage;
	}

	// signal next semaphore if we have room for more buffers
	if (ctx->cmd_bufs_curr_idx + 1 < ctx->cmd_bufs_len && !dont_signal_next) {
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores =
		    &ctx->semaphores[ctx->cmd_bufs_curr_idx];
	}

	res = vkQueueSubmit(ctx->compute_queue, 1, &submit_info, VK_NULL_HANDLE);
	if (res != VK_SUCCESS) {
		error_exit(dev, "failed to submit command buffer");
	}

	// advance to next buffer
	ctx->cmd_bufs_curr_idx++;
	ctx->curr_cmd_buf_num_recorded = 0;
}

// begins recording if not recording yet
static void _begin_next_buffer(tnn_device_t *dev) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

	if (ctx->currently_recording) {
		return; // already recording
	}

	VkCommandBuffer cmd = ctx->cmd_bufs[ctx->cmd_bufs_curr_idx];
	VkCommandBufferBeginInfo begin_info = {
	    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};
	VkResult res = vkBeginCommandBuffer(cmd, &begin_info);
	if (res != VK_SUCCESS) {
		error_exit(dev, "failed to begin command buffer");
	}
	ctx->currently_recording = true;
}

// call this when op is called, before recording cmds
// returns current buffer
// the caller should record at least a single operation afterwards
static VkCommandBuffer ensure_ready_for_recording(tnn_device_t *dev) {
	ensure_device_initialized(dev);
	_ensure_enough_cmd_bufs(dev);

	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

	// in case there's no active recording, start a new one
	_begin_next_buffer(dev);

	// in case there's enough ops, submit and start new recording
	if (ctx->curr_cmd_buf_num_recorded >= VK_MAX_CMD_BUF_LEN) {
		_end_submit_current_buffer(dev, false);
		_ensure_enough_cmd_bufs(dev);
		_begin_next_buffer(dev);
	}

	ctx->curr_cmd_buf_num_recorded++;
	return ctx->cmd_bufs[ctx->cmd_bufs_curr_idx];
}

// call this when cpu sync is required; ends any ongoing recording and resets
// resources for next execution cycle
static void sync_device(tnn_device_t *dev) {
	// end the current buffer if recording (and submit if anything recorded)
	// dont_signal_next=true makes sure the next semaphore is not signaled
	_end_submit_current_buffer(dev, true);

	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	VkResult res = vkQueueWaitIdle(ctx->compute_queue);
	if (res != VK_SUCCESS) {
		error_exit(dev, "failed to wait for queue idle");
	}

	// reset command pool to free all command buffers
	res = vkResetCommandPool(ctx->device, ctx->command_pool, 0);
	if (res != VK_SUCCESS) {
		error_exit(dev, "failed to reset command pool");
	}

	// reset descriptor pool
	res = vkResetDescriptorPool(ctx->device, ctx->descriptor_pool, 0);
	if (res != VK_SUCCESS) {
		error_exit(dev, "failed to reset descriptor pool");
	}

	// reset for next ops
	ctx->cmd_bufs_curr_idx = 0;
	ctx->curr_cmd_buf_num_recorded = 0;
	ctx->currently_recording = false;
}

// additional helpers

static VkDescriptorSet _bind_buffers(
    tnn_device_t *dev,
    vk_kernel_t *kernel,
    vk_buffer_t **buffers,
    uint32_t num_buffers
) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

	// allocate descriptor set
	VkDescriptorSetAllocateInfo alloc_info = {
	    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
	    .descriptorPool = ctx->descriptor_pool,
	    .descriptorSetCount = 1,
	    .pSetLayouts = &kernel->descriptor_set_layout,
	};
	VkDescriptorSet descriptor_set;
	vkAllocateDescriptorSets(ctx->device, &alloc_info, &descriptor_set);

	// prepare buffer infos and writes
	VkDescriptorBufferInfo *buffer_infos =
	    safe_malloc(num_buffers * sizeof(VkDescriptorBufferInfo));
	VkWriteDescriptorSet *writes =
	    safe_malloc(num_buffers * sizeof(VkWriteDescriptorSet));

	for (uint32_t i = 0; i < num_buffers; i++) {
		buffer_infos[i] = (VkDescriptorBufferInfo){
		    .buffer = buffers[i]->buffer,
		    .offset = 0,
		    .range = VK_WHOLE_SIZE,
		};

		writes[i] = (VkWriteDescriptorSet){
		    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		    .dstSet = descriptor_set,
		    .dstBinding = i,
		    .descriptorCount = 1,
		    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		    .pBufferInfo = &buffer_infos[i],
		};
	}

	vkUpdateDescriptorSets(ctx->device, num_buffers, writes, 0, NULL);

	free(buffer_infos);
	free(writes);

	return descriptor_set;
}

static void _record_compute_barriers(
    vk_buffer_t **buffers, uint32_t num_buffers, VkCommandBuffer cmd
) {
	VkBufferMemoryBarrier *buf_barriers =
	    safe_malloc(num_buffers * sizeof(VkBufferMemoryBarrier));
	for (uint32_t i = 0; i < num_buffers; i++) {
		VkBufferMemoryBarrier buf_barrier = {
		    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		    .pNext = NULL,
		    .srcAccessMask =
		        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
		        VK_ACCESS_TRANSFER_WRITE_BIT |
		        VK_ACCESS_TRANSFER_READ_BIT, // TODO: track last
		                                     // access type in vk_buffer
		    .dstAccessMask =
		        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .buffer = buffers[i]->buffer, // the VkBuffer handle
		    .offset = 0,
		    .size = VK_WHOLE_SIZE // or specific byte range if you want
		};
		buf_barriers[i] = buf_barrier;
	}
	vkCmdPipelineBarrier(
	    cmd,
	    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
	    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	    0,
	    0,
	    NULL,
	    num_buffers,
	    buf_barriers,
	    0,
	    NULL // no image barriers
	);
	free(buf_barriers);
}

static void dispatch_kernel(
    tnn_device_t *dev,
    vk_kernel_t *kernel,
    const uint8_t *spriv,
    size_t spirv_len,
    vk_buffer_t **buffers,
    uint32_t num_buffers,
    const void *push_constants,
    size_t push_size,
    uint32_t group_x,
    uint32_t group_y,
    uint32_t group_z,
    VkCommandBuffer cmd
) {
	if (kernel->shader_module == VK_NULL_HANDLE) {
		init_kernel(dev, kernel, num_buffers, push_size, spriv, spirv_len);
	}

	VkDescriptorSet desc_set = _bind_buffers(dev, kernel, buffers, num_buffers);
	_record_compute_barriers(buffers, num_buffers, cmd);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline);
	vkCmdBindDescriptorSets(
	    cmd,
	    VK_PIPELINE_BIND_POINT_COMPUTE,
	    kernel->pipeline_layout,
	    0,
	    1,
	    &desc_set,
	    0,
	    NULL
	);

	vkCmdPushConstants(
	    cmd,
	    kernel->pipeline_layout,
	    VK_SHADER_STAGE_COMPUTE_BIT,
	    0,
	    push_size,
	    push_constants
	);

	vkCmdDispatch(cmd, group_x, group_y, group_z);
}

// free context func

static void free_vk_context(tnn_device_t *dev) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

	if (ctx == NULL) {
		return; // never initialized
	}

	if (ctx->device == VK_NULL_HANDLE) {
		free(ctx);
		return; // never initialized
	}

	// wait for any pending work
	sync_device(dev);
	vkDeviceWaitIdle(ctx->device);

	// free semaphores
	for (size_t i = 1; i < ctx->cmd_bufs_len; i++) {
		vkDestroySemaphore(ctx->device, ctx->semaphores[i - 1], NULL);
	}
	free(ctx->semaphores);
	free(ctx->cmd_bufs); // the buffers themselves freed by pool destruction

	// destroy descriptor pool
	vkDestroyDescriptorPool(ctx->device, ctx->descriptor_pool, NULL);

	// destroy kernels
	size_t num_kernels =
	    sizeof(ctx->kernels.array) / sizeof(ctx->kernels.array[0]);
	for (size_t i = 0; i < num_kernels; ++i) {
		destroy_kernel(dev, &ctx->kernels.array[i]);
	}

	// destroy command pool (frees command buffers)
	vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);

	// destroy device
	vkDestroyDevice(ctx->device, NULL);

	free(ctx);
}

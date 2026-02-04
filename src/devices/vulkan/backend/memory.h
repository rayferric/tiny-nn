#pragma once

#include <memory.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "../impl.h"
#include <vulkan/vulkan_core.h>

static uint32_t find_memtype(
    VkPhysicalDevice physical_device,
    uint32_t type_filter,
    VkMemoryPropertyFlags properties
) {
	VkPhysicalDeviceMemoryProperties mem_properties;
	vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

	for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
		if ((type_filter & (1 << i)) &&
		    (mem_properties.memoryTypes[i].propertyFlags & properties) ==
		        properties) {
			return i;
		}
	}

	return UINT32_MAX;
}

// static VkCommandBuffer begin_cmds(tnn_device_t *dev) {
// 	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

// 	VkCommandBufferAllocateInfo alloc_info = {
// 	    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
// 	    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
// 	    .commandPool = ctx->command_pool,
// 	    .commandBufferCount = 1,
// 	};

// 	VkCommandBuffer command_buffer;
// 	vkAllocateCommandBuffers(ctx->device, &alloc_info, &command_buffer);

// 	VkCommandBufferBeginInfo begin_info = {
// 	    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
// 	    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
// 	};
// 	vkBeginCommandBuffer(command_buffer, &begin_info);

// 	return command_buffer;
// }

// static void end_cmds(tnn_device_t *dev, VkCommandBuffer command_buffer) {
// 	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

// 	vkEndCommandBuffer(command_buffer);

// 	VkSubmitInfo submit_info = {
// 	    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
// 	    .commandBufferCount = 1,
// 	    .pCommandBuffers = &command_buffer,
// 	};
// 	vkQueueSubmit(ctx->compute_queue, 1, &submit_info, VK_NULL_HANDLE);
// 	vkQueueWaitIdle(ctx->compute_queue);

// 	vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &command_buffer);
// }

static void create_staging_buf(
    tnn_device_t *dev,
    size_t size,
    VkBufferUsageFlags usage,
    VkBuffer *staging_buffer,
    VkDeviceMemory *staging_memory
) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

	// create buffer
	VkBufferCreateInfo buffer_info = {
	    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	    .size = size,
	    .usage = usage,
	    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	vkCreateBuffer(ctx->device, &buffer_info, NULL, staging_buffer);

	// allocate memory
	// TODO: also use allocator
	VkMemoryRequirements mem_requirements;
	vkGetBufferMemoryRequirements(
	    ctx->device, *staging_buffer, &mem_requirements
	);
	uint32_t memory_type_index = find_memtype(
	    ctx->physical_device,
	    mem_requirements.memoryTypeBits,
	    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
	);
	VkMemoryAllocateInfo alloc_info = {
	    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	    .allocationSize = mem_requirements.size,
	    .memoryTypeIndex = memory_type_index,
	};
	vkAllocateMemory(ctx->device, &alloc_info, NULL, staging_memory);
	vkBindBufferMemory(ctx->device, *staging_buffer, *staging_memory, 0);
}

static void
record_transfer_barriers(VkCommandBuffer cmd, VkBuffer src, VkBuffer dst) {
	VkBufferMemoryBarrier barriers[2];
	uint32_t barrier_count = 0;

	if (src != VK_NULL_HANDLE) {
		barriers[barrier_count++] = (VkBufferMemoryBarrier){
		    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		    .pNext = NULL,
		    .srcAccessMask =
		        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
		    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .buffer = src,
		    .offset = 0,
		    .size = VK_WHOLE_SIZE
		};
	}
	if (dst != VK_NULL_HANDLE) {
		barriers[barrier_count++] = (VkBufferMemoryBarrier){
		    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		    .pNext = NULL,
		    .srcAccessMask =
		        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
		        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
		    .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .buffer = dst,
		    .offset = 0,
		    .size = VK_WHOLE_SIZE
		};
	}
	if (barrier_count > 0) {
		vkCmdPipelineBarrier(
		    cmd,
		    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		    VK_PIPELINE_STAGE_TRANSFER_BIT,
		    0,
		    0,
		    NULL,
		    barrier_count,
		    barriers,
		    0,
		    NULL
		);
	}
}

static void *buf_alloc(tnn_device_t *dev, size_t bytes) {
	TNN_TRACY_ZONE_START();

	ensure_device_initialized(dev);
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;

	vk_buffer_t *vk_buf = safe_malloc(sizeof(vk_buffer_t));
	vk_buf->size = bytes;

	// create buffer
	VkBufferCreateInfo buffer_info = {
	    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	    .size = bytes,
	    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
	             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	if (vkCreateBuffer(ctx->device, &buffer_info, NULL, &vk_buf->buffer) !=
	    VK_SUCCESS) {
		error_exit(dev, "failed to create buffer");
	}

	// find memory type
	VkMemoryRequirements mem_requirements;
	vkGetBufferMemoryRequirements(
	    ctx->device, vk_buf->buffer, &mem_requirements
	); // what the buffer allows
	uint32_t memory_type_index = find_memtype(
	    ctx->physical_device,
	    mem_requirements.memoryTypeBits,
	    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT // what we want
	);
	if (memory_type_index == UINT32_MAX) {
		error_exit(dev, "failed to find suitable memory type");
	}

	// alloc memory
	// TODO: implement a buddy allocator
	VkMemoryAllocateInfo alloc_info = {
	    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	    .allocationSize = mem_requirements.size,
	    .memoryTypeIndex = memory_type_index,
	};
	if (vkAllocateMemory(ctx->device, &alloc_info, NULL, &vk_buf->memory) !=
	    VK_SUCCESS) {
		error_exit(dev, "failed to allocate buffer memory");
	}

	// bind memory to buffer
	vkBindBufferMemory(ctx->device, vk_buf->buffer, vk_buf->memory, 0);

	TNN_TRACY_ZONE_END();

	return vk_buf;
}

static void buf_free(tnn_device_t *dev, void *ptr) {
	if (!ptr) {
		return;
	}

	TNN_TRACY_ZONE_START();

	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	vk_buffer_t *vk_buf = (vk_buffer_t *)ptr;

	sync_device(dev); // TODO: queue frees until next sync instead

	vkDestroyBuffer(ctx->device, vk_buf->buffer, NULL);
	vkFreeMemory(ctx->device, vk_buf->memory, NULL);
	free(vk_buf);

	TNN_TRACY_ZONE_END();
}

static void buf_copy(tnn_device_t *dev, void *dst, const void *src, size_t sz) {
	vk_buffer_t *dst_buf = (vk_buffer_t *)dst;
	vk_buffer_t *src_buf = (vk_buffer_t *)src;

	VkCommandBuffer cmd = ensure_ready_for_recording(dev);

	record_transfer_barriers(cmd, src_buf->buffer, dst_buf->buffer);

	VkBufferCopy copy_region = {
	    .srcOffset = 0,
	    .dstOffset = 0,
	    .size = sz,
	};
	vkCmdCopyBuffer(cmd, src_buf->buffer, dst_buf->buffer, 1, &copy_region);
}

static void
buf_copy_to_host(tnn_device_t *dev, void *dst, const void *src, size_t sz) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	vk_buffer_t *src_buf = (vk_buffer_t *)src;

	// create staging buffer
	VkBuffer staging_buffer;
	VkDeviceMemory staging_memory;
	create_staging_buf(
	    dev,
	    sz,
	    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	    &staging_buffer,
	    &staging_memory
	);

	// copy to staging buffer
	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	record_transfer_barriers(cmd, src_buf->buffer, staging_buffer);
	VkBufferCopy copy_region = {
	    .srcOffset = 0,
	    .dstOffset = 0,
	    .size = sz,
	};
	vkCmdCopyBuffer(cmd, src_buf->buffer, staging_buffer, 1, &copy_region);

	sync_device(dev);

	// copy to host
	void *mapped;
	vkMapMemory(ctx->device, staging_memory, 0, sz, 0, &mapped);
	memcpy(dst, mapped, sz);
	vkUnmapMemory(ctx->device, staging_memory);

	// destroy staging buffer
	// all work is finished so it's a good moment to free now
	vkDestroyBuffer(ctx->device, staging_buffer, NULL);
	vkFreeMemory(ctx->device, staging_memory, NULL);
}

static void
buf_copy_to_device(tnn_device_t *dev, void *dst, const void *src, size_t sz) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	vk_buffer_t *dst_buf = (vk_buffer_t *)dst;

	// create staging buffer
	VkBuffer staging_buffer;
	VkDeviceMemory staging_memory;
	create_staging_buf(
	    dev,
	    sz,
	    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	    &staging_buffer,
	    &staging_memory
	);

	// copy from host to staging buffer
	void *mapped;
	vkMapMemory(ctx->device, staging_memory, 0, sz, 0, &mapped);
	memcpy(mapped, src, sz);
	vkUnmapMemory(ctx->device, staging_memory);

	// copy from staging buffer to device buffer
	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	record_transfer_barriers(cmd, VK_NULL_HANDLE, dst_buf->buffer);
	VkBufferCopy copy_region = {
	    .srcOffset = 0,
	    .dstOffset = 0,
	    .size = sz,
	};
	vkCmdCopyBuffer(cmd, staging_buffer, dst_buf->buffer, 1, &copy_region);

	// now need sync before destroying the staging buffer
	sync_device(dev);

	// destroy staging buffer
	vkDestroyBuffer(ctx->device, staging_buffer, NULL);
	vkFreeMemory(ctx->device, staging_memory, NULL);

	// TODO: this destruction should be queued until next sync instead
}

static void buf_fill_f(tnn_device_t *dev, void *dst, float value, size_t n) {
	vk_device_context_t *ctx = (vk_device_context_t *)dev->_ctx;
	vk_buffer_t *vk_buf = (vk_buffer_t *)dst;

	VkCommandBuffer cmd = ensure_ready_for_recording(dev);
	record_transfer_barriers(cmd, VK_NULL_HANDLE, vk_buf->buffer);

	vkCmdFillBuffer(
	    cmd, vk_buf->buffer, 0, n * sizeof(float), *((uint32_t *)&value)
	);
}

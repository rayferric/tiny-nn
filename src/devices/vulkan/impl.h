#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <vulkan/vulkan.h>

#include <tnn/tnn.h>

#include "../../util/safe_malloc.h"

typedef struct {
	VkPhysicalDevice physical_device;
	VkDevice device;
	VkQueue compute_queue;
	uint32_t queue_family_index;
	VkCommandPool command_pool;
} vk_device_context_t;

typedef struct {
	VkBuffer buffer;
	VkDeviceMemory memory;
	size_t size;
} vk_buffer_t;

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
		fprintf(
		    stderr,
		    "no compute queue family found for %s - %s",
		    dev->name,
		    dev->desc
		);
		exit(1);
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
	VkResult result = vkCreateDevice(
	    ctx->physical_device, &device_create_info, NULL, &ctx->device
	);
	if (result != VK_SUCCESS) {
		fprintf(
		    stderr,
		    "failed to create logical device for %s - %s",
		    dev->name,
		    dev->desc
		);
		exit(1);
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
	result =
	    vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->command_pool);
	if (result != VK_SUCCESS) {
		fprintf(
		    stderr,
		    "failed to create command pool for %s - %s",
		    dev->name,
		    dev->desc
		);
		exit(1);
	}
}

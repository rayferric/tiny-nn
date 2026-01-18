#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <vulkan/vulkan.h>

#include <tnn/tnn.h>

#include "./vulkan.h"

typedef struct {
	VkInstance instance;
	tnn_device_t *devices;
	size_t num_devices;
} vk_globals_t;
vk_globals_t vk_globals = {0};

void *vk_alloc(tnn_device_t *dev, size_t bytes) {
	return 0;
}
void vk_free(tnn_device_t *dev, void *ptr) {}
void vk_buf_copy_to_host(
    tnn_device_t *dev, void *dst, void *src, size_t bytes
) {}
void vk_buf_copy_to_device(
    tnn_device_t *dev, void *dst, void *src, size_t bytes
) {}
static _tnn_device_ops_t vk_device_ops = {
    .buf_alloc = vk_alloc,
    .buf_free = vk_free,
    .buf_copy_to_host = vk_buf_copy_to_host,
    .buf_copy_to_device = vk_buf_copy_to_device,
};

int vk_init() {
	return 0;
}
void vk_terminate() {
	if (vk_globals.instance) {
		vkDestroyInstance(vk_globals.instance, NULL);
		vk_globals.instance = NULL;
	}
	if (vk_globals.devices) {
		free(vk_globals.devices);
		vk_globals.devices = NULL;
		vk_globals.num_devices = 0;
	}
}

size_t vk_list_devices(tnn_device_t **out_devs) {
	if (vk_globals.instance == NULL) {
		VkApplicationInfo app_info = {
		    .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		    .pApplicationName = "TinyNN",
		    .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		    .apiVersion = VK_API_VERSION_1_0,
		};
		VkInstanceCreateInfo create_info = {
		    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		    .pApplicationInfo = &app_info,
		};
		VkResult result =
		    vkCreateInstance(&create_info, NULL, &vk_globals.instance);
		if (result != VK_SUCCESS) {
			vk_globals.instance = NULL;
			return 0;
		}
	}

	uint32_t num_devices_u32 = 0;
	vkEnumeratePhysicalDevices(vk_globals.instance, &num_devices_u32, NULL);
	if (num_devices_u32 == 0) {
		return 0;
	}

	if (vk_globals.devices) {
		free(vk_globals.devices);
	}
	vk_globals.devices = calloc(num_devices_u32, sizeof(tnn_device_t));
	if (!vk_globals.devices) {
		fprintf(stderr, "vk_list_devices: out of memory\n");
		exit(1);
	}
	vk_globals.num_devices = num_devices_u32;

	VkPhysicalDevice *devices =
	    malloc(num_devices_u32 * sizeof(VkPhysicalDevice));
	vkEnumeratePhysicalDevices(vk_globals.instance, &num_devices_u32, devices);

	for (uint32_t i = 0; i < num_devices_u32; i++) {
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(devices[i], &props);

		if (num_devices_u32 > 1) {
			snprintf(
			    vk_globals.devices[i].name,
			    TNN_MAX_DEVICE_NAME_LENGTH,
			    "vulkan:%u",
			    i
			);
		} else {
			snprintf(
			    vk_globals.devices[i].name, TNN_MAX_DEVICE_NAME_LENGTH, "vulkan"
			);
		}
		snprintf(
		    vk_globals.devices[i].desc,
		    TNN_MAX_DEVICE_DESC_LENGTH,
		    "%s",
		    props.deviceName
		);
		vk_globals.devices[i]._ops = vk_device_ops;
		vk_globals.devices[i]._ctx = NULL;
		vk_globals.devices[i]._is_cpu = false;

		if (out_devs) {
			out_devs[i] = &vk_globals.devices[i];
		}
	}

	free(devices);
	return num_devices_u32;
}

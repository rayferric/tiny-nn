#ifdef ENABLE_VULKAN_BACKEND

#include "./vulkan.h"

#include <memory.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <vulkan/vulkan.h>

#include <tnn/tnn.h>

#include "../../util/safe_malloc.h"

#include "./backend/backend.h"
#include "./impl.h"

typedef struct {
	VkInstance instance;
	tnn_device_t *devices;
	size_t num_devices;
#ifdef ENABLE_VULKAN_VALIDATION
	VkDebugUtilsMessengerEXT debug_messenger;
#endif
} vk_globals_t;
vk_globals_t vk_globals;

#ifdef ENABLE_VULKAN_VALIDATION
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT *data,
    void *user
) {
	if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
		fprintf(stderr, "[VK] %s\n", data->pMessage);
	}
	raise(SIGTRAP);
	return VK_FALSE;
}
#endif

int vk_init() {
	memset(&vk_globals, 0, sizeof(vk_globals));
	return 0;
}
void vk_terminate() {
	if (vk_globals.devices) {
		for (size_t i = 0; i < vk_globals.num_devices; i++) {
			free_vk_context(&vk_globals.devices[i]);
		}
		free(vk_globals.devices);
		vk_globals.devices = NULL;
		vk_globals.num_devices = 0;
	}
	if (vk_globals.instance) {
#ifdef ENABLE_VULKAN_VALIDATION
		if (vk_globals.debug_messenger) {
			PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger =
			    (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
			        vk_globals.instance, "vkDestroyDebugUtilsMessengerEXT"
			    );

			if (destroy_messenger) {
				destroy_messenger(
				    vk_globals.instance, vk_globals.debug_messenger, NULL
				);
			}
		}
#endif

		vkDestroyInstance(vk_globals.instance, NULL);
		vk_globals.instance = NULL;
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

#ifdef ENABLE_VULKAN_VALIDATION
		VkDebugUtilsMessengerCreateInfoEXT debug_info = {
		    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
		    .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
		                       VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
		    .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
		                   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
		                   VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
		    .pfnUserCallback = debug_callback,
		};
		const char *layers[] = {"VK_LAYER_KHRONOS_validation"};
		const char *extensions[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
#endif

		VkInstanceCreateInfo create_info = {
		    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		    .pApplicationInfo = &app_info,
#ifdef ENABLE_VULKAN_VALIDATION
		    .pNext = &debug_info,
		    .enabledLayerCount = 1,
		    .ppEnabledLayerNames = layers,
		    .enabledExtensionCount = 1,
		    .ppEnabledExtensionNames = extensions,
#endif
		};

		VkResult result =
		    vkCreateInstance(&create_info, NULL, &vk_globals.instance);
		if (result != VK_SUCCESS) {
			vk_globals.instance = NULL;
			return 0;
		}

#ifdef ENABLE_VULKAN_VALIDATION
		PFN_vkCreateDebugUtilsMessengerEXT create_messenger =
		    (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
		        vk_globals.instance, "vkCreateDebugUtilsMessengerEXT"
		    );
		if (create_messenger) {
			create_messenger(
			    vk_globals.instance,
			    &debug_info,
			    NULL,
			    &vk_globals.debug_messenger
			);
		}
#endif
	}

	if (vk_globals.devices) {
		for (uint32_t i = 0; i < vk_globals.num_devices; i++) {
			if (out_devs) {
				out_devs[i] = &vk_globals.devices[i];
			}
		}
		return vk_globals.num_devices;
	}

	uint32_t num_devices_u32 = 0;
	vkEnumeratePhysicalDevices(vk_globals.instance, &num_devices_u32, NULL);
	if (num_devices_u32 == 0) {
		return 0;
	}
	if (num_devices_u32 > TNN_MAX_NUM_DEVICES) {
		num_devices_u32 = TNN_MAX_NUM_DEVICES;
	}

	vk_globals.devices = safe_malloc(num_devices_u32 * sizeof(tnn_device_t));
	vk_globals.num_devices = num_devices_u32;

	VkPhysicalDevice *devices =
	    safe_malloc(num_devices_u32 * sizeof(VkPhysicalDevice));
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

		vk_device_context_t *ctx = safe_malloc(sizeof(vk_device_context_t));
		memset(ctx, 0, sizeof(vk_device_context_t));
		ctx->physical_device = devices[i];
		vk_globals.devices[i]._ctx = ctx;
		vk_globals.devices[i]._backend = backend;
		vk_globals.devices[i]._is_cpu = false;

		if (out_devs) {
			out_devs[i] = &vk_globals.devices[i];
		}
	}

	free(devices);
	return vk_globals.num_devices;
}

#endif

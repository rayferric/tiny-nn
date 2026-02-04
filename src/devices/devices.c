#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tnn/tnn.h>

#include "../util/safe_malloc.h"

#include "./cpu/cpu.h"
#include "./devices.h"
#ifdef ENABLE_VULKAN_BACKEND
#include "./vulkan/vulkan.h"
#endif

device_globals_t device_globals;

#define CHAIN_INIT(fn)                                                         \
	do {                                                                       \
		int status = fn();                                                     \
		if (status != 0)                                                       \
			return status;                                                     \
	} while (0)
int devices_init() {
	device_globals.default_device = NULL;
	device_globals.devices_cache = NULL;

	CHAIN_INIT(cpu_init);
#ifdef ENABLE_VULKAN_BACKEND
	CHAIN_INIT(vk_init);
#endif

	// set default to cpu device
	tnn_device_t *devices[64];
	size_t device_count = cpu_list_devices(devices);
	if (device_count > 0) {
		device_globals.default_device = devices[0];
	}

	return 0;
}
void devices_terminate() {
	device_globals.default_device = NULL;

	// free cache
	if (device_globals.devices_cache) {
		free(device_globals.devices_cache);
		device_globals.devices_cache = NULL;
	}

	cpu_terminate();
#ifdef ENABLE_VULKAN_BACKEND
	vk_terminate();
#endif
}

size_t tnn_list_devices(tnn_device_t **out_devs) {
	// build cache if needed
	if (device_globals.devices_cache == NULL) {
		size_t count = 0;
		count += cpu_list_devices(NULL);
#ifdef ENABLE_VULKAN_BACKEND
		count += vk_list_devices(NULL);
#endif

		device_globals.num_devices = count;
		device_globals.devices_cache =
		    safe_malloc(count * sizeof(tnn_device_t *));

		size_t offset = 0;
		offset += cpu_list_devices(device_globals.devices_cache);
#ifdef ENABLE_VULKAN_BACKEND
		offset += vk_list_devices(device_globals.devices_cache + offset);
#endif
	}

	// copy from cache to caller's buffer
	if (out_devs) {
		memcpy(
		    out_devs,
		    device_globals.devices_cache,
		    device_globals.num_devices * sizeof(tnn_device_t *)
		);
	}

	return device_globals.num_devices;
}

tnn_device_t *tnn_get_cpu() {
	tnn_list_devices(NULL);
	return device_globals.devices_cache[0];
}

tnn_device_t *tnn_find_device(const char *name) {
	tnn_device_t *devices[TNN_MAX_NUM_DEVICES];
	size_t device_count = tnn_list_devices(devices);

	if (name == NULL) {
		// fallback to cpu
		return devices[0];
	}

	for (size_t i = 0; i < device_count; i++) {
		if (strncmp(devices[i]->name, name, strlen(name)) == 0) {
			return devices[i];
		}
	}

	return devices[0];
}

void tnn_set_default_device(tnn_device_t *device) {
	device_globals.default_device = device;
}

tnn_device_t *tnn_get_default_device() {
	return device_globals.default_device;
}

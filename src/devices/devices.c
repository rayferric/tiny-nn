#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tnn/tnn.h>

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

	cpu_terminate();
#ifdef ENABLE_VULKAN_BACKEND
	vk_terminate();
#endif
}

size_t tnn_list_devices(tnn_device_t **out_devs) {
	size_t count = 0;

	count += cpu_list_devices(out_devs);
#ifdef ENABLE_VULKAN_BACKEND
	count += vk_list_devices(out_devs ? out_devs + count : NULL);
#endif

	return count;
}

void tnn_set_default_device(const char *name) {
	if (name == NULL) {
		device_globals.default_device = NULL;
		return;
	}

	tnn_device_t *devices[TNN_MAX_NUM_DEVICES];
	size_t device_count = tnn_list_devices(devices);

	for (size_t i = 0; i < device_count; i++) {
		if (strncmp(devices[i]->name, name, strlen(name)) == 0) {
			device_globals.default_device = devices[i];
			return;
		}
	}

	// dev not found, fallback to cpu
	device_globals.default_device = devices[0];
}

const char *tnn_get_default_device() {
	return device_globals.default_device->name;
}

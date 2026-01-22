#pragma once

#include <tnn/tnn.h>

typedef struct {
	tnn_device_t *default_device;
	tnn_device_t **devices_cache;
	size_t num_devices;
} device_globals_t;
extern device_globals_t device_globals;

int devices_init();
void devices_terminate();

tnn_device_t *find_device(const char *name);

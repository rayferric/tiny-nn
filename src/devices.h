#pragma once

#include <tnn/tnn.h>

typedef struct {
	tnn_device_t *default_device;
} device_globals_t;
extern device_globals_t device_globals;

int devices_init();
void devices_terminate();

void tnn_set_default_device(const char *name);
const char *tnn_get_default_device();

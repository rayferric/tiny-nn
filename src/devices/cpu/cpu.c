#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tnn/tnn.h>

#include "./cpu.h"

#include "./backend/backend.h"

typedef struct {
	tnn_device_t device;
} cpu_globals_t;
cpu_globals_t cpu_globals;

int cpu_init() {
	return 0;
}
void cpu_terminate() {}

size_t cpu_list_devices(tnn_device_t **out_devs) {
	tnn_device_t *cpu_device = &cpu_globals.device;

	snprintf(cpu_device->name, TNN_MAX_DEVICE_NAME_LENGTH, "cpu");

#if defined(__linux__)
	char model[128] = "";

	FILE *f = fopen("/proc/cpuinfo", "r");
	if (f) {
		char line[256];
		while (fgets(line, sizeof(line), f)) {
			if (strncmp(line, "model name", 10) == 0) {
				char *colon = strchr(line, ':');
				if (colon) {
					char *start = colon + 2;
					while (*start == ' ') {
						start++;
					}
					snprintf(model, sizeof(model), "%s", start);
					model[strcspn(model, "\n")] = '\0';
					break;
				}
			}
		}
		fclose(f);
	}

	if (model[0] != '\0') {
		snprintf(cpu_device->desc, TNN_MAX_DEVICE_DESC_LENGTH, "%s", model);
	} else {
		snprintf(cpu_device->desc, TNN_MAX_DEVICE_DESC_LENGTH, "System CPU");
	}
#else
	snprintf(cpu_device->desc, TNN_MAX_DEVICE_DESC_LENGTH, "System CPU");
#endif

	cpu_device->_backend = backend;
	cpu_device->_ctx = NULL;
	cpu_device->_is_cpu = true;

	if (out_devs) {
		*out_devs = cpu_device;
	}
	return 1;
}

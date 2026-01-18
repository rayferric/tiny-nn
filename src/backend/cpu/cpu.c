#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tnn/tnn.h>

#include "./cpu.h"

void *cpu_buf_alloc(tnn_device_t *, size_t);
void cpu_buf_free(tnn_device_t *, void *);
void cpu_buf_copy(tnn_device_t *, void *, const void *, size_t);

tnn_tensor_t *cpu_proj(tnn_tensor_t *, size_t);
tnn_tensor_t *cpu_bias(tnn_tensor_t *);
tnn_tensor_t *cpu_relu(tnn_tensor_t *);
tnn_tensor_t *cpu_cross_entropy(tnn_tensor_t *, tnn_tensor_t *);
tnn_tensor_t *cpu_conv(tnn_tensor_t *, size_t, size_t, size_t, size_t);
tnn_tensor_t *cpu_bn(tnn_tensor_t *, float, bool);
tnn_tensor_t *cpu_add(tnn_tensor_t *, tnn_tensor_t *);
tnn_tensor_t *cpu_mean(tnn_tensor_t *, size_t, size_t);
tnn_tensor_t *cpu_reshape(tnn_tensor_t *, const size_t *, size_t);

_tnn_device_ops_t cpu_device_ops = {
    .buf_alloc = cpu_buf_alloc,
    .buf_free = cpu_buf_free,
    .buf_copy = cpu_buf_copy,
    .buf_copy_to_host = NULL,
    .buf_copy_to_device = NULL,
    //
    .proj = cpu_proj,
    .bias = cpu_bias,
    .relu = cpu_relu,
    .cross_entropy = cpu_cross_entropy,
    .conv = cpu_conv,
    .bn = cpu_bn,
    .add = cpu_add,
    .mean = cpu_mean,
    .reshape = cpu_reshape,
};

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

	cpu_device->_ops = cpu_device_ops;
	cpu_device->_ctx = NULL;
	cpu_device->_is_cpu = true;

	if (out_devs) {
		*out_devs = cpu_device;
	}
	return 1;
}

#pragma once

#include <tnn/tnn.h>

int cpu_init();
void cpu_terminate();

size_t cpu_list_devices(tnn_device_t **out_devs);

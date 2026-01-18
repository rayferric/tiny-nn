#pragma once

#include <tnn/tnn.h>

int vk_init();
void vk_terminate();

size_t vk_list_devices(tnn_device_t **out_devs);

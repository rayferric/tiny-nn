#include <stdio.h>

#include <tnn/tnn.h>

int main() {
	if (tnn_init()) {
		fprintf(stderr, "failed to initialize tnn\n");
		return 1;
	}

	size_t num_devices = tnn_list_devices(NULL);
	tnn_device_t **devs = malloc(num_devices * sizeof(tnn_device_t *));
	tnn_list_devices(devs);
	printf("Available devices:\n");
	for (size_t i = 0; i < num_devices; i++) {
		printf("- %s - %s\n", devs[i]->name, devs[i]->desc);
	}
	free(devs);
	printf("Current device: %s\n", tnn_get_default_device());
	tnn_set_default_device("vulkan");
	printf("Current device: %s\n", tnn_get_default_device());
	// tnn_select_device(
	//     "gpu"
	// ); // falls back to "cpu" in case no such device found
	// // future tensors will be created on this device

	tnn_terminate();

	return 0;
}

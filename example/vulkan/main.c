#include <math.h>
#include <stdio.h>

#include <tnn/tnn.h>

int main() {
	if (tnn_init()) {
		fprintf(stderr, "failed to initialize tnn\n");
		return 1;
	}

	// size_t num_devices = tnn_list_devices(NULL);
	// tnn_device_t **devs = malloc(num_devices * sizeof(tnn_device_t *));
	// tnn_list_devices(devs);
	// printf("Available devices:\n");
	// for (size_t i = 0; i < num_devices; i++) {
	// 	printf("- %s - %s\n", devs[i]->name, devs[i]->desc);
	// }
	// free(devs);

	// printf("Current device: %s\n", tnn_get_default_device());
	// tnn_set_default_device("vulkan");
	printf("Current device: %s\n", tnn_get_default_device());

	// get out data from test op
	tnn_tensor_t *a = tnn_alloc((size_t[]){256}, 1);
	tnn_init_randn(a);
	tnn_tensor_t *c = tnn_proj(a, 128);
	TNN_SCOPE("test") {
		c = tnn_proj(c, 256);
	}
	c = tnn_ce(c, c);
	float *c_data = malloc(1 * sizeof(float));
	c->dev->_backend.buf_copy_to_host(
	    c->dev, c_data, c->data, 1 * sizeof(float)
	);

	a = tnn_detach(a, "vulkan");
	tnn_free(c); // frees cpu a

	// run the op again on the same data, compare results in c loop
	tnn_tensor_t *c2;
	c2 = tnn_proj(a, 128);
	TNN_SCOPE("test") {
		c2 = tnn_proj(c2, 256);
	}
	c2 = tnn_ce(c2, c2);
	float error = 0.0f;
	float *c2_data = malloc(1 * sizeof(float));
	c2->dev->_backend.buf_copy_to_host(
	    c2->dev, c2_data, c2->data, 1 * sizeof(float)
	);
	for (size_t i = 0; i < 1; i++) {
		error += fabsf(c_data[i] - c2_data[i]);
	}
	printf("Total error: %.6f\n", error);

	tnn_free(c2);
	free(c_data);
	free(c2_data);
	tnn_terminate();

	return 0;
}

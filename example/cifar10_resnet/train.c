#include <stdio.h>

#include <tnn/tnn.h>

#include "./accuracy.h"
#include "./cifar10.h"
#include "./resnet.h"

int main() {
	if (tnn_init()) {
		fprintf(stderr, "failed to initialize tnn\n");
		return 1;
	}

	cifar10_t cifar;
	cifar10_create(&cifar);
	for (int i = 1; i <= 5; i++) {
		char filename[64];
		snprintf(
		    filename, sizeof(filename), "data/cifar10/data_batch_%d.bin", i
		);
		if (cifar10_load(&cifar, filename) != CIFAR10_LOAD_OK) {
			return 1;
		}
	}
	printf("Loaded CIFAR-10\n");

	tnn_set_default_device(tnn_find_device("vulkan"));
	printf("Current device: %s\n", tnn_get_default_device()->name);

	const size_t num_epochs = 1;
	const size_t batch_size = 100;
	const size_t num_steps = cifar.num_imgs / batch_size;
	for (int i_epoch = 0; i_epoch < num_epochs; i_epoch++) {
		float total_loss = 0.0f;
		float total_acc = 0.0f;
		float total_samples = 0.0f;
		for (int i = 0; i < num_steps; i++) {
			TNN_TRACY_ZONE_START("step");
			tnn_tensor_t *x, *y;
			cifar10_make_batch(&cifar, i * batch_size, batch_size, &x, &y);

			tnn_tensor_t *y_pred = resnet(x, CIFAR10_NUM_LABELS, 16, 4, 3);
			tnn_tensor_t *loss = tnn_ce(y_pred, y);

			tnn_zero_grad();
			tnn_backward(loss);
			tnn_adamw();

			total_loss += tnn_item(loss);
			total_acc += accuracy(y_pred, y);
			total_samples++;

			if (i % 10 == 0) {
				printf("\nStep %d/%zu: ", i + 1, num_steps);
				printf("loss=%.4f", total_loss / total_samples);
				printf(
				    ", accuracy=%.2f%%", (total_acc / total_samples) * 100.0f
				);
				fflush(stdout);

				total_loss = 0.0f;
				total_acc = 0.0f;
				total_samples = 0.0f;
			}

			tnn_free(loss);
			TNN_TRACY_ZONE_END();
			TNN_TRACY_FRAME_END();
		}
	}

	tnn_drop_state("adamw");
	tnn_save("cifar100_mlp.tnn");

	char *keys[TNN_LIST_STATE_KEYS_MAX_LENGTH];
	size_t num_keys = tnn_list_state_keys(keys);
	printf("\n\nState keys:\n");
	for (size_t i = 0; i < num_keys; i++) {
		printf("- %s\n", keys[i]);
	}

	cifar10_destroy(&cifar);
	tnn_terminate();

	return 0;
}

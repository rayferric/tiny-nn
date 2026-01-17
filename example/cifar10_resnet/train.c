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
	printf("cifar loaded\n");
	fflush(stdout);

	const size_t num_epochs = 1;
	const size_t batch_size = 100;
	const size_t num_steps = cifar.num_imgs / batch_size;
	for (int i_epoch = 0; i_epoch < num_epochs; i_epoch++) {
		for (int i = 0; i < num_steps; i++) {
			tnn_tensor_t *x, *y;
			cifar10_make_batch(&cifar, i * batch_size, batch_size, &x, &y);

			tnn_tensor_t *y_pred = resnet(x, CIFAR10_NUM_LABELS, 8, 1, 1);
			tnn_tensor_t *loss = tnn_cross_entropy(y_pred, y);

			tnn_zero_grad();
			tnn_backward(loss);
			tnn_adamw();

			if (i % 10 == 0) {
				printf("\nStep %d/%zu: ", i + 1, num_steps);
				printf("loss=");
				tnn_print(loss);
				float acc = accuracy(y_pred, y);
				printf(", accuracy=%.2f%%", acc * 100.0f);
				fflush(stdout);
			}

			tnn_free(loss);
		}
	}

	tnn_drop_state("adamw");
	char *keys[1024];
	size_t num_keys = tnn_list_state_keys(keys);
	printf("\n\nState keys:\n");
	for (size_t i = 0; i < num_keys; i++) {
		printf("- %s\n", keys[i]);
	}

	tnn_save("cifar100_mlp.tnn");

	cifar10_destroy(&cifar);

	tnn_terminate();

	return 0;
}

#include <stdio.h>

#include <tnn/tnn.h>

#include "./accuracy.h"
#include "./cifar100.h"
#include "./resnet.h"

int main() {
	if (tnn_init()) {
		fprintf(stderr, "failed to initialize tnn\n");
		return 1;
	}

	cifar100_t cifar;
	cifar100_create(&cifar);
	if (cifar100_load(&cifar, "data/cifar100-train.bin") != CIFAR100_LOAD_OK) {
		return 1;
	}
	printf("cifar loaded\n");
	fflush(stdout);

	const size_t num_epochs = 1;
	const size_t num_steps = 1000;
	const size_t batch_size = 20;
	for (int i_epoch = 0; i_epoch < num_epochs; i_epoch++) {
		for (int i = 0; i < num_steps; i++) {
			tnn_tensor_t *x, *y;
			cifar100_make_batch(
			    &cifar, i * batch_size, batch_size, &x, NULL, &y
			);
			// x = tnn_reshape(
			//     x,
			//     (size_t[]){
			//         0, CIFAR100_HEIGHT * CIFAR100_WIDTH * CIFAR100_CHANNELS
			//     },
			//     2
			// );

			// tnn_tensor_t *y_pred =
			//     mlp(x,
			//         MLP_CFG(.dim_out = CIFAR100_NUM_CATEGORIES,
			//                 .dim_hidden = 128,
			//                 .num_hidden = 2));
			tnn_tensor_t *y_pred = resnet(x, CIFAR100_NUM_CATEGORIES, 8, 1, 1);
			tnn_tensor_t *loss = tnn_cross_entropy(y_pred, y);

			tnn_zero_grad();
			tnn_backward(loss);
			tnn_adamw();

			printf("\nStep %d/%zu: ", i + 1, num_steps);
			printf("loss=");
			tnn_print(loss);
			float acc = accuracy(y_pred, y);
			printf(", accuracy=%.2f%%", acc * 100.0f);
			fflush(stdout);

			tnn_free(loss);
		}
	}

	char *keys[1024];
	size_t num_keys = tnn_list_state_keys(keys);
	printf("\n\nState keys:\n");
	for (size_t i = 0; i < num_keys; i++) {
		printf("- %s\n", keys[i]);
	}

	tnn_drop_state("adamw");
	tnn_save("cifar100_mlp.tnn");

	cifar100_destroy(&cifar);

	tnn_terminate();

	return 0;
}

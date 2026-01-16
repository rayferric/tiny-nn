#include <stdio.h>

#include <tnn/tnn.h>

#include "./accuracy.h"
#include "./mlp.h"
#include "./mnist.h"

int main() {
	if (tnn_init()) {
		fprintf(stderr, "failed to initialize tnn\n");
		return 1;
	}

	mnist_t mnist;
	mnist_create(&mnist);
	mnist_load(
	    &mnist,
	    "/home/rayferric/Source/tiny-nn/data/train-images.idx3-ubyte",
	    "/home/rayferric/Source/tiny-nn/data/train-labels.idx1-ubyte"
	);

	mlp_cfg_t mlp_cfg = {.dim_out = 10, .dim_hidden = 128, .num_hidden = 2};

	const size_t num_epochs = 1;
	const size_t num_steps = 50;
	const size_t batch_size = 100;
	for (int i_epoch = 0; i_epoch < num_epochs; i_epoch++) {
		for (int i = 0; i < num_steps; i++) {
			tnn_tensor_t *x =
			    mnist_batch_images(&mnist, i * batch_size, batch_size);
			tnn_tensor_t *y =
			    mnist_batch_labels(&mnist, i * batch_size, batch_size);

			tnn_tensor_t *y_pred = mlp(x, mlp_cfg);
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
	tnn_save("mnist_mlp.tnn");

	mnist_destroy(&mnist);

	tnn_terminate();

	return 0;
}

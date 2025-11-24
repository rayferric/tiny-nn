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
	    "../data/train-images.idx3-ubyte",
	    "../data/train-labels.idx1-ubyte"
	);

	mlp_cfg_t mlp_cfg =
	    MLP_CFG(.dim_out = 10, .dim_hidden = 128, .num_hidden = 2);
	tnn_adamw_cfg_t adamw_cfg = TNN_ADAMW_CFG();

	const size_t num_epochs = 1;
	const size_t num_steps = 60;
	const size_t batch_size = 100;
	for (int i_epoch = 0; i_epoch < num_epochs; i_epoch++) {
		printf("\nepoch %d/%zu:", i_epoch + 1, num_epochs);

		for (int i = 0; i < num_steps; i++) {
			tnn_tensor_t *x =
			    mnist_batch_images(&mnist, i * batch_size, batch_size);
			tnn_tensor_t *y =
			    mnist_batch_labels(&mnist, i * batch_size, batch_size);

			tnn_tensor_t *y_pred = mlp(x, mlp_cfg);
			tnn_tensor_t *loss = tnn_cross_entropy(y_pred, y);

			tnn_zero_grad();
			tnn_backward(loss);
			tnn_adamw(adamw_cfg);

			printf("\nloss: ");
			tnn_print(loss);

			// calc accuracy
			float acc = accuracy(y_pred, y);
			printf(" accuracy: %.2f%%", acc * 100.0f);

			tnn_free(loss, TNN_INPUT | TNN_OUTPUT);
		}
	}

	// debug: list parameters
	size_t num_keys = tnn_params(NULL);
	char **keys = malloc(num_keys * sizeof(char *));
	tnn_params(keys);
	printf("\n\nParameter table:\n");
	for (size_t i = 0; i < num_keys; i++) {
		printf("- %s\n", keys[i]);
	}
	free(keys);

	tnn_drop("adamw");
	tnn_save("mnist_mlp.tnn");

	tnn_terminate();

	return 0;
}

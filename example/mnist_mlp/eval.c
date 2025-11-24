#include <tnn/tnn.h>

#include "./accuracy.h"
#include "./mlp.h"
#include "./mnist.h"

int main() {
	if (tnn_init()) {
		fprintf(stderr, "failed to initialize tnn\n");
		return 1;
	}

	tnn_load("mnist_mlp.tnn");

	// debug: list parameters
	size_t num_keys = tnn_params(NULL);
	char **keys = malloc(num_keys * sizeof(char *));
	tnn_params(keys);
	printf("Parameter table:\n");
	for (size_t i = 0; i < num_keys; i++) {
		printf("- %s\n", keys[i]);
	}
	free(keys);

	mlp_cfg_t mlp_cfg =
	    MLP_CFG(.dim_out = 10, .dim_hidden = 128, .num_hidden = 2);

	// compute accuracy on 10k test samples
	mnist_t mnist;
	mnist_create(&mnist);
	mnist_load(
	    &mnist,
	    "../data/train-images.idx3-ubyte",
	    "../data/train-labels.idx1-ubyte"
	);
	size_t batch_size = 100;
	size_t num_batches = 100;
	size_t total_correct = 0;
	for (size_t i = 0; i < num_batches; i++) {
		tnn_tensor_t *x =
		    mnist_batch_images(&mnist, i * batch_size, batch_size);
		tnn_tensor_t *y =
		    mnist_batch_labels(&mnist, i * batch_size, batch_size);

		tnn_tensor_t *y_pred = mlp(x, mlp_cfg);
		float acc = accuracy(y_pred, y);
		total_correct += (size_t)(acc * batch_size);

		tnn_free(x, TNN_INPUT);
		tnn_free(y, TNN_INPUT);
		tnn_free(y_pred, TNN_OUTPUT);
	}

	printf("Train accuracy: %.2f%%\n", (float)total_correct / 10000 * 100.0f);

	tnn_terminate();

	return 0;
}

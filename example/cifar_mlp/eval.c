#include <stdio.h>

#include <tnn/tnn.h>

#include "./mlp.h"

int main() {
	if (tnn_init()) {
		fprintf(stderr, "Failed to initialize TNN.\n");
		return 1;
	}

	tnn_mlp_cfg_t mlp_cfg =
	    TNN_MLP_CFG(.dim_out = 10, .dim_hidden = 128, .num_hidden = 2);

	tnn_load("cifar_mlp.tnn");

	const size_t num_steps = 60;
	const size_t batch_size = 100;
	float loss_sum = 0.0f;
	for (int i = 0; i < num_steps; i++) {
		tnn_vec_t *x, *y; // ...

		tnn_vec_t *y_pred = tnn_mlp(x, mlp_cfg);
		tnn_vec_t *loss = tnn_cross_entropy(y, y_pred);

		float loss_value = tnn_item(loss);
		loss_sum += loss_value;
		printf("\nloss: %f", loss_value);
		tnn_free(loss);
	}

	printf(
	    "\nAverage loss over %zu steps: %f\n", num_steps, loss_sum / num_steps
	);

	tnn_terminate();

	return 0;
}

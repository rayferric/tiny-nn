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
	tnn_adamw_cfg_t adamw_cfg = TNN_ADAMW_CFG(.lr = 0.001f);

	const size_t num_steps = 600;
	const size_t batch_size = 100;
	for (int i = 0; i < num_steps; i++) {
		tnn_vec_t *x, *y; // ...

		tnn_vec_t *y_pred = tnn_mlp(x, mlp_cfg);
		tnn_vec_t *loss = tnn_cross_entropy(y, y_pred);

		tnn_zero_grad(loss);
		tnn_backward(loss);
		tnn_clip_grad_norm(loss, 1.0f);
		tnn_adamw(loss, adamw_cfg);

		printf("\nloss: ");
		tnn_print(loss);
		tnn_free(loss);
	}

	tnn_drop("adamw");
	tnn_save("cifar_mlp.tnn");

	tnn_terminate();

	return 0;
}

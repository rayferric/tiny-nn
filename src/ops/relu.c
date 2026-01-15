#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

static void relu_backward(tnn_tensor_t *self) {
	tnn_tensor_t *input = self->parents[0];

	if (input->requires_grad) {
		size_t total_size = tnn_size(input);
		for (size_t i = 0; i < total_size; i++) {
			// dself/dinput = 1 if input > 0, else 0
			if (input->data[i] > 0.0f) {
				input->grad[i] += self->grad[i];
			}
		}
	}
}

tnn_tensor_t *tnn_relu(tnn_tensor_t *input) {
	assert(input != NULL);

	// alloc output with same dims as input
	tnn_tensor_t *output = tnn_alloc(input->dims, input->num_dims);

	// output = max(0, input)
	size_t total_size = tnn_size(input);
	for (size_t i = 0; i < total_size; i++) {
		output->data[i] = input->data[i] > 0.0f ? input->data[i] : 0.0f;
	}

	output->requires_grad = input->requires_grad;
	output->parents[0] = input;
	output->num_parents = 1;
	input->num_children++;
	output->backward = relu_backward;

	return output;
}

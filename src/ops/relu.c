#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

static void relu_backward(tnn_tensor_t *self) {
	tnn_tensor_t *input = self->parents[0];

	if (input->type == TNN_OUTPUT) {
		size_t total_size = tnn_size(input);
		for (size_t i = 0; i < total_size; i++) {
			// dself/dinput = 1 if input > 0, else 0
			if (input->data[i] > 0.0f) {
				input->grad[i] += self->grad[i];
			}
		}
	}
}

#include "./impl/alloc_tensor.h"

tnn_tensor_t *tnn_relu(tnn_tensor_t *input) {
	assert(input != NULL);

	// alloc output with same dims as input
	tnn_tensor_t *output =
	    _tnn_alloc_tensor(input->dims, input->num_dims, TNN_OUTPUT);

	// output = max(0, input)
	size_t total_size = tnn_size(input);
	for (size_t i = 0; i < total_size; i++) {
		output->data[i] = input->data[i] > 0.0f ? input->data[i] : 0.0f;
	}

	output->parents[0] = input;
	output->num_parents = 1;
	output->backward = relu_backward;

	return output;
}

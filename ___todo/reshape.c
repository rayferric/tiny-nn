#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

static void reshape_backward(tnn_tensor_t *self) {
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

tnn_tensor_t *
tnn_reshape(tnn_tensor_t *input, const size_t *dims, size_t num_dims) {
	assert(input != NULL);

	// alloc output with given dims
	tnn_tensor_t *output = _tnn_alloc_tensor(dims, num_dims, TNN_OUTPUT);

	assert(tnn_size(input) == tnn_size(output));

	memcpy(output->data, input->data, tnn_size(input));

	output->parents[0] = input;
	output->num_parents = 1;
	output->backward = reshape_backward;

	return output;
}

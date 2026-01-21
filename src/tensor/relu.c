#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

static void relu_backward(tnn_tensor_t *self) {
	tnn_tensor_t *input = self->parents[0];

	if (input->requires_grad) {
		// dout/din = 1 if in > 0, else 0
		// in_grad += out_grad if out_data > 0 else 0
		input->dev->_backend.relu_bw(
		    input->dev, self->data, self->grad, input->grad, tnn_size(input)
		);
	}
}

tnn_tensor_t *tnn_relu(tnn_tensor_t *input) {
	assert(input != NULL);

	// alloc output with same dims as input
	tnn_tensor_t *output = tnn_alloc(input->dims, input->num_dims);

	// output = max(0, input)
	input->dev->_backend.relu_fw(
	    input->dev, input->data, output->data, tnn_size(input)
	);

	output->requires_grad = input->requires_grad;
	output->parents[0] = input;
	output->num_parents = 1;
	input->num_children++;
	output->_backward = relu_backward;

	return output;
}

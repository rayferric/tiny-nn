#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "./impl.h"

static void bias_backward(tnn_tensor_t *self) {
	tnn_tensor_t *input = self->parents[0];
	tnn_tensor_t *bias = self->parents[1];

	size_t dim_batch = 1;
	for (size_t i = 0; i < input->num_dims - 1; i++) {
		dim_batch *= input->dims[i];
	}
	size_t dim_feat = bias->dims[0];

	// input->grad += self->grad (bias doesn't affect input gradient
	// calculation)
	if (input->requires_grad) {
		size_t total_size = dim_batch * dim_feat;
		self->dev->_backend.accum(
		    self->dev, self->grad, input->grad, total_size
		);
	}

	// bias->grad += sum(self->grad over batch dimension)
	if (bias->requires_grad) {
		self->dev->_backend.sum_reduce(
		    self->dev,
		    self->grad,
		    bias->grad,
		    1,
		    dim_batch,
		    dim_feat,
		    1.0f,
		    true
		);
	}
}

tnn_tensor_t *tnn_bias(tnn_tensor_t *input) {
	assert(input->num_dims >= 1);

	size_t dim_batch = 1;
	for (size_t i = 0; i < input->num_dims - 1; i++) {
		dim_batch *= input->dims[i];
	}
	size_t dim_feat = input->dims[input->num_dims - 1];

	// get bias parameter
	size_t bias_dims[1] = {dim_feat};
	bool bias_created = false;
	tnn_tensor_t *bias = alloc_or_get_state_tensor_on_device(
	    bias_dims, 1, input->dev, "bias", &bias_created
	);
	bias->requires_grad = true;
	if (bias_created) {
		tnn_init_fill(bias, 0);
	}

	// alloc output with same dims as input
	tnn_tensor_t *output =
	    alloc_tensor_on_device(input->dims, input->num_dims, input->dev);

	// output = input + bias (broadcast over batch dimension)
	input->dev->_backend.add(
	    input->dev, input->data, bias->data, output->data, dim_batch, dim_feat
	);

	output->requires_grad = true;
	output->parents[0] = input;
	output->parents[1] = bias;
	output->num_parents = 2;
	input->num_children++;
	bias->num_children++;
	output->_backward = bias_backward;

	return output;
}

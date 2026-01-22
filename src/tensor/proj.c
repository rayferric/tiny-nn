#include <tnn/tnn.h>

#include <assert.h>
#include <memory.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "./impl.h"

static void proj_backward(tnn_tensor_t *self) {
	tnn_tensor_t *input = self->parents[0];
	tnn_tensor_t *weight = self->parents[1];

	size_t dim_batch = 1;
	for (size_t i = 0; i < input->num_dims - 1; i++) {
		dim_batch *= input->dims[i];
	}
	size_t dim_in = weight->dims[0];
	size_t dim_out = weight->dims[1];

	if (input->requires_grad) {
		// input->grad = self->grad @ weight^T
		self->dev->_backend.matmul(
		    self->dev,
		    (float *)self->grad,
		    (float *)weight->data,
		    (float *)input->grad,
		    dim_batch,
		    dim_out,
		    dim_in,
		    false,
		    true, // tpose weight
		    true  // accum into input->grad
		);
	}

	if (weight->requires_grad) {
		// weight->grad += input^T @ self->grad
		self->dev->_backend.matmul(
		    self->dev,
		    (float *)input->data,
		    (float *)self->grad,
		    (float *)weight->grad,
		    dim_in,
		    dim_batch,
		    dim_out,
		    true,
		    false,
		    true
		);
	}
}

tnn_tensor_t *tnn_proj(tnn_tensor_t *input, size_t dim_out) {
	assert(input->num_dims >= 1);

	size_t dim_batch = 1;
	for (size_t i = 0; i < input->num_dims - 1; i++) {
		dim_batch *= input->dims[i];
	}
	size_t dim_in = input->dims[input->num_dims - 1];

	// get weights
	size_t weight_dims[2] = {dim_in, dim_out};
	bool weight_created = false;
	tnn_tensor_t *weight = alloc_or_get_state_tensor_on_device(
	    weight_dims, 2, input->dev, "proj", &weight_created
	);
	weight->requires_grad = true;
	if (weight_created) {
		tnn_init_xavier(weight, dim_in, dim_out);
	}

	size_t output_dims[100];
	if (input->num_dims > 100) {
		fprintf(stderr, "input has too many dims (%zu)\n", input->num_dims);
		exit(1);
	}
	memcpy(output_dims, input->dims, (input->num_dims - 1) * sizeof(size_t));
	output_dims[input->num_dims - 1] = dim_out;
	tnn_tensor_t *output =
	    alloc_tensor_on_device(output_dims, input->num_dims, input->dev);

	// output = input @ weight
	input->dev->_backend.matmul(
	    input->dev,
	    input->data,
	    weight->data,
	    output->data,
	    dim_batch,
	    dim_in,
	    dim_out,
	    false, // no tpose
	    false, // no tpose
	    false  // no accum
	);

	output->parents[0] = input;
	output->parents[1] = weight;
	output->num_parents = 2;
	output->requires_grad = true;
	input->num_children++;
	weight->num_children++;
	output->_backward = proj_backward;

	return output;
}

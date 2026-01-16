#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

static void bias_backward(tnn_tensor_t *self) {
	tnn_tensor_t *input = self->parents[0];
	tnn_tensor_t *bias = self->parents[1];

	size_t dim_batch = 1;
	for (size_t i = 0; i < input->num_dims - 1; i++) {
		dim_batch *= input->dims[i];
	}
	size_t dim_features = bias->dims[0];

	// input->grad += self->grad (bias doesn't affect input gradient
	// calculation)
	if (input->requires_grad) {
		size_t total_size = dim_batch * dim_features;
		for (size_t i = 0; i < total_size; i++) {
			input->grad[i] += self->grad[i];
		}
	}

	// bias->grad += sum(self->grad over batch dimension)
	for (size_t i_batch = 0; i_batch < dim_batch; i_batch++) {
		for (size_t i_feat = 0; i_feat < dim_features; i_feat++) {
			bias->grad[i_feat] += self->grad[i_batch * dim_features + i_feat];
		}
	}
}

tnn_tensor_t *tnn_bias(tnn_tensor_t *input) {
	assert(input->num_dims >= 1);

	size_t dim_batch = 1;
	for (size_t i = 0; i < input->num_dims - 1; i++) {
		dim_batch *= input->dims[i];
	}
	size_t dim_in = input->dims[input->num_dims - 1];

	// get bias parameter
	size_t bias_dims[1] = {dim_in};
	bool bias_created = false;
	tnn_tensor_t *bias =
	    tnn_alloc_or_get_state(bias_dims, 1, "bias", &bias_created);
	bias->requires_grad = true;
	if (bias_created) {
		tnn_init_fill(bias, 0);
	}

	// alloc output with same dims as input
	tnn_tensor_t *output = tnn_alloc(input->dims, input->num_dims);

	// output = input + bias (broadcast over batch dimension)
	for (size_t i_batch = 0; i_batch < dim_batch; i_batch++) {
		for (size_t i_feat = 0; i_feat < dim_in; i_feat++) {
			output->data[i_batch * dim_in + i_feat] =
			    input->data[i_batch * dim_in + i_feat] + bias->data[i_feat];
		}
	}

	output->requires_grad = true;
	output->parents[0] = input;
	output->parents[1] = bias;
	output->num_parents = 2;
	input->num_children++;
	bias->num_children++;
	output->backward = bias_backward;

	return output;
}

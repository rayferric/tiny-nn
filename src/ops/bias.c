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
	if (input->type == TNN_OUTPUT) {
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

#include "./impl/alloc_tensor.h"
#include "./impl/param_table.h"

tnn_tensor_t *tnn_bias(tnn_tensor_t *input) {
	assert(input->num_dims >= 1);

	size_t dim_batch = 1;
	for (size_t i = 0; i < input->num_dims - 1; i++) {
		dim_batch *= input->dims[i];
	}
	size_t dim_features = input->dims[input->num_dims - 1];

	// get bias parameter
	size_t bias_dims[1] = {dim_features};
	bool bias_created = false;
	tnn_tensor_t *bias = _tnn_get_or_create_param(
	    "bias", bias_dims, 1, TNN_PARAMETER, &bias_created
	);
	if (bias_created) {
		// zero init
		memset(bias->data, 0, dim_features * sizeof(float));
	}

	// alloc output with same dims as input
	tnn_tensor_t *output =
	    _tnn_alloc_tensor(input->dims, input->num_dims, TNN_OUTPUT);

	// output = input + bias (broadcast over batch dimension)
	for (size_t i_batch = 0; i_batch < dim_batch; i_batch++) {
		for (size_t i_feat = 0; i_feat < dim_features; i_feat++) {
			output->data[i_batch * dim_features + i_feat] =
			    input->data[i_batch * dim_features + i_feat] +
			    bias->data[i_feat];
		}
	}

	output->parents[0] = input;
	output->parents[1] = bias;
	output->num_parents = 2;
	output->backward = bias_backward;

	return output;
}

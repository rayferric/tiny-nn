#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "../util/safe_malloc.h"
#include "impl.h"

typedef struct {
	size_t num_averaged, outer_size, inner_size;
} mean_context_t;

static void mean_free_context(void *ctx) {
	mean_context_t *m_ctx = (mean_context_t *)ctx;
	free(m_ctx);
}

static void mean_backward(tnn_tensor_t *self) {
	tnn_tensor_t *input = self->parents[0];

	if (!input->requires_grad) {
		return;
	}

	assert(self->_ctx != NULL);
	mean_context_t *ctx = (mean_context_t *)self->_ctx;

	// broadcast gradient from output to input
	input->dev->_backend.sum_broadcast(
	    input->dev,
	    self->grad,
	    input->grad,
	    ctx->outer_size,
	    ctx->num_averaged,
	    ctx->inner_size,
	    // gradient coefficient - each input element contributed 1/n to the mean
	    1.0f / (float)ctx->num_averaged,
	    true
	);
}

tnn_tensor_t *_tnn_mean(tnn_tensor_t *input, size_t i_dim, size_t num_dims) {
	assert(input != NULL);
	assert(num_dims > 0);
	assert(i_dim + num_dims <= input->num_dims);

	// calculate output dimensions - remove the reduced dimensions
	size_t output_num_dims = input->num_dims - num_dims;
	size_t *output_dims = safe_malloc(output_num_dims * sizeof(size_t));
	// copy dims before and after the reduced range
	for (size_t i = 0; i < i_dim; i++) {
		output_dims[i] = input->dims[i];
	}
	for (size_t i = i_dim + num_dims; i < input->num_dims; i++) {
		output_dims[i - num_dims] = input->dims[i];
	}
	// allocate output tensor
	tnn_tensor_t *output =
	    alloc_tensor_on_device(output_dims, output_num_dims, input->dev);
	free(output_dims);

	// the number of elements to average over
	size_t num_averaged = 1;
	for (size_t i = i_dim; i < i_dim + num_dims; i++) {
		num_averaged *= input->dims[i];
	}
	// outer size (product of dims before reduced range)
	size_t outer_size = 1;
	for (size_t i = 0; i < i_dim; i++) {
		outer_size *= input->dims[i];
	}
	// inner size (product of dims after reduced range)
	size_t inner_size = 1;
	for (size_t i = i_dim + num_dims; i < input->num_dims; i++) {
		inner_size *= input->dims[i];
	}

	// compute the mean
	input->dev->_backend.sum_reduce(
	    input->dev,
	    input->data,
	    output->data,
	    outer_size,
	    num_averaged,
	    inner_size,
	    1.0f / (float)num_averaged,
	    false
	);

	mean_context_t *ctx = safe_malloc(sizeof(mean_context_t));
	ctx->num_averaged = num_averaged;
	ctx->outer_size = outer_size;
	ctx->inner_size = inner_size;

	output->parents[0] = input;
	output->num_parents = 1;
	input->num_children++;
	output->requires_grad = input->requires_grad;
	output->_backward = mean_backward;
	output->_ctx = ctx;
	output->_free_ctx = mean_free_context;

	return output;
}

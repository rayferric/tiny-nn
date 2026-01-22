#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <string.h>

#include "../util/safe_malloc.h"
#include "impl.h"

typedef struct {
	size_t *input_dims;
	size_t input_num_dims;
} reshape_context_t;

static void reshape_free_context(void *ctx) {
	reshape_context_t *r_ctx = (reshape_context_t *)ctx;
	free(r_ctx->input_dims);
	free(r_ctx);
}

static void reshape_backward(tnn_tensor_t *self) {
	tnn_tensor_t *input = self->parents[0];

	assert(self->_ctx != NULL);
	reshape_context_t *ctx = (reshape_context_t *)self->_ctx;

	// copy gradient back to input with original shape
	if (input->requires_grad) {
		input->dev->_backend.accum(
		    input->dev, self->grad, input->grad, tnn_size(input)
		);
	}
}

tnn_tensor_t *
tnn_reshape(tnn_tensor_t *input, const size_t *dims, size_t num_dims) {
	assert(input != NULL);
	assert(dims != NULL);
	assert(num_dims > 0);

	size_t input_size = tnn_size(input);

	// handle one 0 dimension - autofill
	int zero_idx = -1;
	size_t known_size = 1;
	for (size_t i = 0; i < num_dims; i++) {
		if (dims[i] == 0) {
			assert(zero_idx == -1 && "only one dimension can be 0");
			zero_idx = i;
		} else {
			known_size *= dims[i];
		}
	}
	size_t *actual_dims = safe_malloc(num_dims * sizeof(size_t));
	memcpy(actual_dims, dims, num_dims * sizeof(size_t));
	if (zero_idx >= 0) {
		assert(input_size % known_size == 0 && "size must divide evenly");
		actual_dims[zero_idx] = input_size / known_size;
	}

	// verify total size matches
	size_t output_size = 1;
	for (size_t i = 0; i < num_dims; i++) {
		output_size *= actual_dims[i];
	}
	assert(
	    input_size == output_size && "reshape: total size must remain the same"
	);

	reshape_context_t *ctx = safe_malloc(sizeof(reshape_context_t));
	ctx->input_num_dims = input->num_dims;
	ctx->input_dims = safe_malloc(input->num_dims * sizeof(size_t));
	memcpy(ctx->input_dims, input->dims, input->num_dims * sizeof(size_t));

	tnn_tensor_t *output =
	    alloc_tensor_on_device(actual_dims, num_dims, input->dev);
	free(actual_dims);

	input->dev->_backend.buf_copy(
	    input->dev, output->data, input->data, input_size * sizeof(float)
	);

	output->parents[0] = input;
	output->num_parents = 1;
	input->num_children++;
	output->requires_grad = input->requires_grad;
	output->_backward = reshape_backward;
	output->_ctx = ctx;
	output->_free_ctx = reshape_free_context;

	return output;
}

#include <tnn/tnn.h>

#include <assert.h>
#include <memory.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "../util/safe_malloc.h"

typedef struct {
	size_t dim_out;
	size_t kernel_size;
	size_t stride;
	size_t padding;
} conv_context_t;

static void conv_free_context(void *ctx) {
	free(ctx);
}

static void conv_backward(tnn_tensor_t *self) {
	tnn_tensor_t *input = self->parents[0];
	tnn_tensor_t *weight = self->parents[1];

	assert(self->_ctx != NULL);
	conv_context_t *ctx = (conv_context_t *)self->_ctx;

	size_t batch = 1;
	for (size_t i = 0; i < input->num_dims - 3; i++) {
		batch *= input->dims[i];
	}

	size_t h_in = input->dims[input->num_dims - 3];
	size_t w_in = input->dims[input->num_dims - 2];
	size_t c_in = input->dims[input->num_dims - 1];
	size_t c_out = ctx->dim_out;
	size_t k = ctx->kernel_size;
	size_t s = ctx->stride;
	size_t p = ctx->padding;

	size_t h_out = (h_in + 2 * p - k) / s + 1;
	size_t w_out = (w_in + 2 * p - k) / s + 1;

	if (input->requires_grad) {
		self->dev->_backend.conv_bw_input(
		    self->dev,
		    self->grad,
		    weight->data,
		    input->grad,
		    batch,
		    h_in,
		    w_in,
		    c_in,
		    h_out,
		    w_out,
		    c_out,
		    k,
		    s,
		    p
		);
	}
	if (weight->requires_grad) {
		self->dev->_backend.conv_bw_weight(
		    self->dev,
		    input->data,
		    self->grad,
		    weight->grad,
		    batch,
		    h_in,
		    w_in,
		    c_in,
		    h_out,
		    w_out,
		    c_out,
		    k,
		    s,
		    p
		);
	}
}

tnn_tensor_t *_tnn_conv(
    tnn_tensor_t *input,
    size_t dim_out,
    size_t kernel_size,
    size_t stride,
    size_t padding
) {
	assert(input->num_dims >= 3);

	size_t batch = 1;
	for (size_t i = 0; i < input->num_dims - 3; i++) {
		batch *= input->dims[i];
	}

	size_t h_in = input->dims[input->num_dims - 3];
	size_t w_in = input->dims[input->num_dims - 2];
	size_t dim_in = input->dims[input->num_dims - 1];

	// available space to slide: (h_in + 2*padding - kernel_size)
	// this is the distance from first to last valid kernel position
	// ---
	// number of steps taken: distance / stride
	// if stride=2, you only count every other position
	// ---
	// positions = steps + 1 (fencepost problem)
	size_t h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
	size_t w_out = (w_in + 2 * padding - kernel_size) / stride + 1;

	// weight dims: [out_channels, kernel_size, kernel_size, in_channels]
	size_t weight_dims[4] = {dim_out, kernel_size, kernel_size, dim_in};
	bool weight_created = false;
	tnn_tensor_t *weight =
	    tnn_alloc_or_get_state(weight_dims, 4, "conv", &weight_created);
	weight->requires_grad = true;
	if (weight_created) {
		input->dev->_backend.xavier(
		    input->dev,
		    weight->data,
		    tnn_size(weight),
		    kernel_size * kernel_size * dim_in,
		    kernel_size * kernel_size * dim_out
		);
	}

	size_t output_dims[100];
	if (input->num_dims > 100) {
		fprintf(stderr, "input has too many dims (%zu)\n", input->num_dims);
		exit(1);
	}
	memcpy(output_dims, input->dims, (input->num_dims - 3) * sizeof(size_t));
	output_dims[input->num_dims - 3] = h_out;
	output_dims[input->num_dims - 2] = w_out;
	output_dims[input->num_dims - 1] = dim_out;

	tnn_tensor_t *output = tnn_alloc(output_dims, input->num_dims);
	input->dev->_backend.conv_fw(
	    input->dev,
	    input->data,
	    weight->data,
	    output->data,
	    batch,
	    h_in,
	    w_in,
	    dim_in,
	    h_out,
	    w_out,
	    dim_out,
	    kernel_size,
	    stride,
	    padding
	);

	conv_context_t *ctx = safe_malloc(sizeof(conv_context_t));
	ctx->dim_out = dim_out;
	ctx->kernel_size = kernel_size;
	ctx->stride = stride;
	ctx->padding = padding;

	output->parents[0] = input;
	output->parents[1] = weight;
	output->num_parents = 2;
	output->requires_grad = true;
	input->num_children++;
	weight->num_children++;
	output->_backward = conv_backward;
	output->_ctx = ctx;
	output->_free_ctx = conv_free_context;

	return output;
}

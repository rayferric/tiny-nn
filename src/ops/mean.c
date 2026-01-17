#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "../impl/malloc.h"

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

	assert(self->context != NULL);
	mean_context_t *ctx = (mean_context_t *)self->context;

	// gradient coefficient - each input element contributed 1/n to the mean
	float grad_coeff = 1.0f / (float)ctx->num_averaged;

	// broadcast gradient from output to input
	for (size_t outer = 0; outer < ctx->outer_size; outer++) {
		for (size_t reduced = 0; reduced < ctx->num_averaged; reduced++) {
			for (size_t inner = 0; inner < ctx->inner_size; inner++) {
				size_t input_idx =
				    outer * (ctx->num_averaged * ctx->inner_size) +
				    reduced * ctx->inner_size + inner;
				size_t output_idx = outer * ctx->inner_size + inner;
				input->grad[input_idx] += self->grad[output_idx] * grad_coeff;
			}
		}
	}
}

tnn_tensor_t *_tnn_mean(tnn_tensor_t *input, size_t i_dim, size_t num_dims) {
	assert(input != NULL);
	assert(num_dims > 0);
	assert(i_dim + num_dims <= input->num_dims);

	// calculate output dimensions - remove the reduced dimensions
	size_t output_num_dims = input->num_dims - num_dims;
	size_t *output_dims = tnn_safe_malloc(output_num_dims * sizeof(size_t));
	// copy dims before and after the reduced range
	for (size_t i = 0; i < i_dim; i++) {
		output_dims[i] = input->dims[i];
	}
	for (size_t i = i_dim + num_dims; i < input->num_dims; i++) {
		output_dims[i - num_dims] = input->dims[i];
	}
	// allocate output tensor
	tnn_tensor_t *output = tnn_alloc(output_dims, output_num_dims);
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
	for (size_t outer = 0; outer < outer_size; outer++) {
		for (size_t inner = 0; inner < inner_size; inner++) {
			float sum = 0.0f;
			for (size_t reduced = 0; reduced < num_averaged; reduced++) {
				size_t input_idx = outer * (num_averaged * inner_size) +
				                   reduced * inner_size + inner;
				sum += input->data[input_idx];
			}
			size_t output_idx = outer * inner_size + inner;
			output->data[output_idx] = sum / (float)num_averaged;
		}
	}

	mean_context_t *ctx = tnn_safe_malloc(sizeof(mean_context_t));
	ctx->num_averaged = num_averaged;
	ctx->outer_size = outer_size;
	ctx->inner_size = inner_size;

	output->parents[0] = input;
	output->num_parents = 1;
	input->num_children++;
	output->requires_grad = input->requires_grad;
	output->backward = mean_backward;
	output->context = ctx;
	output->free_context = mean_free_context;

	return output;
}

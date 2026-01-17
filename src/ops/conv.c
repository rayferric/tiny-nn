#include <tnn/tnn.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "../impl/malloc.h"

typedef struct {
	size_t in_channels;
	size_t height;
	size_t width;
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

	assert(self->context != NULL);
	conv_context_t *ctx = (conv_context_t *)self->context;

	size_t batch = 1;
	for (size_t i = 0; i < input->num_dims - 3; i++) {
		batch *= input->dims[i];
	}

	size_t h_in = ctx->height;
	size_t w_in = ctx->width;
	size_t c_in = ctx->in_channels;
	size_t c_out = ctx->dim_out;
	size_t k = ctx->kernel_size;
	size_t s = ctx->stride;
	size_t p = ctx->padding;

	size_t h_out = (h_in + 2 * p - k) / s + 1;
	size_t w_out = (w_in + 2 * p - k) / s + 1;

	// clang-format off
	for (size_t b = 0; b < batch; b++) {
	for (size_t i_out = 0; i_out < h_out; i_out++) {
	for (size_t j_out = 0; j_out < w_out; j_out++) {
	for (size_t c = 0; c < c_out; c++) {
		float grad_val = tnn_grad_at(self, b, i_out, j_out, c);

		for (size_t ki = 0; ki < k; ki++) {
		for (size_t kj = 0; kj < k; kj++) {
			int i_in = i_out * s + ki - p;
			int j_in = j_out * s + kj - p;

			if (i_in >= 0 && i_in < (int)h_in && j_in >= 0 && j_in < (int)w_in) {
				for (size_t c_i = 0; c_i < c_in; c_i++) {
					if (input->requires_grad) {
						float w_val = tnn_value_at(weight, c, ki, kj, c_i);
						tnn_grad_at(input, b, i_in, j_in, c_i) += grad_val * w_val;
					}

					if (weight->requires_grad) {
						float in_val = tnn_value_at(input, b, i_in, j_in, c_i);
						tnn_grad_at(weight, c, ki, kj, c_i) += grad_val * in_val;
					}
				}
			}
		}
		}
	}
	}
	}
	}
	// clang-format on
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
	size_t c_in = input->dims[input->num_dims - 1];

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
	size_t weight_dims[4] = {dim_out, kernel_size, kernel_size, c_in};
	bool weight_created = false;
	tnn_tensor_t *weight =
	    tnn_alloc_or_get_state(weight_dims, 4, "conv", &weight_created);
	weight->requires_grad = true;
	if (weight_created) {
		// uniform xavier init
		size_t fan_in = kernel_size * kernel_size * c_in;
		size_t fan_out = kernel_size * kernel_size * dim_out;

		float limit = sqrtf(6.0f / (fan_in + fan_out));

		size_t total_size = tnn_size(weight);
		for (size_t i = 0; i < total_size; i++) {
			float u = (float)rand() / (float)RAND_MAX;
			weight->data[i] = u * 2.0f * limit - limit;
		}
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

	// forward pass
	// clang-format off
    for (size_t b = 0; b < batch; b++) {
    for (size_t i = 0; i < h_out; i++) {
    for (size_t j = 0; j < w_out; j++) {
    for (size_t c = 0; c < dim_out; c++) {
        float sum = 0.0f;

        for (size_t ki = 0; ki < kernel_size; ki++) {
        for (size_t kj = 0; kj < kernel_size; kj++) {
            int i_in = i * stride + ki - padding;
            int j_in = j * stride + kj - padding;

            if (i_in >= 0 && i_in < (int)h_in && j_in >= 0 && j_in < (int)w_in) {
                for (size_t c_in_idx = 0; c_in_idx < c_in; c_in_idx++) {
					float in_val = tnn_value_at(input, b, i_in, j_in, c_in_idx);
					float w_val = tnn_value_at(weight, c, ki, kj, c_in_idx);
                    sum += in_val * w_val;
                }
            }
        }
        }

		tnn_value_at(output, b, i, j, c) = sum;
    }
    }
    }
	}
	// clang-format on

	conv_context_t *ctx = tnn_safe_malloc(sizeof(conv_context_t));
	ctx->in_channels = c_in;
	ctx->height = h_in;
	ctx->width = w_in;
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
	output->backward = conv_backward;
	output->context = ctx;
	output->free_context = conv_free_context;

	return output;
}

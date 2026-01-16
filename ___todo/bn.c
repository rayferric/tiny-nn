#include <tnn/tnn.h>

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

// Context for batch normalization
typedef struct {
	float *saved_mean;
	float *saved_inv_std;
	size_t num_channels;
} bn_context_t;

static void bn_free_context(void *ctx) {
	bn_context_t *bn_ctx = (bn_context_t *)ctx;
	free(bn_ctx->saved_mean);
	free(bn_ctx->saved_inv_std);
	free(bn_ctx);
}

// Batch Normalization forward pass (HWC format)
// Only performs normalization, no scale/bias
static void bn_forward(
    const float *__restrict input,
    float *__restrict running_mean,
    float *__restrict running_var,
    float *__restrict saved_mean,
    float *__restrict saved_inv_std,
    float *__restrict output,
    size_t batch_spatial,
    size_t channels,
    float momentum,
    float eps
) {
	// Compute mean and variance
	for (size_t c = 0; c < channels; c++) {
		float mean = 0.0f;
		for (size_t bs = 0; bs < batch_spatial; bs++) {
			mean += input[bs * channels + c];
		}
		mean /= (float)batch_spatial;

		float var = 0.0f;
		for (size_t bs = 0; bs < batch_spatial; bs++) {
			float diff = input[bs * channels + c] - mean;
			var += diff * diff;
		}
		var /= (float)batch_spatial;

		// Update running statistics
		running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
		running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var;

		// Save for backward pass
		saved_mean[c] = mean;
		saved_inv_std[c] = 1.0f / sqrtf(var + eps);

		for (size_t bs = 0; bs < batch_spatial; bs++) {
			size_t idx = bs * channels + c;
			output[idx] = (input[idx] - mean) * saved_inv_std[c];
		}
	}
}

// Batch Normalization backward pass
static void bn_backward(
    const float *__restrict grad_output,
    const float *__restrict input,
    const float *__restrict saved_mean,
    const float *__restrict saved_inv_std,
    float *__restrict grad_input,
    size_t batch_spatial,
    size_t channels
) {
	for (size_t c = 0; c < channels; c++) {
		float mean = saved_mean[c];
		float inv_std = saved_inv_std[c];

		if (grad_input != NULL) {
			// Sum of grad_output
			float sum_grad = 0.0f;
			// Sum of grad_output * normalized
			float sum_grad_norm = 0.0f;
			for (size_t bs = 0; bs < batch_spatial; bs++) {
				size_t idx = bs * channels + c;
				float normalized = (input[idx] - mean) * inv_std;
				sum_grad += grad_output[idx];
				sum_grad_norm += grad_output[idx] * normalized;
			}

			float N = (float)batch_spatial;
			for (size_t bs = 0; bs < batch_spatial; bs++) {
				size_t idx = bs * channels + c;
				float normalized = (input[idx] - mean) * inv_std;

				float grad = grad_output[idx];
				grad -= sum_grad / N;
				grad -= normalized * sum_grad_norm / N;
				grad *= inv_std;

				grad_input[idx] += grad;
			}
		}
	}
}

static void bn_backward_wrapper(tnn_tensor_t *self) {
	tnn_tensor_t *input = self->parents[0];

	size_t total_size = tnn_size(input);
	size_t channels = input->dims[input->num_dims - 1];
	size_t batch_spatial = total_size / channels;

	// Retrieve saved statistics from context
	assert(self->context != NULL && "bn_backward: context is NULL");
	bn_context_t *ctx = (bn_context_t *)self->context;

	bn_backward(
	    self->grad,
	    input->data,
	    ctx->saved_mean,
	    ctx->saved_inv_std,
	    input->type == TNN_OUTPUT ? input->grad : NULL,
	    batch_spatial,
	    channels
	);

	// Clean up context after backward
	bn_free_context(self->context);
	self->context = NULL;
	self->free_context = NULL;
}

tnn_tensor_t *tnn_bn(tnn_tensor_t *input, float momentum) {
	assert(input != NULL);
	assert(input->num_dims >= 2);

	tnn_push("bn");

	// Extract dimensions
	size_t total_size = tnn_size(input);
	size_t channels = input->dims[input->num_dims - 1];
	size_t batch_spatial = total_size / channels;

	// Get or create running mean buffer [channels]
	size_t param_dims[1] = {channels};
	bool running_mean_created = false;
	tnn_tensor_t *running_mean = _tnn_get_or_create_param(
	    "mean", param_dims, 1, TNN_BUFFER, &running_mean_created
	);
	if (running_mean_created) {
		memset(running_mean->data, 0, channels * sizeof(float));
	}

	// Get or create running variance buffer [channels]
	bool running_var_created = false;
	tnn_tensor_t *running_var = _tnn_get_or_create_param(
	    "var", param_dims, 1, TNN_BUFFER, &running_var_created
	);
	if (running_var_created) {
		// Initialize to 1
		for (size_t i = 0; i < channels; i++) {
			running_var->data[i] = 1.0f;
		}
	}

	// Create context for backward pass
	bn_context_t *ctx = malloc(sizeof(bn_context_t));
	ctx->num_channels = channels;
	ctx->saved_mean = malloc(channels * sizeof(float));
	ctx->saved_inv_std = malloc(channels * sizeof(float));

	// Allocate output with same shape as input
	tnn_tensor_t *output =
	    _tnn_alloc_tensor(input->dims, input->num_dims, TNN_OUTPUT);

	// Forward pass
	float eps = 1e-5f;
	bn_forward(
	    input->data,
	    running_mean->data,
	    running_var->data,
	    ctx->saved_mean,
	    ctx->saved_inv_std,
	    output->data,
	    batch_spatial,
	    channels,
	    momentum,
	    eps
	);

	// Store context for backward pass
	output->context = ctx;
	output->free_context = bn_free_context;

	output->parents[0] = input;
	output->num_parents = 1;
	output->backward = bn_backward_wrapper;

	tnn_pop();

	return output;
}

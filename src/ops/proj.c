#include <tnn/tnn.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

static void matmul(
    const float *a,
    const float *b,
    float *c,
    size_t m,
    size_t k,
    size_t n,
    bool tpose_a,
    bool tpose_b,
    bool accum
) {
	// C[M, N] = A[M, K] @ B[K, N]
	for (size_t i_m = 0; i_m < m; i_m++) {
		for (size_t i_n = 0; i_n < n; i_n++) {
			float sum = 0.0f;
			for (size_t i_k = 0; i_k < k; i_k++) {
				float a_val = tpose_a ? a[i_k * m + i_m] : a[i_m * k + i_k];
				float b_val = tpose_b ? b[i_n * k + i_k] : b[i_k * n + i_n];
				sum += a_val * b_val;
			}

			if (accum) {
				c[i_m * n + i_n] += sum;
			} else {
				c[i_m * n + i_n] = sum;
			}
		}
	}
}

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
		matmul(
		    self->grad,
		    weight->data,
		    input->grad,
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
		matmul(
		    input->data,
		    self->grad,
		    weight->grad,
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
	assert(input->num_dims >= 2);

	size_t dim_batch = 1;
	for (size_t i = 0; i < input->num_dims - 1; i++) {
		dim_batch *= input->dims[i];
	}
	size_t dim_in = input->dims[input->num_dims - 1];

	// get weights
	size_t weight_dims[2] = {dim_in, dim_out};
	bool weight_created = false;
	tnn_tensor_t *weight =
	    tnn_alloc_or_get_state(weight_dims, 2, "proj", &weight_created);
	weight->requires_grad = true;
	if (weight_created) {
		// uniform xavier init
		size_t fan_in = weight->dims[weight->num_dims - 2];
		size_t fan_out = weight->dims[weight->num_dims - 1];

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
	memcpy(output_dims, input->dims, (input->num_dims - 1) * sizeof(size_t));
	output_dims[input->num_dims - 1] = dim_out;
	tnn_tensor_t *output = tnn_alloc(output_dims, input->num_dims);

	// output = input @ weight
	matmul(
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
	output->backward = proj_backward;

	return output;
}

#include <tnn/tnn.h>

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

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

	// input->grad = self->grad @ weight^T
	if (input->requires_grad) {
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
		tnn_init_xavier(weight);
	}

	// alloc output
	size_t output_dims[2] = {dim_batch, dim_out};
	tnn_tensor_t *output = tnn_alloc(output_dims, 2);

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

	output->requires_grad = true;
	output->parents[0] = input;
	output->parents[1] = weight;
	output->num_parents = 2;
	input->num_children++;
	weight->num_children++;
	output->backward = proj_backward;

	return output;
}

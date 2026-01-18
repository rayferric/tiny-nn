#include <tnn/tnn.h>

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./devices.h"
#include "./impl/malloc.h"

tnn_tensor_t *
alloc_tensor_on_device(const size_t *dims, size_t num_dims, tnn_device_t *dev) {
	tnn_tensor_t *t = safe_malloc(sizeof(tnn_tensor_t));

	t->device = dev;

	t->num_dims = num_dims;
	if (num_dims > 0) {
		t->dims = safe_malloc(num_dims * sizeof(size_t));
		memcpy(t->dims, dims, num_dims * sizeof(size_t));
	} else {
		t->dims = NULL;
	}

	size_t total_size = tnn_size(t);
	t->data = t->device->_ops.buf_alloc(t->device, total_size * sizeof(float));
	t->grad = NULL;

	t->requires_grad = false;
	t->is_state = false;

	t->num_parents = 0;
	t->num_children = 0;
	t->backward = NULL;

	t->context = NULL;
	t->free_context = NULL;

	return t;
}

tnn_tensor_t *tnn_alloc(const size_t *dims, size_t num_dims) {
	return alloc_tensor_on_device(
	    dims, num_dims, device_globals.default_device
	);
}

tnn_tensor_t *tnn_alloc_or_get_state(
    const size_t *dims, size_t num_dims, const char *key, bool *allocated
) {
	tnn_tensor_t *t = tnn_get_state(key);
	if (t != NULL) {
		if (allocated) {
			*allocated = false;
		}
		return t;
	}

	t = tnn_alloc(dims, num_dims);
	t->is_state = true;
	tnn_set_state(key, t);

	if (allocated) {
		*allocated = true;
	}

	return t;
}

void tnn_free(tnn_tensor_t *t) {
	assert(t != NULL);

	// skip freeing t if still referenced
	if (t->num_children > 0 || t->is_state) {
		return;
	}

	// decrement ref counts for parents and free them recursively
	for (size_t i = 0; i < t->num_parents; i++) {
		tnn_tensor_t *parent = t->parents[i];
		assert(parent != NULL);
		parent->num_children--;
		tnn_free(parent);
	}

	// free current tensor
	t->device->_ops.buf_free(t->device, t->data);
	if (t->grad) {
		t->device->_ops.buf_free(t->device, t->grad);
	}
	free(t->dims);
	if (t->context != NULL && t->free_context != NULL) {
		// (forward was executed without backward)
		t->free_context(t->context);
	}
	free(t);
}

tnn_tensor_t *tnn_detach(tnn_tensor_t *t) {
	tnn_tensor_t *detached =
	    alloc_tensor_on_device(t->dims, t->num_dims, t->device);
	t->device->_ops.buf_copy(
	    t->device, detached->data, t->data, tnn_size(t) * sizeof(float)
	);
	return detached;
}

tnn_tensor_t *tnn_detach_free(tnn_tensor_t *t) {
	tnn_tensor_t *detached = tnn_detach(t);
	tnn_free(t);
	return detached;
}

void tnn_init_from_memory(tnn_tensor_t *t, const float *data) {
	t->device->_ops.buf_copy_to_device(
	    t->device, t->data, data, tnn_size(t) * sizeof(float)
	);
}

void tnn_init_fill(tnn_tensor_t *t, float value) {
	size_t total_size = tnn_size(t);
	float *tmp_buf = (float *)safe_malloc(total_size * sizeof(float));
	for (size_t i = 0; i < total_size; i++) {
		tmp_buf[i] = value;
	}
	t->device->_ops.buf_copy_to_device(
	    t->device, t->data, tmp_buf, total_size * sizeof(float)
	);
	free(tmp_buf);
}

void tnn_init_randn(tnn_tensor_t *t) {
	size_t total_size = tnn_size(t);
	float *tmp_buf = (float *)safe_malloc(total_size * sizeof(float));
	for (size_t i = 0; i < total_size; i++) {
		// box-muller transform for normal distribution
		float u1 = (float)rand() / (float)RAND_MAX;
		float u2 = (float)rand() / (float)RAND_MAX;
		float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
		tmp_buf[i] = z;
	}
	t->device->_ops.buf_copy_to_device(
	    t->device, t->data, tmp_buf, total_size * sizeof(float)
	);
	free(tmp_buf);
}

size_t tnn_dim(tnn_tensor_t *t, int32_t i_dim) {
	// wrap negative indices
	if (i_dim < 0) {
		i_dim += t->num_dims;
	}
	assert(i_dim < t->num_dims);
	return t->dims[i_dim];
}

size_t tnn_size(tnn_tensor_t *t) {
	size_t total_size = 1;
	for (size_t i = 0; i < t->num_dims; i++) {
		total_size *= t->dims[i];
	}
	return total_size;
}

size_t tnn_index_at(tnn_tensor_t *t, size_t *indices, size_t num_indices) {
	// if there's less indices than dims, treat leading dims as part of the
	// first indexed dim
	size_t offset = indices[0];
	for (size_t i = 1; i < num_indices; i++) {
		offset = offset * t->dims[i + (t->num_dims - num_indices)] + indices[i];
	}
	return offset;
}

void tnn_print(tnn_tensor_t *t) {
	assert(t->num_dims <= 2 && "tnn_print: only 0D/1D/2D supported");

	// verify cpu device
	if (!t->device->_is_cpu) {
		fprintf(stderr, "tnn_print: tensor must be on cpu\n");
		return;
	}

	if (t->num_dims == 0) {
		printf("%.4f", t->data[0]);
	} else if (t->num_dims == 1) {
		printf("[");
		for (size_t i = 0; i < t->dims[0]; i++) {
			printf("%.4f", t->data[i]);
			if (i < t->dims[0] - 1) {
				printf(", ");
			}
		}
		printf("]");
	} else if (t->num_dims == 2) {
		printf("[\n");
		for (size_t i = 0; i < t->dims[0]; i++) {
			printf("  [");
			for (size_t j = 0; j < t->dims[1]; j++) {
				printf("%.4f", t->data[i * t->dims[1] + j]);
				if (j < t->dims[1] - 1) {
					printf(", ");
				}
			}
			printf("]");
			if (i < t->dims[0] - 1) {
				printf(",");
			}
			printf("\n");
		}
		printf("]");
	}
}

float tnn_item(tnn_tensor_t *t) {
	float value;
	t->device->_ops.buf_copy_to_host(t->device, &value, t->data, sizeof(float));
	return value;
}

// OPERATION ROUTING

tnn_tensor_t *tnn_proj(tnn_tensor_t *input, size_t dim_out) {
	//
}

// tnn_tensor_t *tnn_bias(tnn_tensor_t *input);

// tnn_tensor_t *tnn_relu(tnn_tensor_t *input);

// tnn_tensor_t *tnn_cross_entropy(tnn_tensor_t *pred, tnn_tensor_t *target);

// tnn_tensor_t *_tnn_conv(
//     tnn_tensor_t *input,
//     size_t dim_out,
//     size_t kernel_size,
//     size_t stride,
//     size_t padding
// );

// tnn_tensor_t *_tnn_bn(tnn_tensor_t *input, float momentum, bool test);

// tnn_tensor_t *tnn_add(tnn_tensor_t *a, tnn_tensor_t *b);

// tnn_tensor_t *_tnn_mean(tnn_tensor_t *input, size_t i_dim, size_t num_dims);

// tnn_tensor_t *
// tnn_reshape(tnn_tensor_t *input, const size_t *dims, size_t num_dims);

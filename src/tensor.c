#include <tnn/tnn.h>

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./impl/malloc.h"

tnn_tensor_t *tnn_alloc(const size_t *dims, size_t num_dims) {
	tnn_tensor_t *t = tnn_safe_malloc(sizeof(tnn_tensor_t));

	t->num_dims = num_dims;
	if (num_dims > 0) {
		t->dims = tnn_safe_malloc(num_dims * sizeof(size_t));
		memcpy(t->dims, dims, num_dims * sizeof(size_t));
	} else {
		t->dims = NULL;
	}

	size_t total_size = tnn_size(t);
	t->data = tnn_safe_malloc(total_size * sizeof(float));
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
	free(t->data);
	if (t->grad) {
		free(t->grad);
	}
	free(t->dims);
	if (t->context != NULL && t->free_context != NULL) {
		// (forward was executed without backward)
		t->free_context(t->context);
	}
	free(t);
}

tnn_tensor_t *tnn_detach(tnn_tensor_t *t) {
	tnn_tensor_t *detached = tnn_alloc(t->dims, t->num_dims);
	tnn_init_from_memory(detached, t->data);
	return detached;
}

tnn_tensor_t *tnn_detach_free(tnn_tensor_t *t) {
	tnn_tensor_t *detached = tnn_detach(t);
	tnn_free(t);
	return detached;
}

void tnn_init_from_memory(tnn_tensor_t *t, const float *data) {
	size_t total_size = tnn_size(t);
	memcpy(t->data, data, total_size * sizeof(float));
}

void tnn_init_fill(tnn_tensor_t *t, float value) {
	size_t total_size = tnn_size(t);
	memset(t->data, value, total_size * sizeof(float));
}

void tnn_init_randn(tnn_tensor_t *t) {
	size_t total_size = tnn_size(t);
	for (size_t i = 0; i < total_size; i++) {
		// box-muller transform for normal distribution
		float u1 = (float)rand() / (float)RAND_MAX;
		float u2 = (float)rand() / (float)RAND_MAX;
		float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
		t->data[i] = z;
	}
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
	return t->data[0];
}

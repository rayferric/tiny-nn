#include <tnn/tnn.h>

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../devices/devices.h"
#include "../util/safe_malloc.h"

#include "./impl.h"

tnn_tensor_t *tnn_alloc(const size_t *dims, size_t num_dims) {
	return alloc_tensor_on_device(
	    dims, num_dims, device_globals.default_device
	);
}

tnn_tensor_t *tnn_alloc_or_get_state(
    const size_t *dims, size_t num_dims, const char *key, bool *allocated
) {
	return alloc_or_get_state_tensor_on_device(
	    dims, num_dims, device_globals.default_device, key, allocated
	);
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
	t->dev->_backend.buf_free(t->dev, t->data);
	if (t->grad) {
		t->dev->_backend.buf_free(t->dev, t->grad);
	}
	free(t->dims);
	if (t->_ctx != NULL && t->_free_ctx != NULL) {
		// (forward was executed without backward)
		t->_free_ctx(t->_ctx);
	}
	free(t);
}

tnn_tensor_t *_tnn_detach(tnn_tensor_t *t, const char *new_device) {
	tnn_tensor_t *detached =
	    alloc_tensor_on_device(t->dims, t->num_dims, t->dev);
	t->dev->_backend.buf_copy(
	    t->dev, detached->data, t->data, tnn_size(t) * sizeof(float)
	);

	if (new_device != NULL) {
		tnn_device_t *dev = tnn_find_device(new_device);
		ensure_tensor_on_device(detached, dev);
	}

	return detached;
}

tnn_tensor_t *_tnn_detach_free(tnn_tensor_t *t, const char *new_device) {
	tnn_tensor_t *detached = tnn_detach(t, new_device);
	tnn_free(t);
	return detached;
}

void tnn_init_from_memory(tnn_tensor_t *t, const float *data) {
	t->dev->_backend.buf_copy_to_device(
	    t->dev, t->data, data, tnn_size(t) * sizeof(float)
	);
}

void tnn_init_fill(tnn_tensor_t *t, float value) {
	size_t total_size = tnn_size(t);
	float *tmp_buf = (float *)safe_malloc(total_size * sizeof(float));
	for (size_t i = 0; i < total_size; i++) {
		tmp_buf[i] = value;
	}
	t->dev->_backend.buf_copy_to_device(
	    t->dev, t->data, tmp_buf, total_size * sizeof(float)
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
	t->dev->_backend.buf_copy_to_device(
	    t->dev, t->data, tmp_buf, total_size * sizeof(float)
	);
	free(tmp_buf);
}

void tnn_init_xavier(tnn_tensor_t *t, size_t fan_in, size_t fan_out) {
	float limit = sqrtf(6.0f / (fan_in + fan_out));
	size_t total_size = tnn_size(t);
	float *tmp_buf = (float *)safe_malloc(total_size * sizeof(float));
	for (size_t i = 0; i < total_size; i++) {
		float u = (float)rand() / (float)RAND_MAX; // uniform [0,1]
		tmp_buf[i] = u * 2.0f * limit - limit;     // uniform [-limit, limit]
	}
	t->dev->_backend.buf_copy_to_device(
	    t->dev, t->data, tmp_buf, total_size * sizeof(float)
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

	float *data_ptr = NULL;
	float *tmp_data = NULL;

	if (!t->dev->_is_cpu) {
		size_t total_size = tnn_size(t);
		tmp_data = (float *)malloc(total_size * sizeof(float));
		t->dev->_backend.buf_copy_to_host(
		    t->dev, tmp_data, t->data, total_size * sizeof(float)
		);
		data_ptr = tmp_data;
	} else {
		data_ptr = (float *)t->data;
	}

	if (t->num_dims == 0) {
		printf("%.4f", data_ptr[0]);
	} else if (t->num_dims == 1) {
		printf("[");
		for (size_t i = 0; i < t->dims[0]; i++) {
			printf("%.4f", data_ptr[i]);
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
				printf("%.4f", data_ptr[i * t->dims[1] + j]);
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

	if (tmp_data) {
		free(tmp_data);
	}
}

float tnn_item(tnn_tensor_t *t) {
	float value;
	t->dev->_backend.buf_copy_to_host(t->dev, &value, t->data, sizeof(float));
	return value;
}

void tnn_memcpy_data(tnn_tensor_t *t, float *out) {
	t->dev->_backend.buf_copy_to_host(
	    t->dev, out, t->data, tnn_size(t) * sizeof(float)
	);
}

void tnn_memcpy_grad(tnn_tensor_t *t, float *out) {
	assert(t->grad != NULL && "tnn_memcpy_grad: tensor has no grad");
	t->dev->_backend.buf_copy_to_host(
	    t->dev, out, t->grad, tnn_size(t) * sizeof(float)
	);
}

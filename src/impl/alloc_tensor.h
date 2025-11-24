#pragma once

#include <memory.h>

#include <tnn/tnn.h>

static tnn_tensor_t *
_tnn_alloc_tensor(const size_t *dims, size_t num_dims, tnn_tensor_type_t type) {
	tnn_tensor_t *vec = malloc(sizeof(tnn_tensor_t));

	vec->num_dims = num_dims;
	if (num_dims > 0) {
		vec->dims = malloc(num_dims * sizeof(size_t));
		memcpy(vec->dims, dims, num_dims * sizeof(size_t));
	} else {
		vec->dims = NULL;
	}

	size_t total_size = tnn_size(vec);
	vec->data = malloc(total_size * sizeof(float));
	vec->grad = NULL;

	vec->num_parents = 0;
	vec->backward = NULL;

	vec->type = type;

	return vec;
}

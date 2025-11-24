#pragma once

#include <memory.h>

#include <tnn/tnn.h>

static tnn_tensor_t *
_tnn_alloc_tensor(const size_t *dims, size_t num_dims, tnn_tensor_type_t type) {
	tnn_tensor_t *t = malloc(sizeof(tnn_tensor_t));

	t->num_dims = num_dims;
	if (num_dims > 0) {
		t->dims = malloc(num_dims * sizeof(size_t));
		memcpy(t->dims, dims, num_dims * sizeof(size_t));
	} else {
		t->dims = NULL;
	}

	size_t total_size = tnn_size(t);
	t->data = malloc(total_size * sizeof(float));
	t->grad = NULL;

	t->num_parents = 0;
	t->backward = NULL;

	t->type = type;

	return t;
}

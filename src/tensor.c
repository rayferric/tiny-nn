#include <tnn/tnn.h>

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./impl/toposort.h"

void tnn_free(tnn_tensor_t *t, tnn_tensor_type_t types) {
	assert(t != NULL);

	size_t num_nodes;
	tnn_tensor_t **nodes = _tnn_toposort(t, &num_nodes);

	for (size_t i = 0; i < num_nodes; i++) {
		tnn_tensor_t *node = nodes[i];

		if (!(node->type & types)) {
			continue;
		}

		free(node->data);
		if (node->grad) {
			free(node->grad);
		}
		free(node->dims);
		free(node);
	}
	free(nodes);
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
	for (uint8_t i = 0; i < t->num_dims; i++) {
		total_size *= t->dims[i];
	}
	return total_size;
}

#include "./impl/alloc_tensor.h"

tnn_tensor_t *tnn_data(const size_t *dims, size_t num_dims, const float *data) {
	tnn_tensor_t *t = _tnn_alloc_tensor(dims, num_dims, TNN_INPUT);

	size_t total_size = tnn_size(t);
	memcpy(t->data, data, total_size * sizeof(float));

	return t;
}

tnn_tensor_t *tnn_zeros(const size_t *dims, size_t num_dims) {
	tnn_tensor_t *t = _tnn_alloc_tensor(dims, num_dims, TNN_INPUT);

	size_t total_size = tnn_size(t);
	memset(t->data, 0, total_size * sizeof(float));

	return t;
}

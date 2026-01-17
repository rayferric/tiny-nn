#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

static void add_backward(tnn_tensor_t *self) {
	tnn_tensor_t *a = self->parents[0];
	tnn_tensor_t *b = self->parents[1];

	size_t total_size = tnn_size(self);

	if (a->requires_grad) {
		for (size_t i = 0; i < total_size; i++) {
			// d(a+b)/da = 1
			a->grad[i] += self->grad[i];
		}
	}

	if (b->requires_grad) {
		for (size_t i = 0; i < total_size; i++) {
			// d(a+b)/db = 1
			b->grad[i] += self->grad[i];
		}
	}
}

tnn_tensor_t *tnn_add(tnn_tensor_t *a, tnn_tensor_t *b) {
	assert(a != NULL);
	assert(b != NULL);
	assert(a->num_dims == b->num_dims);

	// verify that dimensions match
	for (size_t i = 0; i < a->num_dims; i++) {
		assert(a->dims[i] == b->dims[i]);
	}

	// alloc output with same dims as inputs
	tnn_tensor_t *output = tnn_alloc(a->dims, a->num_dims);

	// output = a + b (element-wise)
	size_t total_size = tnn_size(a);
	for (size_t i = 0; i < total_size; i++) {
		output->data[i] = a->data[i] + b->data[i];
	}

	output->requires_grad = a->requires_grad || b->requires_grad;
	output->parents[0] = a;
	output->parents[1] = b;
	output->num_parents = 2;
	a->num_children++;
	b->num_children++;
	output->backward = add_backward;

	return output;
}

#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

#include "./impl.h"

static void add_backward(tnn_tensor_t *self) {
	tnn_tensor_t *a = self->parents[0];
	tnn_tensor_t *b = self->parents[1];

	size_t total_size = tnn_size(self);

	if (a->requires_grad) {
		// d(a+b)/da = 1
		a->dev->_backend.accum(a->dev, self->grad, a->grad, total_size);
	}

	if (b->requires_grad) {
		// d(a+b)/db = 1
		b->dev->_backend.accum(b->dev, self->grad, b->grad, total_size);
	}
}

tnn_tensor_t *tnn_add(tnn_tensor_t *a, tnn_tensor_t *b) {
	assert(a != NULL);
	assert(b != NULL);
	assert(a->num_dims == b->num_dims);
	assert(a->dev == b->dev);

	// verify that dimensions match
	for (size_t i = 0; i < a->num_dims; i++) {
		assert(a->dims[i] == b->dims[i]);
	}

	// alloc output with same dims as inputs
	tnn_tensor_t *output = alloc_tensor_on_device(a->dims, a->num_dims, a->dev);

	// output = a + b (element-wise)
	a->dev->_backend.add(
	    a->dev, a->data, b->data, output->data, 1, tnn_size(a)
	);

	output->requires_grad = a->requires_grad || b->requires_grad;
	output->parents[0] = a;
	output->parents[1] = b;
	output->num_parents = 2;
	a->num_children++;
	b->num_children++;
	output->_backward = add_backward;

	return output;
}

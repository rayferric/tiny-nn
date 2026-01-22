#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

#include "./impl.h"

static void ce_backward(tnn_tensor_t *self) {
	tnn_tensor_t *target = self->parents[0];
	tnn_tensor_t *pred = self->parents[1];

	size_t batch_size = pred->dims[0];
	size_t num_classes = pred->dims[1];

	if (pred->requires_grad) {
		pred->dev->_backend.ce_bw(
		    pred->dev,
		    pred->data,
		    target->data,
		    self->grad,
		    pred->grad,
		    batch_size,
		    num_classes
		);
	}
}

tnn_tensor_t *tnn_ce(tnn_tensor_t *pred, tnn_tensor_t *target) {
	size_t batch_size = 1;
	for (size_t i = 0; i < pred->num_dims - 1; i++) {
		batch_size *= pred->dims[i];
	}
	size_t num_classes = pred->dims[pred->num_dims - 1];

	size_t batch_size_tgt = 1;
	for (size_t i = 0; i < target->num_dims - 1; i++) {
		batch_size_tgt *= target->dims[i];
	}
	size_t num_classes_tgt = target->dims[target->num_dims - 1];
	assert(
	    batch_size == batch_size_tgt &&
	    "tnn_ce: batch size of pred and target must match"
	);
	assert(
	    num_classes == num_classes_tgt &&
	    "tnn_ce: number of classes of pred and target must match"
	);

	assert(
	    pred->dev == target->dev &&
	    "tnn_ce: pred and target must be on the same device"
	);

	tnn_tensor_t *output = alloc_tensor_on_device(NULL, 0, pred->dev);
	output->requires_grad = pred->requires_grad;

	pred->dev->_backend.ce_fw(
	    pred->dev,
	    pred->data,
	    target->data,
	    output->data,
	    batch_size,
	    num_classes
	);

	output->parents[0] = target;
	output->parents[1] = pred;
	output->num_parents = 2;
	pred->num_children++;
	target->num_children++;

	if (output->requires_grad) {
		output->_backward = ce_backward;
	}

	return output;
}

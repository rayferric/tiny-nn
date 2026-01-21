#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

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
	assert(target->num_dims == 2 && "target must be 2D [batch, num_classes]");
	assert(pred->num_dims == 2 && "pred must be 2D [batch, num_classes]");

	size_t batch_size = pred->dims[0];
	size_t num_classes = pred->dims[1];

	assert(target->dims[0] == batch_size);
	assert(target->dims[1] == num_classes);

	tnn_tensor_t *output = tnn_alloc(NULL, 0);
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

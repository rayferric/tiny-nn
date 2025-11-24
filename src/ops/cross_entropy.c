#include <tnn/tnn.h>

#include <assert.h>
#include <math.h>
#include <stdbool.h>

static void cross_entropy_backward(tnn_tensor_t *self) {
	tnn_tensor_t *target = self->parents[0];
	tnn_tensor_t *pred = self->parents[1];

	assert(target->num_dims == 2);
	assert(pred->num_dims == 2);

	size_t batch_size = pred->dims[0];
	size_t num_classes = pred->dims[1];

	// pred->grad = (softmax(pred) - target) / batch_size
	if (pred->type == TNN_OUTPUT) {
		for (size_t i = 0; i < batch_size; i++) {
			for (size_t j = 0; j < num_classes; j++) {
				float softmax_val = pred->data[i * num_classes + j];
				float target_val = target->data[i * num_classes + j];
				pred->grad[i * num_classes + j] +=
				    (softmax_val - target_val) / (float)batch_size;
			}
		}
	}
}

#include "./impl/alloc_tensor.h"

tnn_tensor_t *tnn_cross_entropy(tnn_tensor_t *pred, tnn_tensor_t *target) {
	assert(target->num_dims == 2 && "target must be 2D [batch, num_classes]");
	assert(pred->num_dims == 2 && "pred must be 2D [batch, num_classes]");

	size_t batch_size = pred->dims[0];
	size_t num_classes = pred->dims[1];

	assert(target->dims[0] == batch_size);
	assert(target->dims[1] == num_classes);

	// allocate scalar output
	tnn_tensor_t *output = _tnn_alloc_tensor(NULL, 0, TNN_OUTPUT);

	float total_loss = 0.0f;

	for (size_t i = 0; i < batch_size; i++) {
		// compute softmax for this sample
		float max_logit = pred->data[i * num_classes];
		for (size_t j = 1; j < num_classes; j++) {
			float logit = pred->data[i * num_classes + j];
			if (logit > max_logit) {
				max_logit = logit;
			}
		}

		float sum_exp = 0.0f;
		for (size_t j = 0; j < num_classes; j++) {
			sum_exp += expf(pred->data[i * num_classes + j] - max_logit);
		}

		// compute cross entropy: -sum(target * log(softmax(pred)))
		for (size_t j = 0; j < num_classes; j++) {
			float softmax_val =
			    expf(pred->data[i * num_classes + j] - max_logit) / sum_exp;
			float target_val = target->data[i * num_classes + j];

			if (target_val > 0.0f) {
				total_loss -= target_val * logf(softmax_val);
			}

			// store softmax in pred->data for backward
			pred->data[i * num_classes + j] = softmax_val;
		}
	}

	output->data[0] = total_loss / (float)batch_size;

	output->parents[0] = target;
	output->parents[1] = pred;
	output->num_parents = 2;
	output->backward = cross_entropy_backward;

	return output;
}

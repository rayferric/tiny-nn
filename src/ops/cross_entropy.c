#include <tnn/tnn.h>

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "../impl/malloc.h"

static void calc_softmax_parts(
    float *out_max_logit,
    float *out_sum_exp,
    tnn_tensor_t *pred,
    size_t i_batch,
    size_t num_classes
) {
	// compute values for softmax
	*out_max_logit = pred->data[i_batch * num_classes];
	for (size_t i_logit = 1; i_logit < num_classes; i_logit++) {
		float logit = pred->data[i_batch * num_classes + i_logit];
		if (logit > *out_max_logit) {
			*out_max_logit = logit;
		}
	}
	*out_sum_exp = 0.0f;
	for (size_t i_logit = 0; i_logit < num_classes; i_logit++) {
		*out_sum_exp +=
		    expf(pred->data[i_batch * num_classes + i_logit] - *out_max_logit);
	}
}

static float softmax_from_parts(float logit, float max_logit, float sum_exp) {
	return expf(logit - max_logit) / sum_exp;
}

typedef struct {
	float *softmax; // [batch_size, num_classes]
} cross_entropy_context_t;

static void cross_entropy_free_context(void *ctx) {
	cross_entropy_context_t *ce_ctx = (cross_entropy_context_t *)ctx;
	free(ce_ctx->softmax);
	free(ce_ctx);
}

static void cross_entropy_backward(tnn_tensor_t *self) {
	tnn_tensor_t *target = self->parents[0];
	tnn_tensor_t *pred = self->parents[1];

	assert(target->num_dims == 2);
	assert(pred->num_dims == 2);
	assert(self->context != NULL);

	cross_entropy_context_t *ctx = (cross_entropy_context_t *)self->context;

	size_t batch_size = pred->dims[0];
	size_t num_classes = pred->dims[1];

	// pred->grad = (softmax(pred) - target) / batch_size
	if (pred->requires_grad) {
		for (size_t i = 0; i < batch_size; i++) {
			for (size_t j = 0; j < num_classes; j++) {
				size_t idx = i * num_classes + j;
				float softmax_val = ctx->softmax[idx];
				float target_val = target->data[idx];
				pred->grad[idx] += self->grad[0] * (softmax_val - target_val) /
				                   (float)batch_size;
			}
		}
	}
}

tnn_tensor_t *tnn_cross_entropy(tnn_tensor_t *pred, tnn_tensor_t *target) {
	assert(target->num_dims == 2 && "target must be 2D [batch, num_classes]");
	assert(pred->num_dims == 2 && "pred must be 2D [batch, num_classes]");

	size_t batch_size = pred->dims[0];
	size_t num_classes = pred->dims[1];

	assert(target->dims[0] == batch_size);
	assert(target->dims[1] == num_classes);

	// allocate scalar output
	tnn_tensor_t *output = tnn_alloc(NULL, 0);
	output->requires_grad = pred->requires_grad;

	// allocate context for storing softmax values
	cross_entropy_context_t *ctx;
	if (output->requires_grad) {
		ctx = tnn_safe_malloc(sizeof(cross_entropy_context_t));
		ctx->softmax =
		    tnn_safe_malloc(batch_size * num_classes * sizeof(float));
	}

	float total_loss = 0.0f;

	for (size_t i = 0; i < batch_size; i++) {
		float max_logit, sum_exp;
		calc_softmax_parts(&max_logit, &sum_exp, pred, i, num_classes);

		// compute cross entropy: -sum(target * log(softmax(pred)))
		for (size_t j = 0; j < num_classes; j++) {
			size_t idx = i * num_classes + j;
			float softmax_val =
			    softmax_from_parts(pred->data[idx], max_logit, sum_exp);
			if (output->requires_grad) {
				ctx->softmax[idx] = softmax_val;
			}
			float target_val = target->data[idx];

			if (target_val > 0.0f) {
				total_loss -= target_val * logf(softmax_val);
			}
		}
	}

	output->data[0] = total_loss / (float)batch_size;

	output->parents[0] = target;
	output->parents[1] = pred;
	output->num_parents = 2;
	pred->num_children++;
	target->num_children++;
	if (output->requires_grad) {
		output->backward = cross_entropy_backward;
		output->context = ctx;
		output->free_context = cross_entropy_free_context;
	}

	return output;
}

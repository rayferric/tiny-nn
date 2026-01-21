#pragma once

#include <math.h>
#include <memory.h>
#include <stdio.h>

#include <tnn/tnn.h>

static void _calc_softmax_parts(
    float *out_max_logit,
    float *out_sum_exp,
    const float *pred,
    size_t i_batch,
    size_t num_classes
) {
	// compute values for softmax
	*out_max_logit = pred[i_batch * num_classes];
	for (size_t i_logit = 1; i_logit < num_classes; i_logit++) {
		float logit = pred[i_batch * num_classes + i_logit];
		if (logit > *out_max_logit) {
			*out_max_logit = logit;
		}
	}
	*out_sum_exp = 0.0f;
	for (size_t i_logit = 0; i_logit < num_classes; i_logit++) {
		*out_sum_exp +=
		    expf(pred[i_batch * num_classes + i_logit] - *out_max_logit);
	}
}

static void ce_fw(
    tnn_device_t *dev,
    const void *pred,
    const void *tgt,
    void *out,
    size_t n,
    size_t c
) {
	const float *pred_f = (const float *)pred;
	const float *target_f = (const float *)tgt;
	float *out_f = (float *)out;

	float total_loss = 0.0f;

	for (size_t i = 0; i < n; i++) {
		const float *logits = pred_f + i * c;
		const float *tgt = target_f + i * c;

		float max_logit, sum_exp;
		_calc_softmax_parts(&max_logit, &sum_exp, pred_f, i, c);

		// compute loss: -sum(target * log(softmax))
		// log(softmax(x)) = log(exp(x - max) / sum_exp) = (x - max) -
		// log(sum_exp)
		float log_sum_exp = logf(sum_exp);
		for (size_t j = 0; j < c; j++) {
			float log_softmax = (logits[j] - max_logit) - log_sum_exp;
			total_loss -= tgt[j] * log_softmax;
		}
	}

	*out_f = total_loss / (float)n;
}

static void ce_bw(
    tnn_device_t *dev,
    const void *pred,
    const void *tgt,
    const void *out_grad,
    void *pred_grad,
    size_t n,
    size_t c
) {
	const float *pred_f = (const float *)pred;
	const float *target_f = (const float *)tgt;
	const float *out_grad_f = (const float *)out_grad;
	float *pred_grad_f = (float *)pred_grad;

	float scale = out_grad_f[0] / (float)n;

	for (size_t i = 0; i < n; i++) {
		const float *logits = pred_f + i * c;
		const float *tgt = target_f + i * c;
		float *grad = pred_grad_f + i * c;

		float max_logit, sum_exp;
		_calc_softmax_parts(&max_logit, &sum_exp, pred_f, i, c);

		// grad = scale * (softmax - target)
		for (size_t j = 0; j < c; j++) {
			float softmax = expf(logits[j] - max_logit) / sum_exp;
			grad[j] += scale * (softmax - tgt[j]);
		}
	}
}

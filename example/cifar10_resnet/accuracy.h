#pragma once

#include <tnn/tnn.h>

// shapes = (N, C)
// pred contains logits, target is one-hot encoded
float accuracy(tnn_tensor_t *pred, tnn_tensor_t *target) {
	size_t batch_size = pred->dims[0];
	size_t num_classes = pred->dims[1];

	float *pred_data = malloc(batch_size * num_classes * sizeof(float));
	float *target_data = malloc(batch_size * num_classes * sizeof(float));
	tnn_memcpy_data(pred, pred_data);
	tnn_memcpy_data(target, target_data);

	size_t correct = 0;
	for (size_t n = 0; n < batch_size; n++) {
		size_t pred_label = 0;
		float max_logit = pred_data[n * num_classes];
		for (size_t c = 1; c < num_classes; c++) {
			float logit = pred_data[n * num_classes + c];
			if (logit > max_logit) {
				max_logit = logit;
				pred_label = c;
			}
		}
		size_t true_label = 0;
		for (size_t c = 0; c < num_classes; c++) {
			if (target_data[n * num_classes + c] > 0.5f) {
				true_label = c;
				break;
			}
		}
		if (pred_label == true_label) {
			correct++;
		}
	}

	free(pred_data);
	free(target_data);

	return (float)correct / (float)batch_size;
}

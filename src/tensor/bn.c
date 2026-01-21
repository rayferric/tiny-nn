#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

#include "../util/safe_malloc.h"

typedef struct {
	size_t NHW;
	size_t C;
	float momentum;
	float test;
	void *running_var; // for test mode; internal tensor data; DO NOT FREE
	void *batch_var;
	tnn_device_t *dev;
} bn_context_t;

static void bn_free_context(void *ctx) {
	bn_context_t *bn_ctx = (bn_context_t *)ctx;
	if (bn_ctx->batch_var) {
		bn_ctx->dev->_backend.buf_free(bn_ctx->dev, bn_ctx->batch_var);
	}
	free(bn_ctx);
}

static void bn_backward(tnn_tensor_t *self) {
	tnn_tensor_t *input = self->parents[0];

	if (!input->requires_grad) {
		return;
	}

	assert(self->_ctx != NULL);
	bn_context_t *ctx = (bn_context_t *)self->_ctx;

	size_t NHW = ctx->NHW;
	size_t C = ctx->C;
	size_t NHWC = NHW * C;

	void *tmp_grad_sum =
	    self->dev->_backend.buf_alloc(self->dev, C * sizeof(float));
	void *tmp_grad_x_norm_sum =
	    self->dev->_backend.buf_alloc(self->dev, C * sizeof(float));
	self->dev->_backend.bn_bw(
	    self->dev,
	    self->grad,
	    self->data,
	    input->grad,
	    ctx->running_var,
	    ctx->batch_var,
	    tmp_grad_sum,
	    tmp_grad_x_norm_sum,
	    NHW,
	    C,
	    ctx->test
	);
	self->dev->_backend.buf_free(self->dev, tmp_grad_sum);
	self->dev->_backend.buf_free(self->dev, tmp_grad_x_norm_sum);
}

tnn_tensor_t *tnn_bn(tnn_tensor_t *input, float momentum, bool test) {
	assert(input != NULL);
	assert(input->num_dims >= 4); // [..., H, W, C]
	assert(momentum >= 0.0f && momentum <= 1.0f);

	size_t N = 1; // batch size
	for (size_t i = 0; i < input->num_dims - 3; i++) {
		N *= input->dims[i];
	}
	size_t H = input->dims[input->num_dims - 3]; // height
	size_t W = input->dims[input->num_dims - 2]; // width
	size_t C = input->dims[input->num_dims - 1]; // channels
	size_t NHW = N * H * W;
	size_t NHWC = NHW * C;

	// get or create running statistics as buffers (state without grad)
	size_t stats_dims[1] = {C};
	bool running_mean_created = false;
	bool running_var_created = false;

	tnn_tensor_t *running_mean =
	    tnn_alloc_or_get_state(stats_dims, 1, "bn/mean", &running_mean_created);
	tnn_tensor_t *running_var =
	    tnn_alloc_or_get_state(stats_dims, 1, "bn/var", &running_var_created);
	if (running_mean_created) {
		tnn_init_fill(running_mean, 0);
	}
	if (running_var_created) {
		tnn_init_fill(running_var, 1);
	}

	// alloc output with same dims as input
	tnn_tensor_t *output = tnn_alloc(input->dims, input->num_dims);

	// create context for backward pass
	bn_context_t *ctx = safe_malloc(sizeof(bn_context_t));
	ctx->NHW = NHW;
	ctx->C = C;
	ctx->momentum = momentum;
	ctx->test = test;
	ctx->running_var = running_var->data;
	if (!test) {
		ctx->batch_var =
		    input->dev->_backend.buf_alloc(input->dev, C * sizeof(float));
	} else {
		ctx->batch_var = NULL;
	}
	ctx->dev = input->dev;

	// forward pass
	void *tmp_batch_mean =
	    input->dev->_backend.buf_alloc(input->dev, C * sizeof(float));
	input->dev->_backend.bn_fw(
	    input->dev,
	    input->data,
	    output->data,
	    running_mean->data,
	    running_var->data,
	    tmp_batch_mean,
	    ctx->batch_var,
	    NHW,
	    C,
	    momentum,
	    test
	);
	input->dev->_backend.buf_free(input->dev, tmp_batch_mean);

	output->requires_grad = input->requires_grad;
	output->parents[0] = input;
	output->num_parents = 1;
	input->num_children++;
	output->_backward = bn_backward;
	output->_ctx = ctx;
	output->_free_ctx = bn_free_context;

	return output;
}

#include <tnn/tnn.h>

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#include "../impl/malloc.h"

typedef struct {
	size_t NHW;
	size_t C;
	float momentum;
	float test;
	float *running_var; // for test mode; internal tensor data; DO NOT FREE
	float *batch_var;
} bn_context_t;

static void bn_free_context(void *ctx) {
	bn_context_t *bn_ctx = (bn_context_t *)ctx;
	free(bn_ctx->batch_var);
	free(bn_ctx);
}

static void bn_backward(tnn_tensor_t *self) {
	tnn_tensor_t *input = self->parents[0];

	if (!input->requires_grad) {
		return;
	}

	assert(self->context != NULL);
	bn_context_t *ctx = (bn_context_t *)self->context;

	size_t NHW = ctx->NHW;
	size_t C = ctx->C;
	size_t NHWC = NHW * C;

	for (size_t c = 0; c < C; c++) {
		if (ctx->test) {
			//   x' = (x - u) / s
			//   x' = c / s
			// where: c = x - mean
			//   dx'/dx = dx'/dc * dc/dx
			//   dx'/dc = 1/s
			//   dc/dx = 1
			// thus:
			//   dx'/dx = 1/s
			// finally apply chain rule with incoming gradient dL/dx':
			//   dL/dx = dL/dx' / s

			float var = ctx->running_var[c];
			float std_inv = 1.0f / sqrtf(var + 1e-5);

			for (size_t idx = c; idx < NHWC; idx += C) {
				input->grad[idx] += self->grad[idx] * std_inv;
			}
		} else {
			// clang-format off
			// in training, the gradient flows through immediate stats of the
			// batch:
			//   x' = (x - u) / s
			//   u = SUM[i]{x[i]} / N
			//   s = sqrt(SUM[i]{(x[i] - u)^2} / N)
            // where: N = NHW (all batch dims together)
			//
			//   dL/dx = dL/dx' * dx'/dx
			//   dL/dx' -> KNOWN; = self->grad
			// from quotient rule:
			//   dx'/dx = (d(x-u)/dx*s - (x-u)*ds/dx) / s^2
            // with indices:
            //   dx'[j]/dx[i] = (d(x[j]-u)/dx[i]*s - (x[j]-u)*ds/dx[i]) / s^2
            //
            // x minus mean gradient:
			//   d(x[j]-u)/dx[i] = dx[j]/dx[i] - du/dx[i]
			// note: dx[j]/dx[i] is the identity matrix I (ones for i=j cells)
			//   du/dx[i] = 1/N
			// remember: this is a vector of
			// derivatives wrt each element x[i], and other elements are
			// independent -> they zero-out
            //   d(x[j]-u)/dx[i] = I[i,j] - 1/N
            //
            // standard deviation gradient:
			//   ds/dx[i] = d(sqrt(SUM[j]{(x[j]-u)^2}/N))/dx[i]
			//            = 1/(2*sqrt(SUM[j]{(x[j]-u)^2}/N)) * (1/N) * SUM[j]{d((x[j]-u)^2)/dx[i]}
            //   d((x[j]-u)^2)/dx[i] = 2(x[j]-u) * d(x[j]-u)/dx[i]
            //   d(x[j]-u)/dx[i] -> ALREADY COMPUTED
            // plugging d(x[j]-u)/dx[i] into ds/dx[i]:
            //   ds/dx[i] = 1/(2*sqrt(SUM[j]{(x[j]-u)^2}/N)) * (1/N) * SUM[j]{2(x[j]-u)*(I[i,j]-1/N)}
            // where: I is identity matrix
            //   ds/dx[i] = SUM[j]{2(x[j]-u)*(I[i,j]-1/N)} / (2N*sqrt(SUM[j]{(x[j]-u)^2}/N))
            //            = SUM[j]{(x[j]-u)*(I[i,j]-1/N)} / (N*sqrt(SUM[j]{(x[j] - u)^2}/N))
            //            = [ SUM[j]{(x[j]-u)*I[i,j]} + SUM[j]{(x[j]-u)*(-1/N)} ] / ...
            //   SUM[j]{(x[j] - u) * I[i,j]} = x[i] - u
            // because: I[i,j]=1 only for i=j
            //   SUM[j]{(x[j] - u) * (-1/N)} = 0
            // because: summing all centered elements = 0
            // also notice:
            //   sqrt(SUM[j]{(x[j] - u)^2}/N) = s
            // thus:
            //   ds/dx[i] = (x[i] - u) / Ns
            //
            // finally:
            //   dx'[j]/dx[i] = ((I[i,j] - 1/N)*s - (x[j]-u)*((x[i] - u) / Ns)) / s^2
            //                = (I[i,j]-1/N)/s - (x[j]-u)*(x[i]-u)/Ns^3
            // plugging into full loss formula:
            //   dL/dx[i] = SUM[j]{dL/dx'[j] * dx'[j]/dx[i]}
            //            = SUM[j]{dL/dx'[j] * [(I[i,j]-1/N)/s - (x[j]-u)*(x[i]-u)/Ns^3]}
            // split the sum:
            //            = SUM[j]{dL/dx'[j] * (I[i,j]-1/N)/s} - SUM[j]{dL/dx'[j] * (x[j]-u)*(x[i]-u)/Ns^3}
            // first term:
            //   (1/s) * [dL/dx'[i] - (1/N)*SUM[j]{dL/dx'[j]}] ...
            // second term (factor out (x[i]-u)/Ns^2):
            //   ... - (x[i]-u)/Ns^2 * SUM[j]{dL/dx'[j] * (x[j]-u)/s}
            // combine and use x'[i] = (x[i]-u)/s:
            //   dL/dx[i] = dL/dx'[i] * (1/s) - (1/N)*(1/s)*SUM[j]{dL/dx'[j]} 
            //           - x'[i] * (1/Ns) * SUM[j]{dL/dx'[j] * x'[j]}
            //
            // let K = 1/Ns:
            //   dL/dx[i] = dL/dx'[i] * (1/s) - ( SUM[j]{dL/dx'[j]} + x'[i]*SUM[j]{dL/dx'[j]*x'[j]} ) * K
            //
            // in the following implementation:
            //   x_norm = self->data[idx] = x'[j]
            //   sum_grad = SUM[j]{dL/dx'[j]}
            //   sum_grad_x_norm = SUM[j]{dL/dx'[j]*x'[j]}
            //   k = K
			// clang-format on

			float var = ctx->batch_var[c];
			float std_inv = 1.0f / sqrtf(var + 1e-5f);

			float sum_grad = 0.0f;
			float sum_grad_x_norm = 0.0f;
			for (size_t idx = c; idx < NHWC; idx += C) {
				sum_grad += self->grad[idx];
				sum_grad_x_norm += self->grad[idx] * self->data[idx];
			}

			float k = std_inv / NHW;
			for (size_t idx = c; idx < NHWC; idx += C) {
				input->grad[idx] +=
				    self->grad[idx] * std_inv -
				    (sum_grad + self->data[idx] * sum_grad_x_norm) * k;
			}
		}
	}
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
	bn_context_t *ctx = tnn_safe_malloc(sizeof(bn_context_t));
	ctx->NHW = NHW;
	ctx->C = C;
	ctx->momentum = momentum;
	ctx->test = test;
	ctx->running_var = running_var->data;
	if (!test) {
		ctx->batch_var = tnn_safe_malloc(C * sizeof(float));
	}

	// forward pass: compute batch statistics and normalize
	for (size_t c = 0; c < C; c++) {
		float mean, var;
		if (test) {
			mean = running_mean->data[c];
			var = running_var->data[c];
		} else {
			// compute mean for this channel
			float sum = 0.0f;
			for (size_t idx = c; idx < NHWC; idx += C) {
				sum += input->data[idx];
			}
			mean = sum / NHW;

			// compute variance
			float var_sum = 0.0f;
			for (size_t idx = c; idx < NHWC; idx += C) {
				float diff = input->data[idx] - mean;
				var_sum += diff * diff;
			}
			var = var_sum / NHW;

			// update running stats
			running_mean->data[c] =
			    momentum * running_mean->data[c] + (1.0f - momentum) * mean;
			running_var->data[c] =
			    momentum * running_var->data[c] + (1.0f - momentum) * var;

			// pass immediate stats to backward for use in train mode
			ctx->batch_var[c] = var;
		}

		// normalize
		float std_inv = 1 / sqrtf(var + 1e-5);
		for (size_t idx = c; idx < NHWC; idx += C) {
			output->data[idx] = (input->data[idx] - mean) * std_inv;
		}
	}

	output->requires_grad = input->requires_grad;
	output->parents[0] = input;
	output->num_parents = 1;
	input->num_children++;
	output->backward = bn_backward;
	output->context = ctx;
	output->free_context = bn_free_context;

	return output;
}

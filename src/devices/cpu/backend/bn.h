#pragma once

#include <math.h>
#include <memory.h>
#include <stdio.h>

#include <tnn/tnn.h>

// input/output: [NHW, C]
// running_mean/running_var: [C]
// batch_mean/batch_var: [C] - written if !test
static void bn_fw(
    tnn_device_t *dev,
    const void *input,
    void *output,
    void *running_mean,
    void *running_var,
    void *tmp_batch_mean,
    void *batch_var,
    size_t nhw,
    size_t c,
    float momentum,
    bool test
) {
	const float *in_f = (const float *)input;
	float *out_f = (float *)output;
	float *run_mean_f = (float *)running_mean;
	float *run_var_f = (float *)running_var;
	float *batch_mean_f = (float *)tmp_batch_mean;
	float *batch_var_f = (float *)batch_var;

	size_t total = nhw * c;

	for (size_t channel = 0; channel < c; channel++) {
		float mean, var;

		if (test) {
			// use running statistics
			mean = run_mean_f[channel];
			var = run_var_f[channel];
		} else {
			// compute batch statistics
			float sum = 0.0f;
			for (size_t idx = channel; idx < total; idx += c) {
				sum += in_f[idx];
			}
			mean = sum / nhw;

			float var_sum = 0.0f;
			for (size_t idx = channel; idx < total; idx += c) {
				float diff = in_f[idx] - mean;
				var_sum += diff * diff;
			}
			var = var_sum / nhw;

			// update running statistics with EMA
			run_mean_f[channel] =
			    momentum * run_mean_f[channel] + (1.0f - momentum) * mean;
			run_var_f[channel] =
			    momentum * run_var_f[channel] + (1.0f - momentum) * var;

			// save batch statistics for backward
			batch_mean_f[channel] = mean;
			batch_var_f[channel] = var;
		}

		// normalize
		float std_inv = 1.0f / sqrtf(var + 1e-5f);
		for (size_t idx = channel; idx < total; idx += c) {
			out_f[idx] = (in_f[idx] - mean) * std_inv;
		}
	}
}

static void bn_bw(
    tnn_device_t *dev,
    const void *out_grad,      // [NHW, C]
    const void *out_data,      // [NHW, C]
    void *in_grad,             // [NHW, C]
    const void *running_var,   // [C]
    const void *batch_var,     // [C]
    void *tmp_grad_sum,        // [C]
    void *tmp_grad_x_norm_sum, // [C]
    size_t nhw,
    size_t c,
    bool test
) {
	const float *out_grad_f = (const float *)out_grad;
	const float *in_norm_f = (const float *)out_data;
	float *in_grad_f = (float *)in_grad;
	const float *run_var_f = (const float *)running_var;
	const float *batch_var_f = (const float *)batch_var;
	float *tmp_grad_sum_f = (float *)tmp_grad_sum;
	float *tmp_grad_x_norm_sum_f = (float *)tmp_grad_x_norm_sum;

	size_t total = nhw * c;

	if (test) {
		// simple case: just scale by 1/std
		for (size_t channel = 0; channel < c; channel++) {
			float var = run_var_f[channel];
			float std_inv = 1.0f / sqrtf(var + 1e-5f);

			for (size_t idx = channel; idx < total; idx += c) {
				in_grad_f[idx] += out_grad_f[idx] * std_inv;
			}
		}
	} else {
		// compute tmp sums first
		for (size_t channel = 0; channel < c; channel++) {
			float sum_grad = 0.0f;
			float sum_grad_x_norm = 0.0f;

			for (size_t idx = channel; idx < total; idx += c) {
				sum_grad += out_grad_f[idx];
				sum_grad_x_norm += out_grad_f[idx] * in_norm_f[idx];
			}

			tmp_grad_sum_f[channel] = sum_grad;
			tmp_grad_x_norm_sum_f[channel] = sum_grad_x_norm;
		}

		// apply gradient formula
		for (size_t channel = 0; channel < c; channel++) {
			float var = batch_var_f[channel];
			float std_inv = 1.0f / sqrtf(var + 1e-5f);
			float k = std_inv / nhw;

			float sum_grad = tmp_grad_sum_f[channel];
			float sum_grad_x_norm = tmp_grad_x_norm_sum_f[channel];

			for (size_t idx = channel; idx < total; idx += c) {
				in_grad_f[idx] +=
				    out_grad_f[idx] * std_inv -
				    (sum_grad + in_norm_f[idx] * sum_grad_x_norm) * k;
			}
		}
	}
}

// TEST-TIME BACKWARD PASS DERIVATION
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

// TRAINING-TIME BACKWARD PASS DERIVATION
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

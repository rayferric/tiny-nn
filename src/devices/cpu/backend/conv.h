#pragma once

#include <memory.h>
#include <stdio.h>

#include <tnn/tnn.h>

// input: [batch, h_in, w_in, c_in]
// weight: [c_out, k, k, c_in]
// output: [batch, h_out, w_out, c_out]
static void conv_fw(
    tnn_device_t *dev,
    const void *input,
    const void *weight,
    void *output,
    size_t batch,
    size_t h_in,
    size_t w_in,
    size_t c_in,
    size_t h_out,
    size_t w_out,
    size_t c_out,
    size_t kernel_size,
    size_t stride,
    size_t padding
) {
	const float *in_f = (const float *)input;
	const float *w_f = (const float *)weight;
	float *out_f = (float *)output;

	// clang-format off
	for (size_t b = 0; b < batch; b++) {
	for (size_t ho = 0; ho < h_out; ho++) {
	for (size_t wo = 0; wo < w_out; wo++) {
	for (size_t co = 0; co < c_out; co++) {
		float sum = 0.0f;

		for (size_t ki = 0; ki < kernel_size; ki++) {
		for (size_t kj = 0; kj < kernel_size; kj++) {
			int hi = ho * stride + ki - padding;
			int wi = wo * stride + kj - padding;

			if (hi >= 0 && hi < (int)h_in && wi >= 0 && wi < (int)w_in) {
				for (size_t ci = 0; ci < c_in; ci++) {
					size_t in_idx = ((b * h_in + hi) * w_in + wi) * c_in + ci;
					size_t w_idx = ((co * kernel_size + ki) * kernel_size + kj) * c_in + ci;
					sum += in_f[in_idx] * w_f[w_idx];
				}
			}
		}
		}

		size_t out_idx = ((b * h_out + ho) * w_out + wo) * c_out + co;
		out_f[out_idx] = sum;
	}
	}
	}
	}
	// clang-format on
}

// out_grad: [batch, h_out, w_out, c_out]
// weight: [c_out, k, k, c_in]
// in_grad: [batch, h_in, w_in, c_in]
static void conv_bw_input(
    tnn_device_t *dev,
    const void *out_grad,
    const void *weight,
    void *in_grad,
    size_t batch,
    size_t h_in,
    size_t w_in,
    size_t c_in,
    size_t h_out,
    size_t w_out,
    size_t c_out,
    size_t kernel_size,
    size_t stride,
    size_t padding
) {
	const float *out_grad_f = (const float *)out_grad;
	const float *w_f = (const float *)weight;
	float *in_grad_f = (float *)in_grad;

	// clang-format off
	for (size_t b = 0; b < batch; b++) {
	for (size_t ho = 0; ho < h_out; ho++) {
	for (size_t wo = 0; wo < w_out; wo++) {
	for (size_t co = 0; co < c_out; co++) {
		size_t out_idx = ((b * h_out + ho) * w_out + wo) * c_out + co;
		float grad_val = out_grad_f[out_idx];

		for (size_t ki = 0; ki < kernel_size; ki++) {
		for (size_t kj = 0; kj < kernel_size; kj++) {
			int hi = ho * stride + ki - padding;
			int wi = wo * stride + kj - padding;

			if (hi >= 0 && hi < (int)h_in && wi >= 0 && wi < (int)w_in) {
				for (size_t ci = 0; ci < c_in; ci++) {
					size_t in_idx = ((b * h_in + hi) * w_in + wi) * c_in + ci;
					size_t w_idx = ((co * kernel_size + ki) * kernel_size + kj) * c_in + ci;
					in_grad_f[in_idx] += grad_val * w_f[w_idx];
				}
			}
		}
		}
	}
	}
	}
	}
	// clang-format on
}

// input: [batch, h_in, w_in, c_in]
// out_grad: [batch, h_out, w_out, c_out]
// weight_grad: [c_out, k, k, c_in]
static void conv_bw_weight(
    tnn_device_t *dev,
    const void *input,
    const void *out_grad,
    void *weight_grad,
    size_t batch,
    size_t h_in,
    size_t w_in,
    size_t c_in,
    size_t h_out,
    size_t w_out,
    size_t c_out,
    size_t kernel_size,
    size_t stride,
    size_t padding
) {
	const float *in_f = (const float *)input;
	const float *out_grad_f = (const float *)out_grad;
	float *w_grad_f = (float *)weight_grad;

	// clang-format off
	for (size_t b = 0; b < batch; b++) {
	for (size_t ho = 0; ho < h_out; ho++) {
	for (size_t wo = 0; wo < w_out; wo++) {
	for (size_t co = 0; co < c_out; co++) {
		size_t out_idx = ((b * h_out + ho) * w_out + wo) * c_out + co;
		float grad_val = out_grad_f[out_idx];

		for (size_t ki = 0; ki < kernel_size; ki++) {
		for (size_t kj = 0; kj < kernel_size; kj++) {
			int hi = ho * stride + ki - padding;
			int wi = wo * stride + kj - padding;

			if (hi >= 0 && hi < (int)h_in && wi >= 0 && wi < (int)w_in) {
				for (size_t ci = 0; ci < c_in; ci++) {
					size_t in_idx = ((b * h_in + hi) * w_in + wi) * c_in + ci;
					size_t w_idx = ((co * kernel_size + ki) * kernel_size + kj) * c_in + ci;
					w_grad_f[w_idx] += grad_val * in_f[in_idx];
				}
			}
		}
		}
	}
	}
	}
	}
	// clang-format on
}

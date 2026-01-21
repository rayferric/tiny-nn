#pragma once

#include <math.h>
#include <memory.h>
#include <stdio.h>

#include <tnn/tnn.h>

static void *buf_alloc(tnn_device_t *dev, size_t bytes) {
	void *ptr = malloc(bytes);
	if (!ptr) {
		fprintf(stderr, "cpu_alloc: out of memory\n");
		exit(1);
	}
	return ptr;
}

static void buf_free(tnn_device_t *dev, void *ptr) {
	free(ptr);
}

static void
buf_copy(tnn_device_t *dev, void *dst, const void *src, size_t bytes) {
	memcpy(dst, src, bytes);
}

static void buf_copy_to_device(
    tnn_device_t *dev, void *dst, const void *src, size_t bytes
) {
	memcpy(dst, src, bytes);
}

static void
buf_copy_to_host(tnn_device_t *dev, void *dst, const void *src, size_t bytes) {
	memcpy(dst, src, bytes);
}

#include "./matmul.h"

static void
xavier(tnn_device_t *dev, void *out, size_t sz, size_t fan_in, size_t fan_out) {
	float limit = sqrtf(6.0f / (fan_in + fan_out));
	float *fdata = (float *)out;
	for (size_t i = 0; i < sz; i++) {
		float u = (float)rand() / (float)RAND_MAX; // uniform [0,1]
		fdata[i] = u * 2.0f * limit - limit;       // uniform [-limit, limit]
	}
}

// out[outer, inner] = a[outer, inner] + b[inner]
static void
add(tnn_device_t *dev,
    const void *a,
    const void *b,
    void *out,
    size_t outer,
    size_t inner) {
	const float *a_f = (const float *)a;
	const float *b_f = (const float *)b;
	float *out_f = (float *)out;

	for (size_t i_outer = 0; i_outer < outer; i_outer++) {
		for (size_t i_inner = 0; i_inner < inner; i_inner++) {
			out_f[i_outer * inner + i_inner] =
			    a_f[i_outer * inner + i_inner] + b_f[i_inner];
		}
	}
}

// out[i] += in[i]
static void accum(tnn_device_t *dev, const void *in, void *out, size_t n) {
	const float *in_f = (const float *)in;
	float *out_f = (float *)out;

	for (size_t i = 0; i < n; i++) {
		out_f[i] += in_f[i];
	}
}

// out[outer, inner] = sum(in[outer, reduced, inner], axis=1)
// if accum, add to out, else overwrite
// if reverse, broadcast instead of reducing:
//   out[outer, reduced, inner] = in[outer, inner]
static void
sum(tnn_device_t *dev,
    const void *in,
    void *out,
    size_t outer,
    size_t reduced,
    size_t inner,
    float scale,
    bool accum,
    bool reverse) {
	const float *in_f = (const float *)in;
	float *out_f = (float *)out;

	for (size_t i_outer = 0; i_outer < outer; i_outer++) {
		for (size_t i_inner = 0; i_inner < inner; i_inner++) {
			if (!reverse) {
				float sum = 0.0f;
				for (size_t i_red = 0; i_red < reduced; i_red++) {
					sum += in_f[(i_outer * reduced + i_red) * inner + i_inner];
				}
				if (accum) {
					out_f[i_outer * inner + i_inner] += sum * scale;
				} else {
					out_f[i_outer * inner + i_inner] = sum * scale;
				}
			} else {
				float val = in_f[i_outer * inner + i_inner] * scale;
				for (size_t i_red = 0; i_red < reduced; i_red++) {
					if (accum) {
						out_f[(i_outer * reduced + i_red) * inner + i_inner] +=
						    val;
					} else {
						out_f[(i_outer * reduced + i_red) * inner + i_inner] =
						    val;
					}
				}
			}
		}
	}
}

static void relu_fw(tnn_device_t *dev, const void *in, void *out, size_t n) {
	const float *in_f = (const float *)in;
	float *out_f = (float *)out;
	for (size_t i = 0; i < n; i++) {
		out_f[i] = in_f[i] > 0.0f ? in_f[i] : 0.0f;
	}
}

static void relu_bw(
    tnn_device_t *dev,
    const void *out_data,
    const void *out_grad,
    void *in_grad,
    size_t n
) {
	const float *out_data_f = (const float *)out_data;
	const float *out_grad_f = (const float *)out_grad;
	float *in_grad_f = (float *)in_grad;
	for (size_t i = 0; i < n; i++) {
		if (out_data_f[i] > 0.0f) {
			in_grad_f[i] += out_grad_f[i];
		}
	}
}

#include "./bn.h"
#include "./ce.h"
#include "./conv.h"

void adamw(
    tnn_device_t *dev,
    void *param_data,
    void *param_grad,
    void *m1_data,
    void *m2_data,
    void *timestep_data,
    size_t param_size,
    float lr,
    float b1,
    float b2,
    float eps,
    float wd
) {
	// increment timestep
	float t = ((float *)timestep_data)[0] + 1.0f;
	((float *)timestep_data)[0] = t;

	// pre-compute bias correction factors
	float bias1 = 1.0f - powf(b1, t);
	float bias2 = 1.0f - powf(b2, t);

	for (size_t j = 0; j < param_size; j++) {
		float grad = ((float *)param_grad)[j];

		// update moments
		((float *)m1_data)[j] = b1 * ((float *)m1_data)[j] + (1.0f - b1) * grad;
		((float *)m2_data)[j] =
		    b2 * ((float *)m2_data)[j] + (1.0f - b2) * grad * grad;

		// correct moment biases
		float m_hat = ((float *)m1_data)[j] / bias1;
		float v_hat = ((float *)m2_data)[j] / bias2;

		// update
		((float *)param_data)[j] -=
		    lr * (m_hat / (sqrtf(v_hat) + eps) + wd * ((float *)param_data)[j]);
	}
}

static _tnn_backend_t backend = {
    .buf_alloc = buf_alloc,
    .buf_free = buf_free,
    .buf_copy = buf_copy,
    .buf_copy_to_host = buf_copy_to_host,
    .buf_copy_to_device = buf_copy_to_device,
    .matmul = matmul,
    .xavier = xavier,
    .add = add,
    .accum = accum,
    .sum = sum,
    .relu_fw = relu_fw,
    .relu_bw = relu_bw,
    .ce_fw = ce_fw,
    .ce_bw = ce_bw,
    .conv_fw = conv_fw,
    .conv_bw_input = conv_bw_input,
    .conv_bw_weight = conv_bw_weight,
    .bn_fw = bn_fw,
    .bn_bw = bn_bw,
    .adamw = adamw
};

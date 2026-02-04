#pragma once

#include <memory.h>
#include <stdio.h>

#include <tnn/tnn.h>

#ifdef ENABLE_AVX2
#include <immintrin.h>
#endif

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

			size_t in_base_idx = ((b * h_in + hi) * w_in + wi) * c_in;
			size_t w_base_idx = ((co * kernel_size + ki) * kernel_size + kj) * c_in;

			if (hi >= 0 && hi < (int)h_in && wi >= 0 && wi < (int)w_in) {
				size_t ci = 0;

#ifdef ENABLE_AVX2
				__m256 sum_vec = _mm256_setzero_ps();

				// main vectorized loop - process 8 at a time
				for (; ci + 8 <= c_in; ci += 8) {
					__m256 in_vec = _mm256_loadu_ps(&in_f[in_base_idx + ci]);
					__m256 w_vec = _mm256_loadu_ps(&w_f[w_base_idx + ci]);
					sum_vec = _mm256_fmadd_ps(in_vec, w_vec, sum_vec); // fused multiply-add
				}

				// horizontal sum to get scalar
				__m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
				__m128 sum_low = _mm256_castps256_ps128(sum_vec);
				sum_low = _mm_add_ps(sum_low, sum_high);
				sum_low = _mm_hadd_ps(sum_low, sum_low);
				sum_low = _mm_hadd_ps(sum_low, sum_low);
				sum += _mm_cvtss_f32(sum_low);
#endif

				for (; ci < c_in; ci++) {
					sum += in_f[in_base_idx + ci] * w_f[w_base_idx + ci];
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
#ifdef ENABLE_AVX2
		__m256 grad_vec = _mm256_set1_ps(grad_val);
#endif

		for (size_t ki = 0; ki < kernel_size; ki++) {
		for (size_t kj = 0; kj < kernel_size; kj++) {
			int hi = ho * stride + ki - padding;
			int wi = wo * stride + kj - padding;

			if (hi >= 0 && hi < (int)h_in && wi >= 0 && wi < (int)w_in) {
				size_t in_base_idx = ((b * h_in + hi) * w_in + wi) * c_in;
                size_t w_base_idx = ((co * kernel_size + ki) * kernel_size + kj) * c_in;

				size_t ci = 0;
#ifdef ENABLE_AVX2
                for (; ci + 8 <= c_in; ci += 8) {
                    __m256 w_vec = _mm256_loadu_ps(&w_f[w_base_idx + ci]);
                    __m256 in_grad_vec = _mm256_loadu_ps(&in_grad_f[in_base_idx + ci]);
                    in_grad_vec = _mm256_fmadd_ps(grad_vec, w_vec, in_grad_vec);
                    _mm256_storeu_ps(&in_grad_f[in_base_idx + ci], in_grad_vec);
                }
#endif

                for (; ci < c_in; ci++) {
                    in_grad_f[in_base_idx + ci] += grad_val * w_f[w_base_idx + ci];
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
#ifdef ENABLE_AVX2
		__m256 grad_vec = _mm256_set1_ps(grad_val);
#endif

		for (size_t ki = 0; ki < kernel_size; ki++) {
		for (size_t kj = 0; kj < kernel_size; kj++) {
			int hi = ho * stride + ki - padding;
			int wi = wo * stride + kj - padding;

			if (hi >= 0 && hi < (int)h_in && wi >= 0 && wi < (int)w_in) {
				size_t in_base_idx = ((b * h_in + hi) * w_in + wi) * c_in;
                size_t w_base_idx = ((co * kernel_size + ki) * kernel_size + kj) * c_in;

				size_t ci = 0;
#ifdef ENABLE_AVX2
                for (; ci + 8 <= c_in; ci += 8) {
                    __m256 in_vec = _mm256_loadu_ps(&in_f[in_base_idx + ci]);
					__m256 w_grad_vec = _mm256_loadu_ps(&w_grad_f[w_base_idx + ci]);
					w_grad_vec = _mm256_fmadd_ps(grad_vec, in_vec, w_grad_vec);
					_mm256_storeu_ps(&w_grad_f[w_base_idx + ci], w_grad_vec);
				}
#endif
				
				for (; ci < c_in; ci++) {
					w_grad_f[w_base_idx + ci] += grad_val * in_f[in_base_idx + ci];
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

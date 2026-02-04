#pragma once

#include <memory.h>
#include <stdio.h>

#ifdef ENABLE_AVX2
#include <immintrin.h>
#endif

#include <tnn/tnn.h>

#ifdef ENABLE_AVX2
static inline float _avx2_reduce_sum(__m256 v) {
	__m128 lo = _mm256_castps256_ps128(v);
	__m128 hi = _mm256_extractf128_ps(v, 1);
	lo = _mm_add_ps(lo, hi);
	hi = _mm_movehl_ps(hi, lo);
	lo = _mm_add_ps(lo, hi);
	hi = _mm_shuffle_ps(lo, lo, 0x1);
	lo = _mm_add_ss(lo, hi);
	return _mm_cvtss_f32(lo);
}
#endif

// c[M, N] = a[M, K] @ b[K, N]
static void matmul(
    tnn_device_t *dev,
    const void *a,
    const void *b,
    void *c,
    size_t m,
    size_t k,
    size_t n,
    bool tpose_a,
    bool tpose_b,
    bool accum
) {
	TNN_TRACY_ZONE_START();

	// Cast input pointers to float pointers
	const float *a_f = (const float *)a;
	const float *b_f = (const float *)b;
	float *c_f = (float *)c;

	for (size_t i_m = 0; i_m < m; i_m++) {
		for (size_t i_n = 0; i_n < n; i_n++) {
			float sum = 0.0f;
			size_t i_k = 0;

#ifdef ENABLE_AVX2
			__m256 sum_vec = _mm256_setzero_ps();

			// vectorized loop - process 8 floats at a time
			for (; i_k + 7 < k; i_k += 8) {
				__m256 a_vec, b_vec;

				if (tpose_a) {
					// gather is slower but handles non-contiguous access
					a_vec = _mm256_set_ps(
					    a_f[(i_k + 7) * m + i_m],
					    a_f[(i_k + 6) * m + i_m],
					    a_f[(i_k + 5) * m + i_m],
					    a_f[(i_k + 4) * m + i_m],
					    a_f[(i_k + 3) * m + i_m],
					    a_f[(i_k + 2) * m + i_m],
					    a_f[(i_k + 1) * m + i_m],
					    a_f[i_k * m + i_m]
					);
				} else {
					a_vec = _mm256_loadu_ps(&a_f[i_m * k + i_k]);
				}

				if (tpose_b) {
					b_vec = _mm256_loadu_ps(&b_f[i_n * k + i_k]);
				} else {
					b_vec = _mm256_set_ps(
					    b_f[(i_k + 7) * n + i_n],
					    b_f[(i_k + 6) * n + i_n],
					    b_f[(i_k + 5) * n + i_n],
					    b_f[(i_k + 4) * n + i_n],
					    b_f[(i_k + 3) * n + i_n],
					    b_f[(i_k + 2) * n + i_n],
					    b_f[(i_k + 1) * n + i_n],
					    b_f[i_k * n + i_n]
					);
				}

				sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
			}

			sum = _avx2_reduce_sum(sum_vec);
#endif

			// handle remainder
			for (; i_k < k; i_k++) {
				float a_val = tpose_a ? a_f[i_k * m + i_m] : a_f[i_m * k + i_k];
				float b_val = tpose_b ? b_f[i_n * k + i_k] : b_f[i_k * n + i_n];
				sum += a_val * b_val;
			}

			if (accum) {
				c_f[i_m * n + i_n] += sum;
			} else {
				c_f[i_m * n + i_n] = sum;
			}
		}
	}

	TNN_TRACY_ZONE_END();
}

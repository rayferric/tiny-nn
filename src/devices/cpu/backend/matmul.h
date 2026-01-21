#pragma once

#include <memory.h>
#include <stdio.h>

#include <cpuid.h>
#include <immintrin.h>

#include <tnn/tnn.h>

static bool _check_avx2_support() {
	unsigned int eax, ebx, ecx, edx;
	if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
		return false;
	}
	if ((ecx & bit_OSXSAVE) == 0 || (ecx & bit_AVX) == 0) {
		return false;
	}
	if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
		return false;
	}
	return (ebx & bit_AVX2) != 0;
}

__attribute__((target("avx2,fma"))) static inline float
_mm256_reduce_add_ps(__m256 v) {
	__m128 lo = _mm256_castps256_ps128(v);
	__m128 hi = _mm256_extractf128_ps(v, 1);
	lo = _mm_add_ps(lo, hi);
	hi = _mm_movehl_ps(hi, lo);
	lo = _mm_add_ps(lo, hi);
	hi = _mm_shuffle_ps(lo, lo, 0x1);
	lo = _mm_add_ss(lo, hi);
	return _mm_cvtss_f32(lo);
}

// avx-2 version, 1.35x faster in practice, needs optimization
__attribute__((target("avx2,fma"))) static void matmul_avx2(
    const float *a_f,
    const float *b_f,
    float *c_f,
    size_t m,
    size_t k,
    size_t n,
    bool tpose_a,
    bool tpose_b,
    bool accum
) {
	for (size_t i_m = 0; i_m < m; i_m++) {
		for (size_t i_n = 0; i_n < n; i_n++) {
			__m256 sum_vec = _mm256_setzero_ps();
			size_t i_k = 0;

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

			float sum = _mm256_reduce_add_ps(sum_vec);

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
}

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
	// Cast input pointers to float pointers
	const float *a_f = (const float *)a;
	const float *b_f = (const float *)b;
	float *c_f = (float *)c;

	static bool checked_avx2 = false;
	static bool has_avx2 = false;
	if (!checked_avx2) {
		has_avx2 = _check_avx2_support();
		checked_avx2 = true;
	}
	if (has_avx2) {
		matmul_avx2(a_f, b_f, c_f, m, k, n, tpose_a, tpose_b, accum);
		return;
	}

	for (size_t i_m = 0; i_m < m; i_m++) {
		for (size_t i_n = 0; i_n < n; i_n++) {
			float sum = 0.0f;
			for (size_t i_k = 0; i_k < k; i_k++) {
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
}

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <tnn/tnn.h>

static float randf() {
	return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static float compare_buffers(const float *a, const float *b, size_t n) {
	float max_error = 0.0f;
	for (size_t i = 0; i < n; i++) {
		float diff = fabsf(a[i] - b[i]);
		if (diff > max_error) {
			max_error = diff;
		}
	}
	return max_error;
}

void test_matmul(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t m = 64, k = 128, n = 32;
	size_t a_size = m * k;
	size_t b_size = k * n;
	size_t c_size = m * n;

	float *a_host = malloc(a_size * sizeof(float));
	float *b_host = malloc(b_size * sizeof(float));
	float *c_cpu = malloc(c_size * sizeof(float));
	float *c_dev = malloc(c_size * sizeof(float));

	for (size_t i = 0; i < a_size; i++) {
		a_host[i] = randf();
	}
	for (size_t i = 0; i < b_size; i++) {
		b_host[i] = randf();
	}

	cpu->_backend.matmul(
	    cpu, a_host, b_host, c_cpu, m, k, n, false, false, false
	);

	void *a_dev_buf = dev->_backend.buf_alloc(dev, a_size * sizeof(float));
	void *b_dev_buf = dev->_backend.buf_alloc(dev, b_size * sizeof(float));
	void *c_dev_buf = dev->_backend.buf_alloc(dev, c_size * sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, a_dev_buf, a_host, a_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, b_dev_buf, b_host, b_size * sizeof(float)
	);

	dev->_backend.matmul(
	    dev, a_dev_buf, b_dev_buf, c_dev_buf, m, k, n, false, false, false
	);

	dev->_backend.buf_copy_to_host(
	    dev, c_dev, c_dev_buf, c_size * sizeof(float)
	);

	dev->_backend.buf_free(dev, a_dev_buf);
	dev->_backend.buf_free(dev, b_dev_buf);
	dev->_backend.buf_free(dev, c_dev_buf);

	float max_error = compare_buffers(c_cpu, c_dev, c_size);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "matmul err:",
	    max_error,
	    max_error < 1e-4f ? "PASS" : "FAIL"
	);

	free(a_host);
	free(b_host);
	free(c_cpu);
	free(c_dev);
}

void test_add(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t outer = 128, inner = 64;
	size_t a_size = outer * inner;
	size_t b_size = inner;
	size_t out_size = outer * inner;

	float *a_host = malloc(a_size * sizeof(float));
	float *b_host = malloc(b_size * sizeof(float));
	float *out_cpu = malloc(out_size * sizeof(float));
	float *out_dev = malloc(out_size * sizeof(float));

	for (size_t i = 0; i < a_size; i++) {
		a_host[i] = randf();
	}
	for (size_t i = 0; i < b_size; i++) {
		b_host[i] = randf();
	}

	cpu->_backend.add(cpu, a_host, b_host, out_cpu, outer, inner);

	void *a_dev_buf = dev->_backend.buf_alloc(dev, a_size * sizeof(float));
	void *b_dev_buf = dev->_backend.buf_alloc(dev, b_size * sizeof(float));
	void *out_dev_buf = dev->_backend.buf_alloc(dev, out_size * sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, a_dev_buf, a_host, a_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, b_dev_buf, b_host, b_size * sizeof(float)
	);

	dev->_backend.add(dev, a_dev_buf, b_dev_buf, out_dev_buf, outer, inner);

	dev->_backend.buf_copy_to_host(
	    dev, out_dev, out_dev_buf, out_size * sizeof(float)
	);

	dev->_backend.buf_free(dev, a_dev_buf);
	dev->_backend.buf_free(dev, b_dev_buf);
	dev->_backend.buf_free(dev, out_dev_buf);

	float max_error = compare_buffers(out_cpu, out_dev, out_size);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "add err:",
	    max_error,
	    max_error < 1e-4f ? "PASS" : "FAIL"
	);

	free(a_host);
	free(b_host);
	free(out_cpu);
	free(out_dev);
}

void test_accum(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t n = 8192;

	float *in_host = malloc(n * sizeof(float));
	float *out_cpu = malloc(n * sizeof(float));
	float *out_dev = malloc(n * sizeof(float));

	for (size_t i = 0; i < n; i++) {
		in_host[i] = randf();
		out_cpu[i] = randf();
		out_dev[i] = out_cpu[i];
	}

	cpu->_backend.accum(cpu, in_host, out_cpu, n);

	void *in_dev_buf = dev->_backend.buf_alloc(dev, n * sizeof(float));
	void *out_dev_buf = dev->_backend.buf_alloc(dev, n * sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, in_dev_buf, in_host, n * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, out_dev_buf, out_dev, n * sizeof(float)
	);

	dev->_backend.accum(dev, in_dev_buf, out_dev_buf, n);

	dev->_backend.buf_copy_to_host(
	    dev, out_dev, out_dev_buf, n * sizeof(float)
	);

	dev->_backend.buf_free(dev, in_dev_buf);
	dev->_backend.buf_free(dev, out_dev_buf);

	float max_error = compare_buffers(out_cpu, out_dev, n);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "accum err:",
	    max_error,
	    max_error < 1e-4f ? "PASS" : "FAIL"
	);

	free(in_host);
	free(out_cpu);
	free(out_dev);
}

void test_sum_reduce(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t outer = 32, reduced = 256, inner = 16;
	size_t in_size = outer * reduced * inner;
	size_t out_size = outer * inner;
	float scale = 0.5f;

	float *in_host = malloc(in_size * sizeof(float));
	float *out_cpu = malloc(out_size * sizeof(float));
	float *out_dev = malloc(out_size * sizeof(float));

	for (size_t i = 0; i < in_size; i++) {
		in_host[i] = randf();
	}
	for (size_t i = 0; i < out_size; i++) {
		out_cpu[i] = 0.0f;
		out_dev[i] = 0.0f;
	}

	cpu->_backend.sum_reduce(
	    cpu, in_host, out_cpu, outer, reduced, inner, scale, false
	);

	void *in_dev_buf = dev->_backend.buf_alloc(dev, in_size * sizeof(float));
	void *out_dev_buf = dev->_backend.buf_alloc(dev, out_size * sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, in_dev_buf, in_host, in_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, out_dev_buf, out_dev, out_size * sizeof(float)
	);

	dev->_backend.sum_reduce(
	    dev, in_dev_buf, out_dev_buf, outer, reduced, inner, scale, false
	);

	dev->_backend.buf_copy_to_host(
	    dev, out_dev, out_dev_buf, out_size * sizeof(float)
	);

	dev->_backend.buf_free(dev, in_dev_buf);
	dev->_backend.buf_free(dev, out_dev_buf);

	float max_error = compare_buffers(out_cpu, out_dev, out_size);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "sum_reduce err:",
	    max_error,
	    max_error < 1e-3f ? "PASS" : "FAIL"
	);

	free(in_host);
	free(out_cpu);
	free(out_dev);
}

void test_sum_broadcast(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t outer = 32, reduced = 256, inner = 16;
	size_t in_size = outer * inner;
	size_t out_size = outer * reduced * inner;
	float scale = 0.25f;

	float *in_host = malloc(in_size * sizeof(float));
	float *out_cpu = malloc(out_size * sizeof(float));
	float *out_dev = malloc(out_size * sizeof(float));

	for (size_t i = 0; i < in_size; i++) {
		in_host[i] = randf();
	}
	for (size_t i = 0; i < out_size; i++) {
		out_cpu[i] = 0.0f;
		out_dev[i] = 0.0f;
	}

	cpu->_backend.sum_broadcast(
	    cpu, in_host, out_cpu, outer, reduced, inner, scale, false
	);

	void *in_dev_buf = dev->_backend.buf_alloc(dev, in_size * sizeof(float));
	void *out_dev_buf = dev->_backend.buf_alloc(dev, out_size * sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, in_dev_buf, in_host, in_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, out_dev_buf, out_dev, out_size * sizeof(float)
	);

	dev->_backend.sum_broadcast(
	    dev, in_dev_buf, out_dev_buf, outer, reduced, inner, scale, false
	);

	dev->_backend.buf_copy_to_host(
	    dev, out_dev, out_dev_buf, out_size * sizeof(float)
	);

	dev->_backend.buf_free(dev, in_dev_buf);
	dev->_backend.buf_free(dev, out_dev_buf);

	float max_error = compare_buffers(out_cpu, out_dev, out_size);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "sum_broadcast err:",
	    max_error,
	    max_error < 1e-4f ? "PASS" : "FAIL"
	);

	free(in_host);
	free(out_cpu);
	free(out_dev);
}

void test_relu_fw(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t n = 8192;

	float *in_host = malloc(n * sizeof(float));
	float *out_cpu = malloc(n * sizeof(float));
	float *out_dev = malloc(n * sizeof(float));

	for (size_t i = 0; i < n; i++) {
		in_host[i] = randf();
	}

	cpu->_backend.relu_fw(cpu, in_host, out_cpu, n);

	void *in_dev_buf = dev->_backend.buf_alloc(dev, n * sizeof(float));
	void *out_dev_buf = dev->_backend.buf_alloc(dev, n * sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, in_dev_buf, in_host, n * sizeof(float)
	);

	dev->_backend.relu_fw(dev, in_dev_buf, out_dev_buf, n);

	dev->_backend.buf_copy_to_host(
	    dev, out_dev, out_dev_buf, n * sizeof(float)
	);

	dev->_backend.buf_free(dev, in_dev_buf);
	dev->_backend.buf_free(dev, out_dev_buf);

	float max_error = compare_buffers(out_cpu, out_dev, n);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "relu_fw err:",
	    max_error,
	    max_error < 1e-6f ? "PASS" : "FAIL"
	);

	free(in_host);
	free(out_cpu);
	free(out_dev);
}

void test_relu_bw(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t n = 8192;

	float *out_data_host = malloc(n * sizeof(float));
	float *out_grad_host = malloc(n * sizeof(float));
	float *in_grad_cpu = malloc(n * sizeof(float));
	float *in_grad_dev = malloc(n * sizeof(float));

	for (size_t i = 0; i < n; i++) {
		out_data_host[i] = randf();
		out_grad_host[i] = randf();
		in_grad_cpu[i] = 0.0f;
		in_grad_dev[i] = 0.0f;
	}

	cpu->_backend.relu_bw(cpu, out_data_host, out_grad_host, in_grad_cpu, n);

	void *out_data_buf = dev->_backend.buf_alloc(dev, n * sizeof(float));
	void *out_grad_buf = dev->_backend.buf_alloc(dev, n * sizeof(float));
	void *in_grad_buf = dev->_backend.buf_alloc(dev, n * sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, out_data_buf, out_data_host, n * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, out_grad_buf, out_grad_host, n * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, in_grad_buf, in_grad_dev, n * sizeof(float)
	);

	dev->_backend.relu_bw(dev, out_data_buf, out_grad_buf, in_grad_buf, n);

	dev->_backend.buf_copy_to_host(
	    dev, in_grad_dev, in_grad_buf, n * sizeof(float)
	);

	dev->_backend.buf_free(dev, out_data_buf);
	dev->_backend.buf_free(dev, out_grad_buf);
	dev->_backend.buf_free(dev, in_grad_buf);

	float max_error = compare_buffers(in_grad_cpu, in_grad_dev, n);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "relu_bw err:",
	    max_error,
	    max_error < 1e-6f ? "PASS" : "FAIL"
	);

	free(out_data_host);
	free(out_grad_host);
	free(in_grad_cpu);
	free(in_grad_dev);
}

void test_ce_fw(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t n = 128, c = 10;
	size_t pred_size = n * c;

	float *pred_host = malloc(pred_size * sizeof(float));
	float *tgt_host = malloc(pred_size * sizeof(float));
	float out_cpu = 0.0f;
	float out_dev = 0.0f;

	for (size_t i = 0; i < pred_size; i++) {
		pred_host[i] = randf();
		tgt_host[i] = 0.0f;
	}
	for (size_t i = 0; i < n; i++) {
		size_t idx = i * c + (rand() % c);
		tgt_host[idx] = 1.0f;
	}

	cpu->_backend.ce_fw(cpu, pred_host, tgt_host, &out_cpu, n, c);

	void *pred_buf = dev->_backend.buf_alloc(dev, pred_size * sizeof(float));
	void *tgt_buf = dev->_backend.buf_alloc(dev, pred_size * sizeof(float));
	void *out_buf = dev->_backend.buf_alloc(dev, sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, pred_buf, pred_host, pred_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, tgt_buf, tgt_host, pred_size * sizeof(float)
	);

	dev->_backend.ce_fw(dev, pred_buf, tgt_buf, out_buf, n, c);

	dev->_backend.buf_copy_to_host(dev, &out_dev, out_buf, sizeof(float));

	dev->_backend.buf_free(dev, pred_buf);
	dev->_backend.buf_free(dev, tgt_buf);
	dev->_backend.buf_free(dev, out_buf);

	float max_error = fabsf(out_cpu - out_dev);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "ce_fw err:",
	    max_error,
	    max_error < 1e-3f ? "PASS" : "FAIL"
	);

	free(pred_host);
	free(tgt_host);
}

void test_ce_bw(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t n = 128, c = 10;
	size_t pred_size = n * c;

	float *pred_host = malloc(pred_size * sizeof(float));
	float *tgt_host = malloc(pred_size * sizeof(float));
	float out_grad_host = 1.0f;
	float *pred_grad_cpu = malloc(pred_size * sizeof(float));
	float *pred_grad_dev = malloc(pred_size * sizeof(float));

	for (size_t i = 0; i < pred_size; i++) {
		pred_host[i] = randf();
		tgt_host[i] = 0.0f;
		pred_grad_cpu[i] = 0.0f;
		pred_grad_dev[i] = 0.0f;
	}
	for (size_t i = 0; i < n; i++) {
		size_t idx = i * c + (rand() % c);
		tgt_host[idx] = 1.0f;
	}

	cpu->_backend.ce_bw(
	    cpu, pred_host, tgt_host, &out_grad_host, pred_grad_cpu, n, c
	);

	void *pred_buf = dev->_backend.buf_alloc(dev, pred_size * sizeof(float));
	void *tgt_buf = dev->_backend.buf_alloc(dev, pred_size * sizeof(float));
	void *out_grad_buf = dev->_backend.buf_alloc(dev, sizeof(float));
	void *pred_grad_buf =
	    dev->_backend.buf_alloc(dev, pred_size * sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, pred_buf, pred_host, pred_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, tgt_buf, tgt_host, pred_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, out_grad_buf, &out_grad_host, sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, pred_grad_buf, pred_grad_dev, pred_size * sizeof(float)
	);

	dev->_backend.ce_bw(
	    dev, pred_buf, tgt_buf, out_grad_buf, pred_grad_buf, n, c
	);

	dev->_backend.buf_copy_to_host(
	    dev, pred_grad_dev, pred_grad_buf, pred_size * sizeof(float)
	);

	dev->_backend.buf_free(dev, pred_buf);
	dev->_backend.buf_free(dev, tgt_buf);
	dev->_backend.buf_free(dev, out_grad_buf);
	dev->_backend.buf_free(dev, pred_grad_buf);

	float max_error = compare_buffers(pred_grad_cpu, pred_grad_dev, pred_size);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "ce_bw err:",
	    max_error,
	    max_error < 1e-4f ? "PASS" : "FAIL"
	);

	free(pred_host);
	free(tgt_host);
	free(pred_grad_cpu);
	free(pred_grad_dev);
}

void test_conv_fw(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t batch = 4, h_in = 28, w_in = 28, c_in = 3;
	size_t c_out = 8, kernel_size = 3, stride = 1, padding = 1;
	size_t h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
	size_t w_out = (w_in + 2 * padding - kernel_size) / stride + 1;

	size_t input_size = batch * h_in * w_in * c_in;
	size_t weight_size = c_out * kernel_size * kernel_size * c_in;
	size_t output_size = batch * h_out * w_out * c_out;

	float *input_host = malloc(input_size * sizeof(float));
	float *weight_host = malloc(weight_size * sizeof(float));
	float *output_cpu = malloc(output_size * sizeof(float));
	float *output_dev = malloc(output_size * sizeof(float));

	for (size_t i = 0; i < input_size; i++) {
		input_host[i] = randf();
	}
	for (size_t i = 0; i < weight_size; i++) {
		weight_host[i] = randf();
	}

	cpu->_backend.conv_fw(
	    cpu,
	    input_host,
	    weight_host,
	    output_cpu,
	    batch,
	    h_in,
	    w_in,
	    c_in,
	    h_out,
	    w_out,
	    c_out,
	    kernel_size,
	    stride,
	    padding
	);

	void *input_buf = dev->_backend.buf_alloc(dev, input_size * sizeof(float));
	void *weight_buf =
	    dev->_backend.buf_alloc(dev, weight_size * sizeof(float));
	void *output_buf =
	    dev->_backend.buf_alloc(dev, output_size * sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, input_buf, input_host, input_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, weight_buf, weight_host, weight_size * sizeof(float)
	);

	dev->_backend.conv_fw(
	    dev,
	    input_buf,
	    weight_buf,
	    output_buf,
	    batch,
	    h_in,
	    w_in,
	    c_in,
	    h_out,
	    w_out,
	    c_out,
	    kernel_size,
	    stride,
	    padding
	);

	dev->_backend.buf_copy_to_host(
	    dev, output_dev, output_buf, output_size * sizeof(float)
	);

	dev->_backend.buf_free(dev, input_buf);
	dev->_backend.buf_free(dev, weight_buf);
	dev->_backend.buf_free(dev, output_buf);

	float max_error = compare_buffers(output_cpu, output_dev, output_size);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "conv_fw err:",
	    max_error,
	    max_error < 1e-3f ? "PASS" : "FAIL"
	);

	free(input_host);
	free(weight_host);
	free(output_cpu);
	free(output_dev);
}

void test_conv_bw_input(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t batch = 4, h_in = 28, w_in = 28, c_in = 3;
	size_t c_out = 8, kernel_size = 3, stride = 1, padding = 1;
	size_t h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
	size_t w_out = (w_in + 2 * padding - kernel_size) / stride + 1;

	size_t out_grad_size = batch * h_out * w_out * c_out;
	size_t weight_size = c_out * kernel_size * kernel_size * c_in;
	size_t in_grad_size = batch * h_in * w_in * c_in;

	float *out_grad_host = malloc(out_grad_size * sizeof(float));
	float *weight_host = malloc(weight_size * sizeof(float));
	float *in_grad_cpu = malloc(in_grad_size * sizeof(float));
	float *in_grad_dev = malloc(in_grad_size * sizeof(float));

	for (size_t i = 0; i < out_grad_size; i++) {
		out_grad_host[i] = randf();
	}
	for (size_t i = 0; i < weight_size; i++) {
		weight_host[i] = randf();
	}
	for (size_t i = 0; i < in_grad_size; i++) {
		in_grad_cpu[i] = 0.0f;
		in_grad_dev[i] = 0.0f;
	}

	cpu->_backend.conv_bw_input(
	    cpu,
	    out_grad_host,
	    weight_host,
	    in_grad_cpu,
	    batch,
	    h_in,
	    w_in,
	    c_in,
	    h_out,
	    w_out,
	    c_out,
	    kernel_size,
	    stride,
	    padding
	);

	void *out_grad_buf =
	    dev->_backend.buf_alloc(dev, out_grad_size * sizeof(float));
	void *weight_buf =
	    dev->_backend.buf_alloc(dev, weight_size * sizeof(float));
	void *in_grad_buf =
	    dev->_backend.buf_alloc(dev, in_grad_size * sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, out_grad_buf, out_grad_host, out_grad_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, weight_buf, weight_host, weight_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, in_grad_buf, in_grad_dev, in_grad_size * sizeof(float)
	);

	dev->_backend.conv_bw_input(
	    dev,
	    out_grad_buf,
	    weight_buf,
	    in_grad_buf,
	    batch,
	    h_in,
	    w_in,
	    c_in,
	    h_out,
	    w_out,
	    c_out,
	    kernel_size,
	    stride,
	    padding
	);

	dev->_backend.buf_copy_to_host(
	    dev, in_grad_dev, in_grad_buf, in_grad_size * sizeof(float)
	);

	dev->_backend.buf_free(dev, out_grad_buf);
	dev->_backend.buf_free(dev, weight_buf);
	dev->_backend.buf_free(dev, in_grad_buf);

	float max_error = compare_buffers(in_grad_cpu, in_grad_dev, in_grad_size);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "conv_bw_input err:",
	    max_error,
	    max_error < 1e-3f ? "PASS" : "FAIL"
	);

	free(out_grad_host);
	free(weight_host);
	free(in_grad_cpu);
	free(in_grad_dev);
}

void test_conv_bw_weight(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t batch = 4, h_in = 28, w_in = 28, c_in = 3;
	size_t c_out = 8, kernel_size = 3, stride = 1, padding = 1;
	size_t h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
	size_t w_out = (w_in + 2 * padding - kernel_size) / stride + 1;

	size_t input_size = batch * h_in * w_in * c_in;
	size_t out_grad_size = batch * h_out * w_out * c_out;
	size_t weight_grad_size = c_out * kernel_size * kernel_size * c_in;

	float *input_host = malloc(input_size * sizeof(float));
	float *out_grad_host = malloc(out_grad_size * sizeof(float));
	float *weight_grad_cpu = malloc(weight_grad_size * sizeof(float));
	float *weight_grad_dev = malloc(weight_grad_size * sizeof(float));

	for (size_t i = 0; i < input_size; i++) {
		input_host[i] = randf();
	}
	for (size_t i = 0; i < out_grad_size; i++) {
		out_grad_host[i] = randf();
	}
	for (size_t i = 0; i < weight_grad_size; i++) {
		weight_grad_cpu[i] = 0.0f;
		weight_grad_dev[i] = 0.0f;
	}

	cpu->_backend.conv_bw_weight(
	    cpu,
	    input_host,
	    out_grad_host,
	    weight_grad_cpu,
	    batch,
	    h_in,
	    w_in,
	    c_in,
	    h_out,
	    w_out,
	    c_out,
	    kernel_size,
	    stride,
	    padding
	);

	void *input_buf = dev->_backend.buf_alloc(dev, input_size * sizeof(float));
	void *out_grad_buf =
	    dev->_backend.buf_alloc(dev, out_grad_size * sizeof(float));
	void *weight_grad_buf =
	    dev->_backend.buf_alloc(dev, weight_grad_size * sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, input_buf, input_host, input_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, out_grad_buf, out_grad_host, out_grad_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, weight_grad_buf, weight_grad_dev, weight_grad_size * sizeof(float)
	);

	dev->_backend.conv_bw_weight(
	    dev,
	    input_buf,
	    out_grad_buf,
	    weight_grad_buf,
	    batch,
	    h_in,
	    w_in,
	    c_in,
	    h_out,
	    w_out,
	    c_out,
	    kernel_size,
	    stride,
	    padding
	);

	dev->_backend.buf_copy_to_host(
	    dev, weight_grad_dev, weight_grad_buf, weight_grad_size * sizeof(float)
	);

	dev->_backend.buf_free(dev, input_buf);
	dev->_backend.buf_free(dev, out_grad_buf);
	dev->_backend.buf_free(dev, weight_grad_buf);

	float max_error =
	    compare_buffers(weight_grad_cpu, weight_grad_dev, weight_grad_size);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "conv_bw_weight err:",
	    max_error,
	    max_error < 1e-3f ? "PASS" : "FAIL"
	);

	free(input_host);
	free(out_grad_host);
	free(weight_grad_cpu);
	free(weight_grad_dev);
}

void test_bn_fw(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t nhw = 512, c = 16;
	float momentum = 0.1f;
	size_t input_size = nhw * c;
	size_t stats_size = c;

	float *input_host = malloc(input_size * sizeof(float));
	float *output_cpu = malloc(input_size * sizeof(float));
	float *output_dev = malloc(input_size * sizeof(float));
	float *running_mean_cpu = calloc(stats_size, sizeof(float));
	float *running_mean_dev = calloc(stats_size, sizeof(float));
	float *running_var_cpu = calloc(stats_size, sizeof(float));
	float *running_var_dev = calloc(stats_size, sizeof(float));
	float *batch_var_cpu = malloc(stats_size * sizeof(float));
	float *batch_var_dev = malloc(stats_size * sizeof(float));

	for (size_t i = 0; i < input_size; i++) {
		input_host[i] = randf();
	}
	for (size_t i = 0; i < stats_size; i++) {
		running_var_cpu[i] = 1.0f;
		running_var_dev[i] = 1.0f;
	}

	cpu->_backend.bn_fw(
	    cpu,
	    input_host,
	    output_cpu,
	    running_mean_cpu,
	    running_var_cpu,
	    batch_var_cpu,
	    nhw,
	    c,
	    momentum,
	    false
	);

	void *input_buf = dev->_backend.buf_alloc(dev, input_size * sizeof(float));
	void *output_buf = dev->_backend.buf_alloc(dev, input_size * sizeof(float));
	void *running_mean_buf =
	    dev->_backend.buf_alloc(dev, stats_size * sizeof(float));
	void *running_var_buf =
	    dev->_backend.buf_alloc(dev, stats_size * sizeof(float));
	void *batch_var_buf =
	    dev->_backend.buf_alloc(dev, stats_size * sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, input_buf, input_host, input_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, running_mean_buf, running_mean_dev, stats_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, running_var_buf, running_var_dev, stats_size * sizeof(float)
	);

	dev->_backend.bn_fw(
	    dev,
	    input_buf,
	    output_buf,
	    running_mean_buf,
	    running_var_buf,
	    batch_var_buf,
	    nhw,
	    c,
	    momentum,
	    false
	);

	dev->_backend.buf_copy_to_host(
	    dev, output_dev, output_buf, input_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_host(
	    dev, running_mean_dev, running_mean_buf, stats_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_host(
	    dev, running_var_dev, running_var_buf, stats_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_host(
	    dev, batch_var_dev, batch_var_buf, stats_size * sizeof(float)
	);

	dev->_backend.buf_free(dev, input_buf);
	dev->_backend.buf_free(dev, output_buf);
	dev->_backend.buf_free(dev, running_mean_buf);
	dev->_backend.buf_free(dev, running_var_buf);
	dev->_backend.buf_free(dev, batch_var_buf);

	float max_error = compare_buffers(output_cpu, output_dev, input_size);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "bn_fw err:",
	    max_error,
	    max_error < 1e-3f ? "PASS" : "FAIL"
	);

	free(input_host);
	free(output_cpu);
	free(output_dev);
	free(running_mean_cpu);
	free(running_mean_dev);
	free(running_var_cpu);
	free(running_var_dev);
	free(batch_var_cpu);
	free(batch_var_dev);
}

void test_bn_bw(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t nhw = 512, c = 16;
	size_t data_size = nhw * c;
	size_t stats_size = c;

	float *out_grad_host = malloc(data_size * sizeof(float));
	float *out_data_host = malloc(data_size * sizeof(float));
	float *in_grad_cpu = malloc(data_size * sizeof(float));
	float *in_grad_dev = malloc(data_size * sizeof(float));
	float *running_var_host = malloc(stats_size * sizeof(float));
	float *batch_var_host = malloc(stats_size * sizeof(float));

	for (size_t i = 0; i < data_size; i++) {
		out_grad_host[i] = randf();
		out_data_host[i] = randf();
		in_grad_cpu[i] = 0.0f;
		in_grad_dev[i] = 0.0f;
	}
	for (size_t i = 0; i < stats_size; i++) {
		running_var_host[i] = 1.0f + randf() * 0.1f;
		batch_var_host[i] = 1.0f + randf() * 0.1f;
	}

	cpu->_backend.bn_bw(
	    cpu,
	    out_grad_host,
	    out_data_host,
	    in_grad_cpu,
	    running_var_host,
	    batch_var_host,
	    nhw,
	    c,
	    false
	);

	void *out_grad_buf =
	    dev->_backend.buf_alloc(dev, data_size * sizeof(float));
	void *out_data_buf =
	    dev->_backend.buf_alloc(dev, data_size * sizeof(float));
	void *in_grad_buf = dev->_backend.buf_alloc(dev, data_size * sizeof(float));
	void *running_var_buf =
	    dev->_backend.buf_alloc(dev, stats_size * sizeof(float));
	void *batch_var_buf =
	    dev->_backend.buf_alloc(dev, stats_size * sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, out_grad_buf, out_grad_host, data_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, out_data_buf, out_data_host, data_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, in_grad_buf, in_grad_dev, data_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, running_var_buf, running_var_host, stats_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, batch_var_buf, batch_var_host, stats_size * sizeof(float)
	);

	dev->_backend.bn_bw(
	    dev,
	    out_grad_buf,
	    out_data_buf,
	    in_grad_buf,
	    running_var_buf,
	    batch_var_buf,
	    nhw,
	    c,
	    false
	);

	dev->_backend.buf_copy_to_host(
	    dev, in_grad_dev, in_grad_buf, data_size * sizeof(float)
	);

	dev->_backend.buf_free(dev, out_grad_buf);
	dev->_backend.buf_free(dev, out_data_buf);
	dev->_backend.buf_free(dev, in_grad_buf);
	dev->_backend.buf_free(dev, running_var_buf);
	dev->_backend.buf_free(dev, batch_var_buf);

	float max_error = compare_buffers(in_grad_cpu, in_grad_dev, data_size);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "bn_bw err:",
	    max_error,
	    max_error < 1e-3f ? "PASS" : "FAIL"
	);

	free(out_grad_host);
	free(out_data_host);
	free(in_grad_cpu);
	free(in_grad_dev);
	free(running_var_host);
	free(batch_var_host);
}

void test_adamw(const char *device_name) {
	tnn_device_t *cpu = tnn_get_cpu();
	tnn_device_t *dev = tnn_find_device(device_name);

	if (!dev) {
		fprintf(stderr, "device %s not found\n", device_name);
		return;
	}

	size_t param_size = 4096;
	float t = 10.0f;
	float lr = 0.001f;
	float b1 = 0.9f;
	float b2 = 0.999f;
	float eps = 1e-8f;
	float wd = 0.01f;

	float *param_data_cpu = malloc(param_size * sizeof(float));
	float *param_data_dev = malloc(param_size * sizeof(float));
	float *param_grad_cpu = malloc(param_size * sizeof(float));
	float *param_grad_dev = malloc(param_size * sizeof(float));
	float *m1_data_cpu = malloc(param_size * sizeof(float));
	float *m1_data_dev = malloc(param_size * sizeof(float));
	float *m2_data_cpu = malloc(param_size * sizeof(float));
	float *m2_data_dev = malloc(param_size * sizeof(float));

	for (size_t i = 0; i < param_size; i++) {
		param_data_cpu[i] = randf();
		param_data_dev[i] = param_data_cpu[i];
		param_grad_cpu[i] = randf();
		param_grad_dev[i] = param_grad_cpu[i];
		m1_data_cpu[i] = randf() * 0.1f;
		m1_data_dev[i] = m1_data_cpu[i];
		m2_data_cpu[i] = randf() * 0.1f;
		m2_data_dev[i] = m2_data_cpu[i];
	}

	cpu->_backend.adamw(
	    cpu,
	    param_data_cpu,
	    param_grad_cpu,
	    m1_data_cpu,
	    m2_data_cpu,
	    param_size,
	    t,
	    lr,
	    b1,
	    b2,
	    eps,
	    wd
	);

	void *param_data_buf =
	    dev->_backend.buf_alloc(dev, param_size * sizeof(float));
	void *param_grad_buf =
	    dev->_backend.buf_alloc(dev, param_size * sizeof(float));
	void *m1_data_buf =
	    dev->_backend.buf_alloc(dev, param_size * sizeof(float));
	void *m2_data_buf =
	    dev->_backend.buf_alloc(dev, param_size * sizeof(float));

	dev->_backend.buf_copy_to_device(
	    dev, param_data_buf, param_data_dev, param_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, param_grad_buf, param_grad_dev, param_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, m1_data_buf, m1_data_dev, param_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_device(
	    dev, m2_data_buf, m2_data_dev, param_size * sizeof(float)
	);

	dev->_backend.adamw(
	    dev,
	    param_data_buf,
	    param_grad_buf,
	    m1_data_buf,
	    m2_data_buf,
	    param_size,
	    t,
	    lr,
	    b1,
	    b2,
	    eps,
	    wd
	);

	dev->_backend.buf_copy_to_host(
	    dev, param_data_dev, param_data_buf, param_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_host(
	    dev, m1_data_dev, m1_data_buf, param_size * sizeof(float)
	);
	dev->_backend.buf_copy_to_host(
	    dev, m2_data_dev, m2_data_buf, param_size * sizeof(float)
	);

	dev->_backend.buf_free(dev, param_data_buf);
	dev->_backend.buf_free(dev, param_grad_buf);
	dev->_backend.buf_free(dev, m1_data_buf);
	dev->_backend.buf_free(dev, m2_data_buf);

	float max_error =
	    compare_buffers(param_data_cpu, param_data_dev, param_size);

	printf(
	    "%-22s%8.8f [%s]\n",
	    "adamw err:",
	    max_error,
	    max_error < 1e-4f ? "PASS" : "FAIL"
	);

	free(param_data_cpu);
	free(param_data_dev);
	free(param_grad_cpu);
	free(param_grad_dev);
	free(m1_data_cpu);
	free(m1_data_dev);
	free(m2_data_cpu);
	free(m2_data_dev);
}

int main() {
	srand(time(NULL));

	if (tnn_init()) {
		fprintf(stderr, "failed to initialize tnn\n");
		return 1;
	}

	const char *device = "vulkan";

	test_matmul(device);
	test_add(device);
	test_accum(device);
	test_sum_reduce(device);
	test_sum_broadcast(device);
	test_relu_fw(device);
	test_relu_bw(device);
	test_ce_fw(device);
	test_ce_bw(device);
	test_conv_fw(device);
	test_conv_bw_input(device);
	test_conv_bw_weight(device);
	test_bn_fw(device);
	test_bn_bw(device);
	test_adamw(device);

	tnn_terminate();
	return 0;
}

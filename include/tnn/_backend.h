#pragma once

#include <stdbool.h>
#include <stddef.h>

typedef struct tnn_tensor tnn_tensor_t;
typedef struct tnn_device tnn_device_t;

typedef struct {
	// memory

	void *(*buf_alloc)(tnn_device_t *dev, size_t sz);
	void (*buf_free)(tnn_device_t *dev, void *ptr);
	void (*buf_copy)(tnn_device_t *dev, void *dst, const void *src, size_t sz);

	// memory transfer (NULL for devices that use system memory)

	void (*buf_copy_to_host)(
	    tnn_device_t *dev, void *dst, const void *src, size_t sz
	);
	void (*buf_copy_to_device)(
	    tnn_device_t *dev, void *dst, const void *src, size_t sz
	);

	// blas/nn

	void (*matmul)(
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
	);

	// a should have outer * inner elements
	// b is broadcasted from inner to outer * inner
	void (*add)(
	    tnn_device_t *dev,
	    const void *a,
	    const void *b,
	    void *out,
	    size_t outer,
	    size_t inner
	);

	void (*accum)(tnn_device_t *dev, const void *in, void *out, size_t n);

	// reduces across the middle dimension
	// in must have outer * reduced * inner elements
	// out must have outer * inner elements
	// accum=true adds to existing out values
	// reverse=true broadcasts the middle dimension instead of reducing
	void (*sum)(
	    tnn_device_t *dev,
	    const void *in,
	    void *out,
	    size_t outer,
	    size_t reduced,
	    size_t inner,
	    float scale,
	    bool accum,
	    bool reverse
	);

	void (*relu_fw)(tnn_device_t *dev, const void *in, void *out, size_t n);

	void (*relu_bw)(
	    tnn_device_t *dev,
	    const void *out_data,
	    const void *out_grad,
	    void *in_grad,
	    size_t n
	);

	void (*ce_fw)(
	    tnn_device_t *dev,
	    const void *pred, // [N, C]
	    const void *tgt,  // [N, C]
	    void *out,        // [1]
	    size_t n,
	    size_t c
	);

	void (*ce_bw)(
	    tnn_device_t *dev,
	    const void *pred,
	    const void *tgt,
	    const void *out_grad,
	    void *pred_grad,
	    size_t n,
	    size_t c
	);

	// convolution operations for NHWC layout
	// forward: input[batch,h_in,w_in,c_in] * weight[c_out,k,k,c_in] ->
	// output[batch,h_out,w_out,c_out]
	void (*conv_fw)(
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
	);

	// backward for input: accumulates gradients into in_grad
	void (*conv_bw_input)(
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
	);

	// backward for weight: accumulates gradients into weight_grad
	void (*conv_bw_weight)(
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
	);

	// input: [NHW, C]
	// output: [NHW, C]
	// running_mean/running_var: [C] - updated with EMA if !test
	// batch_mean/batch_var: [C] - computed batch stats (output, for backward
	// pass)
	void (*bn_fw)(
	    tnn_device_t *dev,
	    const void *input,
	    void *output,
	    void *running_mean,
	    void *running_var,
	    void *tmp_batch_mean,
	    void *batch_var, // only written if !test
	    size_t nhw,
	    size_t c,
	    float momentum,
	    bool test
	);

	// out_data: normalized output [NHW, C]
	// out_grad: [NHW, C]
	// in_grad: [NHW, C] (accumulated)
	// batch_mean/batch_var: [C] (only used if !test)
	// running_var: [C] (only used if test)
	void (*bn_bw)(
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
	);

	void (*adamw)(
	    tnn_device_t *dev,
	    void *param_data,
	    void *param_grad,
	    void *m1_data,
	    void *m2_data,
	    size_t param_size,
	    float t,
	    float lr,
	    float b1,
	    float b2,
	    float eps,
	    float wd
	);
} _tnn_backend_t;

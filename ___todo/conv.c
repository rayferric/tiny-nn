#include <tnn/tnn.h>

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

// Context structure for conv operation
typedef struct {
	size_t stride;
	size_t padding;
	size_t kernel_size;
} conv_context_t;

static void conv_free_context(void *ctx) {
	free(ctx);
}

// Helper to perform 2D convolution (HWC format)
// input: [batch, h_in, w_in, c_in]
// kernel: [kh, kw, c_in, c_out]
// output: [batch, h_out, w_out, c_out]
static void conv2d_forward(
    const float *__restrict input,
    const float *__restrict kernel,
    float *__restrict output,
    size_t batch,
    size_t h_in,
    size_t w_in,
    size_t c_in,
    size_t kh,
    size_t kw,
    size_t stride,
    size_t padding,
    size_t h_out,
    size_t w_out,
    size_t c_out
) {
	memset(output, 0, batch * h_out * w_out * c_out * sizeof(float));

	for (size_t b = 0; b < batch; b++) {
		for (size_t oh = 0; oh < h_out; oh++) {
			for (size_t ow = 0; ow < w_out; ow++) {
				for (size_t oc = 0; oc < c_out; oc++) {
					float sum = 0.0f;

					// Convolve kernel over input
					for (size_t kh_i = 0; kh_i < kh; kh_i++) {
						for (size_t kw_i = 0; kw_i < kw; kw_i++) {
							for (size_t ic = 0; ic < c_in; ic++) {
								// Input position with padding
								int ih =
								    (int)(oh * stride + kh_i) - (int)padding;
								int iw =
								    (int)(ow * stride + kw_i) - (int)padding;

								// Check bounds
								if (ih >= 0 && ih < (int)h_in && iw >= 0 &&
								    iw < (int)w_in) {
									size_t input_idx =
									    b * (h_in * w_in * c_in) +
									    ih * (w_in * c_in) + iw * c_in + ic;
									size_t kernel_idx =
									    kh_i * (kw * c_in * c_out) +
									    kw_i * (c_in * c_out) + ic * c_out + oc;
									sum +=
									    input[input_idx] * kernel[kernel_idx];
								}
							}
						}
					}

					size_t output_idx = b * (h_out * w_out * c_out) +
					                    oh * (w_out * c_out) + ow * c_out + oc;
					output[output_idx] = sum;
				}
			}
		}
	}
}

// Backward pass: compute gradients w.r.t. input and kernel
static void conv2d_backward(
    const float *__restrict grad_output,
    const float *__restrict input,
    const float *__restrict kernel,
    float *__restrict grad_input,
    float *__restrict grad_kernel,
    size_t batch,
    size_t h_in,
    size_t w_in,
    size_t c_in,
    size_t kh,
    size_t kw,
    size_t stride,
    size_t padding,
    size_t h_out,
    size_t w_out,
    size_t c_out
) {
	// Compute grad_input if needed
	if (grad_input != NULL) {
		for (size_t b = 0; b < batch; b++) {
			for (size_t ih = 0; ih < h_in; ih++) {
				for (size_t iw = 0; iw < w_in; iw++) {
					for (size_t ic = 0; ic < c_in; ic++) {
						float sum = 0.0f;

						// For each output position that uses this input pixel
						for (size_t kh_i = 0; kh_i < kh; kh_i++) {
							for (size_t kw_i = 0; kw_i < kw; kw_i++) {
								for (size_t oc = 0; oc < c_out; oc++) {
									// Which output positions use this input
									// pixel?
									int oh =
									    ((int)ih + (int)padding - (int)kh_i) /
									    (int)stride;
									int ow =
									    ((int)iw + (int)padding - (int)kw_i) /
									    (int)stride;

									// Check if this is a valid output position
									if (oh >= 0 && oh < (int)h_out && ow >= 0 &&
									    ow < (int)w_out) {
										// Check if this input pixel is actually
										// used
										int check_ih = oh * (int)stride +
										               (int)kh_i - (int)padding;
										int check_iw = ow * (int)stride +
										               (int)kw_i - (int)padding;

										if (check_ih == (int)ih &&
										    check_iw == (int)iw) {
											size_t grad_output_idx =
											    b * (h_out * w_out * c_out) +
											    oh * (w_out * c_out) +
											    ow * c_out + oc;
											size_t kernel_idx =
											    kh_i * (kw * c_in * c_out) +
											    kw_i * (c_in * c_out) +
											    ic * c_out + oc;
											sum +=
											    grad_output[grad_output_idx] *
											    kernel[kernel_idx];
										}
									}
								}
							}
						}

						size_t input_idx = b * (h_in * w_in * c_in) +
						                   ih * (w_in * c_in) + iw * c_in + ic;
						grad_input[input_idx] += sum;
					}
				}
			}
		}
	}

	// Compute grad_kernel
	for (size_t kh_i = 0; kh_i < kh; kh_i++) {
		for (size_t kw_i = 0; kw_i < kw; kw_i++) {
			for (size_t ic = 0; ic < c_in; ic++) {
				for (size_t oc = 0; oc < c_out; oc++) {
					float sum = 0.0f;

					for (size_t b = 0; b < batch; b++) {
						for (size_t oh = 0; oh < h_out; oh++) {
							for (size_t ow = 0; ow < w_out; ow++) {
								int ih =
								    (int)(oh * stride + kh_i) - (int)padding;
								int iw =
								    (int)(ow * stride + kw_i) - (int)padding;

								if (ih >= 0 && ih < (int)h_in && iw >= 0 &&
								    iw < (int)w_in) {
									size_t input_idx =
									    b * (h_in * w_in * c_in) +
									    ih * (w_in * c_in) + iw * c_in + ic;
									size_t grad_output_idx =
									    b * (h_out * w_out * c_out) +
									    oh * (w_out * c_out) + ow * c_out + oc;
									sum += input[input_idx] *
									       grad_output[grad_output_idx];
								}
							}
						}
					}

					size_t kernel_idx = kh_i * (kw * c_in * c_out) +
					                    kw_i * (c_in * c_out) + ic * c_out + oc;
					grad_kernel[kernel_idx] += sum;
				}
			}
		}
	}
}

static void conv_backward(tnn_tensor_t *self) {
	tnn_tensor_t *input = self->parents[0];
	tnn_tensor_t *kernel = self->parents[1];

	// Extract dimensions from tensors
	size_t batch = input->dims[0];
	size_t h_in = input->dims[1];
	size_t w_in = input->dims[2];
	size_t c_in = input->dims[3];

	size_t kh = kernel->dims[0];
	size_t kw = kernel->dims[1];
	size_t c_out = kernel->dims[3];

	size_t h_out = self->dims[1];
	size_t w_out = self->dims[2];

	// Retrieve stride and padding from context
	assert(self->context != NULL && "conv_backward: context is NULL");
	conv_context_t *ctx = (conv_context_t *)self->context;
	size_t stride = ctx->stride;
	size_t padding = ctx->padding;

	conv2d_backward(
	    self->grad,
	    input->data,
	    kernel->data,
	    input->type == TNN_OUTPUT ? input->grad : NULL,
	    kernel->grad,
	    batch,
	    h_in,
	    w_in,
	    c_in,
	    kh,
	    kw,
	    stride,
	    padding,
	    h_out,
	    w_out,
	    c_out
	);

	// Clean up context after backward
	conv_free_context(self->context);
	self->context = NULL;
	self->free_context = NULL;
}

tnn_tensor_t *tnn_conv(
    tnn_tensor_t *input,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    size_t out_channels
) {
	assert(input != NULL);
	assert(input->num_dims == 4); // [batch, height, width, channels]

	size_t batch = input->dims[0];
	size_t h_in = input->dims[1];
	size_t w_in = input->dims[2];
	size_t c_in = input->dims[3];

	// Calculate output dimensions
	size_t h_out = (h_in + 2 * padding - kernel_size) / stride + 1;
	size_t w_out = (w_in + 2 * padding - kernel_size) / stride + 1;

	// Get or create kernel parameter [kh, kw, c_in, c_out]
	size_t kernel_dims[4] = {kernel_size, kernel_size, c_in, out_channels};
	bool kernel_created = false;
	tnn_tensor_t *kernel = _tnn_get_or_create_param(
	    "conv_kernel", kernel_dims, 4, TNN_PARAM, &kernel_created
	);

	if (kernel_created) {
		// He initialization for ReLU
		float std = sqrtf(2.0f / (kernel_size * kernel_size * c_in));
		size_t kernel_size_total =
		    kernel_size * kernel_size * c_in * out_channels;
		for (size_t i = 0; i < kernel_size_total; i++) {
			// Box-Muller transform for normal distribution
			float u1 = (float)rand() / (float)RAND_MAX;
			float u2 = (float)rand() / (float)RAND_MAX;
			float z = sqrtf(-2.0f * logf(u1 + 1e-8f)) * cosf(2.0f * M_PI * u2);
			kernel->data[i] = z * std;
		}
	}

	// Allocate output tensor [batch, h_out, w_out, c_out]
	size_t output_dims[4] = {batch, h_out, w_out, out_channels};
	tnn_tensor_t *output = _tnn_alloc_tensor(output_dims, 4, TNN_OUTPUT);

	// Forward pass
	conv2d_forward(
	    input->data,
	    kernel->data,
	    output->data,
	    batch,
	    h_in,
	    w_in,
	    c_in,
	    kernel_size,
	    kernel_size,
	    stride,
	    padding,
	    h_out,
	    w_out,
	    out_channels
	);

	// Store context for backward pass
	conv_context_t *ctx = malloc(sizeof(conv_context_t));
	ctx->stride = stride;
	ctx->padding = padding;
	ctx->kernel_size = kernel_size;
	output->context = ctx;
	output->free_context = conv_free_context;

	output->parents[0] = input;
	output->parents[1] = kernel;
	output->num_parents = 2;
	output->backward = conv_backward;

	return output;
}

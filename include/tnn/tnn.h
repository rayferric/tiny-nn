#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "./_optarg.h"

#pragma clang diagnostic ignored "-Winitializer-overrides"

///
// TENSOR STRUCTURE AND UTILS
// impl: src/tensor.c
///

typedef struct tnn_tensor {
	float *data;
	float *grad;

	size_t *dims;
	size_t num_dims;

	bool requires_grad; // will get a gradient during backprop
	bool is_state;      // should not be freed by tnn_free()

	// computation graph
	struct tnn_tensor *parents[3];
	size_t num_parents;
	size_t num_children; // ref-count

	void (*backward)(struct tnn_tensor *);
	void *context; // pass more info from forward to backward
	void (*free_context)(void *);
} tnn_tensor_t;

tnn_tensor_t *tnn_alloc(const size_t *dims, size_t num_dims);
// gets a state tensor or allocates empty and saves to state dict
tnn_tensor_t *tnn_alloc_or_get_state(
    const size_t *dims, size_t num_dims, const char *key, bool *allocated
);

// frees the tensor and their parents recursively.
void tnn_free(tnn_tensor_t *t);

// creates a copy of the tensor, belonging to a new computation graph. stops
// backprop and recursive free
tnn_tensor_t *tnn_detach(tnn_tensor_t *t);

// in-place initializers
void tnn_init_from_memory(tnn_tensor_t *t, const float *data);
void tnn_init_zeros(tnn_tensor_t *t);
void tnn_init_xavier(tnn_tensor_t *t);
void tnn_init_randn(tnn_tensor_t *t);

// utils
size_t tnn_dim(tnn_tensor_t *t, int32_t i_dim);
size_t tnn_size(tnn_tensor_t *t);
void tnn_print(tnn_tensor_t *t);
float tnn_item(tnn_tensor_t *t);

///
// STATE DICT
// impl: src/state.c
///

int tnn_init();
void tnn_terminate();

void tnn_push(const char *key_fmt, ...);
void tnn_pop();
#define TNN_SCOPE(key_fmt, ...)                                                \
	for (int _tnn_once = (tnn_push(key_fmt, ##__VA_ARGS__), 1); _tnn_once;     \
	     tnn_pop(), _tnn_once = 0)

void tnn_save(const char *filename);
void tnn_load(const char *filename);

size_t tnn_list_state_keys(char **out_keys);
tnn_tensor_t *tnn_get_state(const char *key);
void tnn_set_state(const char *key, tnn_tensor_t *value);
void tnn_drop_state(const char *key);

///
// TENSOR OPERATIONS
// impl: src/ops/*.c
///

// tnn_tensor_t *
// tnn_reshape(tnn_tensor_t *input, const size_t *dims, size_t num_dims);
tnn_tensor_t *tnn_proj(tnn_tensor_t *input, size_t dim_out);
tnn_tensor_t *tnn_bias(tnn_tensor_t *input);
// tnn_tensor_t *tnn_scale(tnn_tensor_t *input);
tnn_tensor_t *tnn_relu(tnn_tensor_t *input);

// - pred is raw logits 2D [batch_size, num_classes]
// - target is one-hot encoded 2D [batch_size, num_classes]
tnn_tensor_t *tnn_cross_entropy(tnn_tensor_t *pred, tnn_tensor_t *target);

// // - input: [..., height, width, in_channels]
// // - kernel_size: size of the square kernel (e.g., 3 for 3x3)
// // - stride: stride for convolution
// // - padding: padding added to all sides
// // - out_channels: number of output channels
// tnn_tensor_t *tnn_conv(
//     tnn_tensor_t *input,
//     size_t out_channels,
//     size_t kernel_size,
//     size_t stride,
//     size_t padding
// );

// // normalization with running mean/var
// // - input: [..., height, width, channels]
// // - momentum: momentum for running mean/var update
// tnn_tensor_t *tnn_bn(tnn_tensor_t *input, float momentum);

///
// BACKPROP
// impl: src/backprop.c
///

void _tnn_zero_grad(const char *scope);
#define tnn_zero_grad(...) OPTARG_FUNC(tnn_zero_grad, __VA_ARGS__)
#define tnn_zero_grad_0() _tnn_zero_grad(NULL)
#define tnn_zero_grad_1(scope) _tnn_zero_grad(scope)

void tnn_backward(tnn_tensor_t *loss);

///
// OPTIMIZERS
// impl: src/optim/*.c
///

typedef struct {
	float lr;
	float b1, b2;
	float eps;
	float wd;
	const char *scope;
} tnn_adamw_cfg_t;

#define TNN_ADAMW_CFG(...)                                                     \
	((tnn_adamw_cfg_t){.lr = 0.001f,                                           \
	                   .b1 = 0.9f,                                             \
	                   .b2 = 0.999f,                                           \
	                   .eps = 1e-4f,                                           \
	                   .wd = 0.01f,                                            \
	                   .scope = NULL,                                          \
	                   __VA_ARGS__})

void tnn_adamw(tnn_adamw_cfg_t cfg);

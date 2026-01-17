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

	// computation graph
	struct tnn_tensor *parents[10];
	size_t num_parents;
	size_t num_children; // ref-count
	bool is_state;       // should not be freed by tnn_free()

	bool requires_grad; // will get a gradient when child's backward() is called
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

// free tensor t, return detached t (see: tnn_detach)
tnn_tensor_t *tnn_detach_free(tnn_tensor_t *t);

void tnn_init_from_memory(tnn_tensor_t *t, const float *data);
void tnn_init_fill(tnn_tensor_t *t, float value);
void tnn_init_randn(tnn_tensor_t *t);

size_t tnn_dim(tnn_tensor_t *t, int32_t i_dim);
size_t tnn_size(tnn_tensor_t *t);

// in case there's less indices than dims, treat leading dims as part of the
// first indexed dim
// - example usage: output->data[tnn_index_at(output, (size_t[]){b, i, j, c},
// 4)] = sum;
// - instead of: output->data[((b * h_out + i) * w_out + j) * c_out + c] = sum;
size_t tnn_index_at(tnn_tensor_t *t, size_t *indices, size_t num_indices);
#define tnn_value_at(t, ...)                                                   \
	t->data[tnn_index_at(                                                      \
	    t,                                                                     \
	    (size_t[]){__VA_ARGS__},                                               \
	    sizeof((size_t[]){__VA_ARGS__}) / sizeof(size_t)                       \
	)]
#define tnn_grad_at(t, ...)                                                    \
	t->grad[tnn_index_at(                                                      \
	    t,                                                                     \
	    (size_t[]){__VA_ARGS__},                                               \
	    sizeof((size_t[]){__VA_ARGS__}) / sizeof(size_t)                       \
	)]

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

tnn_tensor_t *
tnn_reshape(tnn_tensor_t *input, const size_t *dims, size_t num_dims);
tnn_tensor_t *tnn_proj(tnn_tensor_t *input, size_t dim_out);
tnn_tensor_t *tnn_bias(tnn_tensor_t *input);
// tnn_tensor_t *tnn_scale(tnn_tensor_t *input);
tnn_tensor_t *tnn_relu(tnn_tensor_t *input);
tnn_tensor_t *tnn_add(tnn_tensor_t *a, tnn_tensor_t *b);

tnn_tensor_t *_tnn_mean(tnn_tensor_t *input, size_t i_dim, size_t num_dims);
#define tnn_mean(...) OPTARG_FUNC(tnn_mean, __VA_ARGS__)
#define tnn_mean_2(input, i_dim) _tnn_mean(input, i_dim, 1)
#define tnn_mean_3(input, i_dim, num_dims) _tnn_mean(input, i_dim, num_dims)

tnn_tensor_t *_tnn_bn(tnn_tensor_t *input, float momentum, bool test);
#define tnn_bn(...) OPTARG_FUNC(tnn_bn, __VA_ARGS__)
#define tnn_bn_1(input) _tnn_bn(input, 0.9, false)
#define tnn_bn_2(input, momentum) _tnn_bn(input, momentum, false)
#define tnn_bn_3(input, momentum, test) _tnn_bn(input, momentum, test)

// - pred is raw logits 2D [batch_size, num_classes]
// - target is one-hot encoded 2D [batch_size, num_classes]
tnn_tensor_t *tnn_cross_entropy(tnn_tensor_t *pred, tnn_tensor_t *target);

// input dim is [..., height, width, in_channels]
tnn_tensor_t *_tnn_conv(
    tnn_tensor_t *input,
    size_t dim_out,
    size_t kernel_size,
    size_t stride,
    size_t padding
);
#define tnn_conv(...) OPTARG_FUNC(tnn_conv, __VA_ARGS__)
#define tnn_conv_2(input, dim_out) _tnn_conv(input, dim_out, 3, 1, 1)
#define tnn_conv_3(input, dim_out, kernel_size)                                \
	_tnn_conv(input, dim_out, kernel_size, 1, 1)
#define tnn_conv_4(input, dim_out, kernel_size, stride)                        \
	_tnn_conv(input, dim_out, kernel_size, stride, 1)
#define tnn_conv_5(input, dim_out, kernel_size, stride, padding)               \
	_tnn_conv(input, dim_out, kernel_size, stride, padding)

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

void _tnn_adamw(tnn_adamw_cfg_t cfg);
#define tnn_adamw(...) OPTARG_FUNC(tnn_adamw, __VA_ARGS__)
#define tnn_adamw_0() _tnn_adamw(TNN_ADAMW_CFG())
#define tnn_adamw_1(cfg) _tnn_adamw(cfg)

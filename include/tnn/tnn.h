#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "./_optarg.h"

#pragma clang diagnostic ignored "-Winitializer-overrides"

///
// TNN LIFECYCLE + PARAM SYSTEM
// impl: src/tnn.c
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

void tnn_drop(const char *scope);

size_t tnn_params(char **out_keys);

///
// TENSOR STRUCTURE AND UTILS
// impl: src/tensor.c
///

typedef enum {
	TNN_NONE = 0,
	TNN_INPUT = 1 << 0,
	TNN_OUTPUT = 1 << 1,
	TNN_PARAMETER = 1 << 2,
	TNN_BUFFER = 1 << 3,
	TNN_ALL = TNN_INPUT | TNN_OUTPUT | TNN_PARAMETER | TNN_BUFFER
} tnn_tensor_type_t;

typedef struct tnn_tensor {
	float *data;
	float *grad;

	size_t *dims;
	size_t num_dims;

	size_t num_parents;
	struct tnn_tensor *parents[2];
	void (*backward)(struct tnn_tensor *);

	tnn_tensor_type_t type;
} tnn_tensor_t;

void tnn_free(tnn_tensor_t *t, tnn_tensor_type_t types);
void tnn_print(tnn_tensor_t *t);
float tnn_item(tnn_tensor_t *t);
size_t tnn_dim(tnn_tensor_t *t, int32_t i_dim);
size_t tnn_size(tnn_tensor_t *t);
tnn_tensor_t *tnn_data(const size_t *dims, size_t num_dims, const float *data);
tnn_tensor_t *tnn_zeros(const size_t *dims, size_t num_dims);

///
// TENSOR OPERATIONS
// impl: src/ops/*.c
///

tnn_tensor_t *tnn_proj(tnn_tensor_t *input, size_t dim_out);
tnn_tensor_t *tnn_bias(tnn_tensor_t *input);
tnn_tensor_t *tnn_relu(tnn_tensor_t *input);

// - pred is raw logits 2D [batch_size, num_classes]
// - target is one-hot encoded 2D [batch_size, num_classes]
tnn_tensor_t *tnn_cross_entropy(tnn_tensor_t *pred, tnn_tensor_t *target);

///
// BACKPROP
// impl: src/backprop.c
///

void _tnn_zero_grad_impl(const char *scope);
#define tnn_zero_grad(...) OPTARG_FUNC(tnn_zero_grad, __VA_ARGS__)
#define tnn_zero_grad_0() _tnn_zero_grad_impl(NULL)
#define tnn_zero_grad_1(scope) _tnn_zero_grad_impl(scope)

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

// adamw.c
void tnn_adamw(tnn_adamw_cfg_t cfg);

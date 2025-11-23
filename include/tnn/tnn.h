#pragma once

#include <stdint.h>

#pragma clang diagnostic ignored "-Winitializer-overrides"

typedef struct tnn_param_entry {
	char *name;
	struct tnn_vec *vec;
	struct tnn_param_entry *next;
} tnn_param_entry_t;

#define TNN_SCOPE_MAX_LEN 1024
#define TNN_PARAM_TABLE_SIZE 1024

typedef struct {
	char active_scope[TNN_SCOPE_MAX_LEN];
	tnn_param_entry_t *param_table[TNN_PARAM_TABLE_SIZE];
} tnn_globals_t;
extern tnn_globals_t tnn_globals;

// tnn.c
int tnn_init();
void tnn_terminate();

#define TNN_SCOPE(name_fmt, ...)                                               \
	for (int _tnn_once = (tnn_push(name_fmt, ##__VA_ARGS__), 1); _tnn_once;    \
	     tnn_pop(), _tnn_once = 0)

// params.c
void tnn_push(const char *name_fmt, ...);
void tnn_pop();
void tnn_drop(const char *prefix);
void tnn_save(const char *filename);
void tnn_load(const char *filename);

typedef struct tnn_vec {
	float *data;
	float *grad;

	uint32_t *dims;
	uint8_t num_dims;

	uint8_t num_parents;
	struct tnn_vec *parents[2];
	void (*backward)(struct tnn_vec *self);
} tnn_vec_t;

// vec.c
void tnn_free(tnn_vec_t *vec);
void tnn_print(tnn_vec_t *vec);
float tnn_item(tnn_vec_t *vec);
uint32_t tnn_dim(tnn_vec_t *vec, int8_t i_dim);

// proj.c
tnn_vec_t *tnn_proj(tnn_vec_t *input, uint32_t dim_out);
tnn_vec_t *tnn_bias(tnn_vec_t *input);
tnn_vec_t *tnn_relu(tnn_vec_t *input);
tnn_vec_t *tnn_cross_entropy(tnn_vec_t *target, tnn_vec_t *pred);

// grad.c
void tnn_zero_grad(tnn_vec_t *vec);
void tnn_backward(tnn_vec_t *loss);
void tnn_clip_grad_norm(tnn_vec_t *vec, float max_norm);

typedef struct {
	float lr;
	float b1, b2;
	float eps;
	float wd;
} tnn_adamw_cfg_t;

#define TNN_ADAMW_CFG(...)                                                     \
	((tnn_adamw_cfg_t){.lr = 0.001f,                                           \
	                   .b1 = 0.9f,                                             \
	                   .b2 = 0.999f,                                           \
	                   .eps = 1e-8f,                                           \
	                   .wd = 0.01f,                                            \
	                   __VA_ARGS__})

// adamw.c
void tnn_adamw(tnn_vec_t *vec, tnn_adamw_cfg_t cfg);

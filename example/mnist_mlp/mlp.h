#pragma once

#include <tnn/tnn.h>

typedef struct {
	size_t dim_out;
	size_t dim_hidden;
	size_t num_hidden;
} mlp_cfg_t;

#define MLP_CFG(...)                                                           \
	((mlp_cfg_t){                                                              \
	    .dim_out = 10, .dim_hidden = 128, .num_hidden = 2, __VA_ARGS__         \
	})

tnn_tensor_t *mlp(tnn_tensor_t *input, mlp_cfg_t cfg) {
	TNN_SCOPE("mlp") {
		TNN_SCOPE("in") {
			input = tnn_relu(tnn_bias(tnn_proj(input, cfg.dim_hidden)));
		}
		TNN_SCOPE("hidden") {
			for (size_t i = 0; i < cfg.num_hidden; i++) {
				TNN_SCOPE("%zu", i) {
					input = tnn_relu(tnn_bias(tnn_proj(input, cfg.dim_hidden)));
				}
			}
		}
		TNN_SCOPE("out") {
			input = tnn_bias(tnn_proj(input, cfg.dim_out));
		}
	}
	return input;
}

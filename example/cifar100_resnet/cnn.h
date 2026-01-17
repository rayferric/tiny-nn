#pragma once

#include <tnn/tnn.h>

typedef struct {
	size_t dim_out;
	size_t num_layers;
	size_t base_channels;
} cnn_cfg_t;

#define CNN_CFG(...)                                                           \
	((cnn_cfg_t){.num_layers = 1, .base_channels = 32, __VA_ARGS__})

// [N, H, W, C] -> [N, cfg.dim_out]
tnn_tensor_t *cnn(tnn_tensor_t *input, cnn_cfg_t cfg) {
	TNN_SCOPE("cnn") {
		TNN_SCOPE("layers") {
			for (size_t i = 0; i < cfg.num_layers; i++) {
				TNN_SCOPE("%zu", i) {
					size_t out_channels =
					    cfg.base_channels * (1 << i); // 32, 64, 128, ...

					input = tnn_relu(tnn_bias(tnn_conv(
					    tnn_bn(input),
					    TNN_CONV_CFG(
					            .out_channels = out_channels,
					            .kernel_size = 3,
					            .padding = 1,
					            .stride = 1
					    )
					)));

					// TNN_SCOPE("pool") {
					//     input = tnn_conv(
					//         input,
					//         TNN_CONV_CFG(
					//                 .out_channels = out_channels,
					//                 .kernel_size = 2,
					//                 .padding = 0,
					//                 .stride = 2
					//         )
					//     );
					// }
				}
			}
		}
		TNN_SCOPE("fc") {
			input = tnn_reshape(input, (size_t[]){input->dims[0], 0}, 2);
			input = tnn_bias(tnn_proj(input, cfg.dim_out));
		}
	}
	return input;
}

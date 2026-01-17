#pragma once

#include <tnn/tnn.h>

static inline tnn_tensor_t *
resnet_block(tnn_tensor_t *x, size_t dim_out, size_t stride) {
	tnn_tensor_t *skip = x;

	TNN_SCOPE("conv1") {
		x = tnn_relu(tnn_bn(tnn_conv(x, dim_out, 3, stride, 1)));
	}
	TNN_SCOPE("conv2") {
		x = tnn_bn(tnn_conv(x, dim_out, 3, 1, 1));
	}
	if (stride != 1) {
		TNN_SCOPE("skip") {
			skip = tnn_bn(tnn_conv(skip, dim_out, 1, stride, 0));
		}
	}
	return tnn_relu(tnn_add(x, skip));
}

static inline tnn_tensor_t *resnet_layer(
    tnn_tensor_t *x, size_t dim_out, size_t num_blocks, size_t stride
) {
	for (size_t i = 0; i < num_blocks; i++) {
		TNN_SCOPE("block%zu", i) {
			x = resnet_block(x, dim_out, i == 0 ? stride : 1);
		}
	}
	return x;
}

// x shape = NHWC
static inline tnn_tensor_t *resnet(
    tnn_tensor_t *x,
    size_t num_cls,
    size_t base_dim,
    size_t num_layers,
    size_t num_blocks_per_layer
) {
	TNN_SCOPE("resnet") {
		TNN_SCOPE("init") {
			x = tnn_relu(tnn_bn(tnn_conv(x, base_dim, 3, 1, 1)));
		}
		for (size_t i = 0; i < num_layers; i++) {
			size_t dim_out = base_dim * (1 << (i + 1));
			TNN_SCOPE("layer%zu", i) {
				x = resnet_layer(x, dim_out, num_blocks_per_layer, 2);
			}
		}
		TNN_SCOPE("head") {
			x = tnn_mean(x, 1, 2); // -> [N, C]
			x = tnn_proj(x, num_cls);
		}
	}
	return x;
}

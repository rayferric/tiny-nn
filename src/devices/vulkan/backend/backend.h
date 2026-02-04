#pragma once

#include <tnn/tnn.h>

#include "./adamw.h"
#include "./addition.h"
#include "./bn.h"
#include "./ce.h"
#include "./conv.h"
#include "./matmul.h"
#include "./memory.h"
#include "./relu.h"

static _tnn_backend_t backend = {
    .buf_alloc = buf_alloc,
    .buf_free = buf_free,
    .buf_copy = buf_copy,
    .buf_copy_to_host = buf_copy_to_host,
    .buf_copy_to_device = buf_copy_to_device,
    .buf_fill_f = buf_fill_f,
    .matmul = matmul,
    .add = add,
    .accum = accum,
    .sum_reduce = sum_reduce,
    .sum_broadcast = sum_broadcast,
    .relu_fw = relu_fw,
    .relu_bw = relu_bw,
    .ce_fw = ce_fw,
    .ce_bw = ce_bw,
    .conv_fw = conv_fw,
    .conv_bw_input = conv_bw_input,
    .conv_bw_weight = conv_bw_weight,
    .bn_fw = bn_fw,
    .bn_bw = bn_bw,
    .adamw = adamw
};

#pragma once

#include <tnn/tnn.h>

#include "./adamw.h"
#include "./bias.h"
#include "./ce.h"
#include "./matmul.h"
#include "./memory.h"
#include "./relu.h"

static _tnn_backend_t backend = {
    .buf_alloc = buf_alloc,
    .buf_free = buf_free,
    .buf_copy = buf_copy,
    .buf_copy_to_host = buf_copy_to_host,
    .buf_copy_to_device = buf_copy_to_device,
    .matmul = matmul,
    .add = add,
    .accum = accum,
    .sum = sum,
    .relu_fw = relu_fw,
    .relu_bw = relu_bw,
    .ce_fw = ce_fw,
    .ce_bw = ce_bw,
    // todo: conv/bn kernels
    .adamw = adamw
};

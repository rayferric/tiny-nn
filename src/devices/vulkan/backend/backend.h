#pragma once

#include <tnn/tnn.h>

#include "./adamw.h"
#include "./ce.h"
#include "./matmul.h"
#include "./memory.h"

static _tnn_backend_t backend = {
    .buf_alloc = buf_alloc,
    .buf_free = buf_free,
    .buf_copy = buf_copy,
    .buf_copy_to_host = buf_copy_to_host,
    .buf_copy_to_device = buf_copy_to_device,
    .matmul = matmul,
    .ce_fw = ce_fw,
    .ce_bw = ce_bw,
    .adamw = adamw,
};

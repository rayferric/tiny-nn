#pragma once

#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../util/safe_malloc.h"

#include <tnn/tnn.h>

static void *_copy_buffer_between_devices(
    tnn_device_t *old_device,
    tnn_device_t *new_device,
    void *old_buf,
    size_t byte_size
) {
	void *new_buf = new_device->_backend.buf_alloc(new_device, byte_size);

	if (new_device->_is_cpu) {
		old_device->_backend.buf_copy_to_host(
		    old_device, new_buf, old_buf, byte_size
		);
	} else if (old_device->_is_cpu) {
		new_device->_backend.buf_copy_to_device(
		    new_device, new_buf, old_buf, byte_size
		);
	} else {
		void *temp = safe_malloc(byte_size);
		old_device->_backend.buf_copy_to_host(
		    old_device, temp, old_buf, byte_size
		);
		new_device->_backend.buf_copy_to_device(
		    new_device, new_buf, temp, byte_size
		);
		free(temp);
	}

	return new_buf;
}

static void ensure_tensor_on_device(tnn_tensor_t *t, tnn_device_t *new_device) {
	if (t->dev == new_device) {
		return; // already on the correct device
	}

	size_t byte_size = tnn_size(t) * sizeof(float);
	void *old_data = t->data;
	t->data =
	    _copy_buffer_between_devices(t->dev, new_device, t->data, byte_size);
	t->dev->_backend.buf_free(t->dev, old_data);

	if (t->grad) {
		void *old_grad = t->grad;
		t->grad = _copy_buffer_between_devices(
		    t->dev, new_device, t->grad, byte_size
		);
		t->dev->_backend.buf_free(t->dev, old_grad);
	}

	t->dev = new_device;
}

static tnn_tensor_t *
alloc_tensor_on_device(const size_t *dims, size_t num_dims, tnn_device_t *dev) {
	tnn_tensor_t *t = safe_malloc(sizeof(tnn_tensor_t));

	t->dev = dev;

	t->num_dims = num_dims;
	if (num_dims > 0) {
		t->dims = safe_malloc(num_dims * sizeof(size_t));
		memcpy(t->dims, dims, num_dims * sizeof(size_t));
	} else {
		t->dims = NULL;
	}

	size_t total_size = tnn_size(t);
	t->data = t->dev->_backend.buf_alloc(t->dev, total_size * sizeof(float));
	t->grad = NULL;

	t->requires_grad = false;
	t->is_state = false;

	t->num_parents = 0;
	t->num_children = 0;
	t->_backward = NULL;

	t->_ctx = NULL;
	t->_free_ctx = NULL;

	return t;
}

static tnn_tensor_t *alloc_or_get_state_tensor_on_device(
    const size_t *dims,
    size_t num_dims,
    tnn_device_t *dev,
    const char *key,
    bool *allocated
) {
	tnn_tensor_t *t = tnn_get_state(key);
	if (t != NULL) {
		if (allocated) {
			*allocated = false;
		}
		ensure_tensor_on_device(t, dev);
		return t;
	}

	t = alloc_tensor_on_device(dims, num_dims, dev);
	t->is_state = true;
	tnn_set_state(key, t);

	if (allocated) {
		*allocated = true;
	}

	return t;
}

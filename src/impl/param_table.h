#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tnn/tnn.h>

#include "./globals.h"

static uint32_t _hash_string(const char *str) {
	uint32_t hash = 5381;
	int c;
	while ((c = *str++)) {
		hash = ((hash << 5) + hash) + c; // hash * 33 + c
	}
	return hash;
}

static tnn_tensor_t *_tnn_param_get(const char *abs_key) {
	uint32_t hash = _hash_string(abs_key) % TNN_PARAM_TABLE_SIZE;
	tnn_param_entry_t *entry = tnn_globals.param_table[hash];

	while (entry != NULL) {
		if (strcmp(entry->key, abs_key) == 0) {
			return entry->param;
		}
		entry = entry->next;
	}

	return NULL;
}

static void _tnn_param_set(const char *abs_key, tnn_tensor_t *t) {
	uint32_t hash = _hash_string(abs_key) % TNN_PARAM_TABLE_SIZE;

	tnn_param_entry_t *entry = malloc(sizeof(tnn_param_entry_t));
	entry->key = strdup(abs_key); // freed upon release
	entry->param = t;
	entry->next = tnn_globals.param_table[hash];
	// ^ chain with old entry

	tnn_globals.param_table[hash] = entry;
}

#include "./alloc_tensor.h"
#include "./scope_str_utils.h"

static tnn_tensor_t *_tnn_get_or_create_param(
    const char *relative_key,
    const size_t *dims,
    size_t num_dims,
    tnn_tensor_type_t type,
    bool *created
) {
	assert(type == TNN_PARAMETER || type == TNN_BUFFER);

	// prepend active scope
	char full_key[TNN_PARAM_KEY_MAX_LEN];
	_tnn_cat_keys(full_key, tnn_globals.active_scope, relative_key);

	// return if already there
	tnn_tensor_t *param = _tnn_param_get(full_key);
	if (param != NULL) {
		if (created != NULL) {
			*created = false;
		}
		param->type = type;
		return param;
	}

	param = _tnn_alloc_tensor(dims, num_dims, type);
	_tnn_param_set(full_key, param);
	if (created != NULL) {
		*created = true;
	}

	return param;
}

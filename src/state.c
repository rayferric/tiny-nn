#include <tnn/tnn.h>

#include <assert.h>
#include <memory.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "./impl/key_str_utils.h"
#include "./impl/malloc.h"
#include "./impl/state.h"

tnn_state_t tnn_state;

int tnn_init() {
	tnn_state.active_scope[0] = '\0';
	memset(tnn_state.state_dict, 0, sizeof(tnn_state.state_dict));
	return 0;
}

void tnn_terminate() {
	tnn_state.active_scope[0] = '\0';

	// free param table
	for (size_t i = 0; i < TNN_STATE_DICT_SIZE; i++) {
		tnn_state_entry_t *entry = tnn_state.state_dict[i];
		while (entry != NULL) {
			tnn_state_entry_t *next = entry->next;
			free(entry->key);
			entry->param->is_state = false; // allow freeing
			tnn_free(entry->param);
			free(entry);
			entry = next;
		}
		tnn_state.state_dict[i] = NULL;
	}
}

void tnn_push(const char *key_fmt, ...) {
	va_list args;
	va_start(args, key_fmt);

	size_t len = strlen(tnn_state.active_scope);

	// add sep /
	if (len > 0) {
		assert(len + 1 < sizeof(tnn_state.active_scope));
		tnn_state.active_scope[len] = '/';
		len++;
	}

	// append new part
	int num_app = vsnprintf(
	    tnn_state.active_scope + len,
	    sizeof(tnn_state.active_scope) - len,
	    key_fmt,
	    args
	);

	assert(num_app > 0 && len + num_app < sizeof(tnn_state.active_scope));

	va_end(args);
}

void tnn_pop() {
	char *last_slash = strrchr(tnn_state.active_scope, '/');

	if (last_slash != NULL) {
		*last_slash = '\0';
	} else {
		tnn_state.active_scope[0] = '\0';
	}
}

void tnn_save(const char *filename) {
	FILE *fp = fopen(filename, "wb");
	if (fp == NULL) {
		fprintf(stderr, "tnn_save() failed to open: %s\n", filename);
		return;
	}

	for (size_t i = 0; i < TNN_STATE_DICT_SIZE; i++) {
		tnn_state_entry_t *entry = tnn_state.state_dict[i];
		while (entry != NULL) {
			// skip if not under active scope
			if (!_tnn_key_in_scope(entry->key, tnn_state.active_scope)) {
				entry = entry->next;
				continue;
			}

			tnn_tensor_t *t = entry->param;

			const char *relative_key =
			    _tnn_relative_key(entry->key, tnn_state.active_scope);

			// write key
			uint32_t key_len = (uint32_t)strlen(relative_key);
			fwrite(&key_len, sizeof(uint32_t), 1, fp);
			fwrite(relative_key, sizeof(char), key_len, fp);

			// write tensor dims
			uint32_t num_dims = (uint32_t)t->num_dims;
			fwrite(&num_dims, sizeof(uint32_t), 1, fp);
			for (size_t i_dim = 0; i_dim < t->num_dims; i_dim++) {
				uint32_t dim_u32 = (uint32_t)t->dims[i_dim];
				fwrite(&dim_u32, sizeof(uint32_t), 1, fp);
			}

			// write tensor data
			size_t total_size = 1;
			for (uint8_t i_dim = 0; i_dim < t->num_dims; i_dim++) {
				total_size *= t->dims[i_dim];
			}
			fwrite(t->data, sizeof(float), total_size, fp);

			entry = entry->next;
		}
	}

	fclose(fp);
}

static uint32_t _hash_string(const char *str) {
	uint32_t hash = 5381;
	int c;
	while ((c = *str++)) {
		hash = ((hash << 5) + hash) + c; // hash * 33 + c
	}
	return hash;
}

void tnn_load(const char *filename) {
	FILE *fp = fopen(filename, "rb");
	if (fp == NULL) {
		fprintf(stderr, "tnn_load() failed to open: %s\n", filename);
		return;
	}

	while (true) {
		// read key
		uint32_t key_len;
		if (fread(&key_len, sizeof(uint32_t), 1, fp) != 1) {
			break; // eof
		}
		char *relative_key = tnn_safe_malloc(key_len + 1);
		fread(relative_key, sizeof(char), key_len, fp);
		relative_key[key_len] = '\0';

		// read dims
		uint32_t num_dims;
		fread(&num_dims, sizeof(uint32_t), 1, fp);
		size_t *dims = tnn_safe_malloc(num_dims * sizeof(size_t));
		for (size_t i_dim = 0; i_dim < num_dims; i_dim++) {
			uint32_t dim_u32;
			fread(&dim_u32, sizeof(uint32_t), 1, fp);
			dims[i_dim] = dim_u32;
		}

		size_t total_size = 1;
		for (size_t i_dim = 0; i_dim < num_dims; i_dim++) {
			total_size *= dims[i_dim];
		}

		tnn_tensor_t *t = tnn_alloc(dims, num_dims);
		t->is_state = true;

		fread(t->data, sizeof(float), total_size, fp);

		free(dims);

		tnn_set_state(relative_key, t);

		free(relative_key);
	}

	fclose(fp);
}

size_t tnn_list_state_keys(char **out_keys) {
	size_t count = 0;
	for (size_t i = 0; i < TNN_STATE_DICT_SIZE; i++) {
		tnn_state_entry_t *entry = tnn_state.state_dict[i];
		while (entry != NULL) {
			// only list keys in current scope
			if (_tnn_key_in_scope(entry->key, tnn_state.active_scope)) {
				if (out_keys != NULL) {
					const char *relative_key =
					    _tnn_relative_key(entry->key, tnn_state.active_scope);
					out_keys[count] = (char *)relative_key;
				}
				count++;
			}
			entry = entry->next;
		}
	}
	return count;
}

tnn_tensor_t *tnn_get_state(const char *key) {
	// prepend active scope to key
	char full_key[TNN_STATE_KEY_MAX_LEN];
	_tnn_cat_keys(full_key, tnn_state.active_scope, key);

	uint32_t hash = _hash_string(full_key) % TNN_STATE_DICT_SIZE;
	tnn_state_entry_t *entry = tnn_state.state_dict[hash];

	while (entry != NULL) {
		if (strcmp(entry->key, full_key) == 0) {
			return entry->param;
		}
		entry = entry->next;
	}

	return NULL;
}

void tnn_set_state(const char *key, tnn_tensor_t *t) {
	// prepend active scope to key
	char full_key[TNN_STATE_KEY_MAX_LEN];
	_tnn_cat_keys(full_key, tnn_state.active_scope, key);

	uint32_t hash = _hash_string(full_key) % TNN_STATE_DICT_SIZE;

	tnn_state_entry_t *entry = tnn_safe_malloc(sizeof(tnn_state_entry_t));
	entry->key = strdup(full_key); // freed upon release
	entry->param = t;
	entry->next = tnn_state.state_dict[hash];
	// ^ chain with old entry

	tnn_state.state_dict[hash] = entry;
}

void tnn_drop_state(const char *key) {
	// prepend active scope to prefix
	char abs_scope[TNN_STATE_KEY_MAX_LEN];
	_tnn_cat_keys(abs_scope, tnn_state.active_scope, key);

	for (size_t i = 0; i < TNN_STATE_DICT_SIZE; i++) {
		tnn_state_entry_t *entry = tnn_state.state_dict[i];
		tnn_state_entry_t *prev = NULL;

		while (entry != NULL) {
			if (_tnn_key_in_scope(entry->key, abs_scope)) {
				if (prev == NULL) {
					tnn_state.state_dict[i] = entry->next;
				} else {
					prev->next = entry->next;
				}

				entry->param->is_state = false; // allow freeing
				tnn_free(entry->param);
				free(entry->key);

				tnn_state_entry_t *to_free = entry;
				entry = entry->next;
				free(to_free);
			} else {
				prev = entry;
				entry = entry->next;
			}
		}
	}
}

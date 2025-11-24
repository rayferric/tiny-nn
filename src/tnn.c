#include <tnn/tnn.h>

#include <assert.h>
#include <memory.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "./impl/globals.h"

tnn_globals_t tnn_globals;

int tnn_init() {
	tnn_globals.active_scope[0] = '\0';
	memset(tnn_globals.param_table, 0, sizeof(tnn_globals.param_table));
	return 0;
}

void tnn_terminate() {
	tnn_globals.active_scope[0] = '\0';

	// free param table
	for (size_t i = 0; i < TNN_PARAM_TABLE_SIZE; i++) {
		tnn_param_entry_t *entry = tnn_globals.param_table[i];
		while (entry != NULL) {
			tnn_param_entry_t *next = entry->next;
			free(entry->key);
			tnn_free(entry->param, TNN_ALL);
			free(entry);
			entry = next;
		}
		tnn_globals.param_table[i] = NULL;
	}
}

void tnn_push(const char *key_fmt, ...) {
	va_list args;
	va_start(args, key_fmt);

	size_t len = strlen(tnn_globals.active_scope);

	// add sep /
	if (len > 0) {
		assert(len + 1 < sizeof(tnn_globals.active_scope));
		tnn_globals.active_scope[len] = '/';
		len++;
	}

	// append new part
	int num_app = vsnprintf(
	    tnn_globals.active_scope + len,
	    sizeof(tnn_globals.active_scope) - len,
	    key_fmt,
	    args
	);

	assert(num_app > 0 && len + num_app < sizeof(tnn_globals.active_scope));

	va_end(args);
}

void tnn_pop() {
	char *last_slash = strrchr(tnn_globals.active_scope, '/');

	if (last_slash != NULL) {
		*last_slash = '\0';
	} else {
		tnn_globals.active_scope[0] = '\0';
	}
}

#include "./impl/scope_str_utils.h"

void tnn_save(const char *filename) {
	FILE *fp = fopen(filename, "wb");
	if (fp == NULL) {
		fprintf(stderr, "tnn_save() failed to open: %s\n", filename);
		return;
	}

	for (size_t i = 0; i < TNN_PARAM_TABLE_SIZE; i++) {
		tnn_param_entry_t *entry = tnn_globals.param_table[i];
		while (entry != NULL) {
			// skip if not under active scope
			if (!_tnn_key_in_scope(entry->key, tnn_globals.active_scope)) {
				entry = entry->next;
				continue;
			}

			tnn_tensor_t *t = entry->param;

			const char *relative_key =
			    _tnn_relative_key(entry->key, tnn_globals.active_scope);

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

#include "./impl/alloc_tensor.h"
#include "./impl/param_table.h"

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
		char *relative_key = malloc(key_len + 1);
		fread(relative_key, sizeof(char), key_len, fp);
		relative_key[key_len] = '\0';

		// prepend active scope to loaded key
		char full_key[TNN_PARAM_KEY_MAX_LEN];
		_tnn_cat_keys(full_key, tnn_globals.active_scope, relative_key);

		// read dims
		uint32_t num_dims;
		fread(&num_dims, sizeof(uint32_t), 1, fp);
		size_t *dims = malloc(num_dims * sizeof(size_t));
		for (size_t i_dim = 0; i_dim < num_dims; i_dim++) {
			uint32_t dim_u32;
			fread(&dim_u32, sizeof(uint32_t), 1, fp);
			dims[i_dim] = dim_u32;
		}

		size_t total_size = 1;
		for (size_t i_dim = 0; i_dim < num_dims; i_dim++) {
			total_size *= dims[i_dim];
		}

		tnn_tensor_t *t = _tnn_alloc_tensor(dims, num_dims, TNN_NONE);
		// ^ during first forward pass, it is upgraded to PARAMETER or BUFFER
		// upgrade occurs in _tnn_get_or_create_param()

		fread(t->data, sizeof(float), total_size, fp);

		_tnn_param_set(full_key, t);
	}

	fclose(fp);
}

void tnn_drop(const char *scope) {
	// prepend active scope to prefix
	char abs_scope[TNN_PARAM_KEY_MAX_LEN];
	_tnn_cat_keys(abs_scope, tnn_globals.active_scope, scope);

	for (size_t i = 0; i < TNN_PARAM_TABLE_SIZE; i++) {
		tnn_param_entry_t *entry = tnn_globals.param_table[i];
		tnn_param_entry_t *prev = NULL;

		while (entry != NULL) {
			if (_tnn_key_in_scope(entry->key, abs_scope)) {
				if (prev == NULL) {
					tnn_globals.param_table[i] = entry->next;
				} else {
					prev->next = entry->next;
				}

				tnn_free(entry->param, TNN_ALL);
				free(entry->key);

				tnn_param_entry_t *to_free = entry;
				entry = entry->next;
				free(to_free);
			} else {
				prev = entry;
				entry = entry->next;
			}
		}
	}
}

size_t tnn_params(char **out_keys) {
	size_t count = 0;
	for (size_t i = 0; i < TNN_PARAM_TABLE_SIZE; i++) {
		tnn_param_entry_t *entry = tnn_globals.param_table[i];
		while (entry != NULL) {
			if (out_keys != NULL) {
				out_keys[count] = entry->key;
			}
			count++;
			entry = entry->next;
		}
	}
	return count;
}

#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include <tnn/tnn.h>

#include "./state.h"

static bool _tnn_key_in_scope(const char *key, const char *scope) {
	size_t scope_len = strlen(scope);
	if (scope_len == 0) {
		return true;
	}

	if (strncmp(key, scope, scope_len) != 0) {
		return false;
	}

	// ensure exact match down to scope separator
	if (key[scope_len] == '\0' || key[scope_len] == '/') {
		return true;
	}

	return false;
}

// null if not in scope, returns a sub-view of key
static const char *_tnn_relative_key(const char *key, const char *scope) {
	if (!_tnn_key_in_scope(key, scope)) {
		return NULL;
	}

	size_t scope_len = strlen(scope);

	if (scope_len == 0) {
		return key;
	}

	const char *relative = key + scope_len;

	if (relative[0] == '/') {
		relative++;
	}

	return relative;
}

// either key can be NULL or empty, concatenates with '/' separator
static void _tnn_cat_keys(char *out_path, const char *key1, const char *key2) {
	bool has_key1 = (key1 != NULL && key1[0] != '\0');
	bool has_key2 = (key2 != NULL && key2[0] != '\0');

	if (has_key1 && has_key2) {
		snprintf(out_path, TNN_STATE_KEY_MAX_LEN, "%s/%s", key1, key2);
	} else if (has_key1) {
		snprintf(out_path, TNN_STATE_KEY_MAX_LEN, "%s", key1);
	} else if (has_key2) {
		snprintf(out_path, TNN_STATE_KEY_MAX_LEN, "%s", key2);
	} else {
		out_path[0] = '\0';
	}
}

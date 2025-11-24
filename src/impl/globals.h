#pragma once

typedef struct tnn_param_entry {
	char *key;
	struct tnn_tensor *param;
	struct tnn_param_entry *next;
} tnn_param_entry_t;

#define TNN_PARAM_KEY_MAX_LEN 1024
#define TNN_PARAM_TABLE_SIZE 1024

typedef struct {
	char active_scope[TNN_PARAM_KEY_MAX_LEN];
	tnn_param_entry_t *param_table[TNN_PARAM_TABLE_SIZE];
} tnn_globals_t;
extern tnn_globals_t tnn_globals;

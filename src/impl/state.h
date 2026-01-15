#pragma once

typedef struct tnn_state_entry {
	char *key;
	struct tnn_tensor *param;
	struct tnn_state_entry *next;
} tnn_state_entry_t;

#define TNN_STATE_KEY_MAX_LEN 1024
#define TNN_STATE_DICT_SIZE 256

typedef struct {
	char active_scope[TNN_STATE_KEY_MAX_LEN];
	tnn_state_entry_t *state_dict[TNN_STATE_DICT_SIZE];
} tnn_state_t;
extern tnn_state_t tnn_state;

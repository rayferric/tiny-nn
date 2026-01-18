#pragma once

typedef struct tnn_state_entry {
	char *key;
	struct tnn_tensor *param;
	struct tnn_state_entry *next;
} state_entry_t;

#define STATE_DICT_KEY_MAX_LEN 1024
#define STATE_DICT_HASHMAP_SIZE 256

typedef struct {
	char active_scope[STATE_DICT_KEY_MAX_LEN];
	state_entry_t *state_dict[STATE_DICT_HASHMAP_SIZE];
} global_state_t;
extern global_state_t global_state;

int state_init();
void state_terminate();

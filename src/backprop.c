#include <tnn/tnn.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "./impl/key_str_utils.h"
#include "./impl/malloc.h"

static void _tnn_toposort_helper(
    tnn_tensor_t *t,
    tnn_tensor_t ***visited,
    size_t *visited_count,
    size_t *visited_capacity,
    tnn_tensor_t ***result,
    size_t *result_count,
    size_t *result_capacity
) {
	assert(t != NULL);

	// skip if already visited
	for (size_t i = 0; i < *visited_count; i++) {
		if ((*visited)[i] == t) {
			return;
		}
	}

	// mark as visited
	if (*visited_count >= *visited_capacity) {
		*visited_capacity *= 2;
		*visited =
		    realloc(*visited, *visited_capacity * sizeof(tnn_tensor_t *));
	}
	(*visited)[(*visited_count)++] = t;

	// visit all parents
	for (size_t i = 0; i < t->num_parents; i++) {
		_tnn_toposort_helper(
		    t->parents[i],
		    visited,
		    visited_count,
		    visited_capacity,
		    result,
		    result_count,
		    result_capacity
		);
	}

	// add current node to result after all parents
	if (*result_count >= *result_capacity) {
		*result_capacity *= 2;
		*result = realloc(*result, *result_capacity * sizeof(tnn_tensor_t *));
	}
	(*result)[(*result_count)++] = t;
}

static tnn_tensor_t **_tnn_toposort(tnn_tensor_t *t, size_t *count) {
	size_t visited_capacity = 64;
	tnn_tensor_t **visited =
	    tnn_safe_malloc(visited_capacity * sizeof(tnn_tensor_t *));
	size_t visited_count = 0;

	size_t result_capacity = 64;
	tnn_tensor_t **result =
	    tnn_safe_malloc(result_capacity * sizeof(tnn_tensor_t *));
	*count = 0;

	_tnn_toposort_helper(
	    t,
	    &visited,
	    &visited_count,
	    &visited_capacity,
	    &result,
	    count,
	    &result_capacity
	);

	free(visited);
	return result;
}

void _tnn_zero_grad(const char *scope) {
	// prepend active scope to prefix
	char full_prefix[TNN_STATE_KEY_MAX_LEN];
	_tnn_cat_keys(full_prefix, tnn_state.active_scope, scope);

	// go through param table and zero matching grads
	for (size_t i = 0; i < TNN_STATE_DICT_SIZE; i++) {
		tnn_state_entry_t *entry = tnn_state.state_dict[i];
		while (entry != NULL) {
			if (_tnn_key_in_scope(entry->key, full_prefix) &&
			    entry->param->grad != NULL) {
				size_t total_size = tnn_size(entry->param);
				memset(entry->param->grad, 0, total_size * sizeof(float));
			}

			entry = entry->next;
		}
	}
}

void tnn_backward(tnn_tensor_t *loss) {
	assert(loss != NULL);
	assert(tnn_size(loss) == 1 && "tnn_backward: loss must be scalar");

	// allocate initial loss grad wrt itself
	if (loss->grad == NULL) {
		loss->grad = tnn_safe_malloc(sizeof(float));
	}
	loss->grad[0] = 1.0f;

	// pass in reverse topological order
	size_t num_nodes;
	tnn_tensor_t **nodes = _tnn_toposort(loss, &num_nodes);
	for (size_t i = num_nodes; i-- > 0;) {
		tnn_tensor_t *node = nodes[i];
		if (node->backward != NULL) {
			// allocate parent grads if needed
			for (size_t i_parent = 0; i_parent < node->num_parents;
			     i_parent++) {
				tnn_tensor_t *parent = node->parents[i_parent];
				if (parent->requires_grad && parent->grad == NULL) {
					size_t parent_size = tnn_size(parent);
					parent->grad = calloc(parent_size, sizeof(float));
				}
			}

			node->backward(node);

			// free self grad after backward for memory savings
			free(node->grad);
			node->grad = NULL;
		}
	}

	free(nodes);
}

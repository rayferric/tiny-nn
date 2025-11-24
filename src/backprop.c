#include <tnn/tnn.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "./impl/scope_str_utils.h"
#include "./impl/toposort.h"

void _tnn_zero_grad_impl(const char *scope) {
	// prepend active scope to prefix
	char full_prefix[TNN_PARAM_KEY_MAX_LEN];
	_tnn_cat_keys(full_prefix, tnn_globals.active_scope, scope);

	// go through param table and zero matching grads
	for (size_t i = 0; i < TNN_PARAM_TABLE_SIZE; i++) {
		tnn_param_entry_t *entry = tnn_globals.param_table[i];
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
		loss->grad = malloc(sizeof(float));
	}
	loss->grad[0] = 1.0f;

	// pass in reverse topological order
	size_t num_nodes;
	tnn_tensor_t **nodes = _tnn_toposort(loss, &num_nodes);
	for (size_t i = num_nodes; i-- > 0;) {
		tnn_tensor_t *node = nodes[i];
		if (node->backward != NULL && node->type == TNN_OUTPUT) {
			// allocate parent grads if needed
			for (size_t i_parent = 0; i_parent < node->num_parents;
			     i_parent++) {
				tnn_tensor_t *parent = node->parents[i_parent];
				if ((parent->type & (TNN_PARAMETER | TNN_OUTPUT)) &&
				    parent->grad == NULL) {
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

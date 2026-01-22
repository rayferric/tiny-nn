#include <tnn/tnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tensor/impl.h" // for alloc func
#include "./util/key_str_utils.h"

void _tnn_adamw(tnn_adamw_cfg_t cfg) {
	char full_scope[STATE_DICT_KEY_MAX_LEN];
	cat_keys(full_scope, global_state.active_scope, cfg.scope);

	for (size_t i = 0; i < STATE_DICT_HASHMAP_SIZE; i++) {
		state_entry_t *entry = global_state.state_dict[i];
		while (entry != NULL) {
			if (!key_in_scope(entry->key, full_scope) ||
			    !entry->param->requires_grad || entry->param->grad == NULL) {
				entry = entry->next;
				continue;
			}

			tnn_tensor_t *param = entry->param;

			const char *param_rel_key = relative_key(entry->key, full_scope);
			assert(param_rel_key != NULL);

			// get or create m1 and m2 for this param
			tnn_tensor_t *m1, *m2, *timestep;
			TNN_SCOPE("adamw") {
				char m1_rel_key[STATE_DICT_KEY_MAX_LEN];
				char m2_rel_key[STATE_DICT_KEY_MAX_LEN];
				char timestep_rel_key[STATE_DICT_KEY_MAX_LEN];
				cat_keys(m1_rel_key, param_rel_key, "m1");
				cat_keys(m2_rel_key, param_rel_key, "m2");
				cat_keys(timestep_rel_key, param_rel_key, "t");

				bool m1_created = false;
				bool m2_created = false;
				bool timestep_created = false;
				m1 = alloc_or_get_state_tensor_on_device(
				    param->dims,
				    param->num_dims,
				    param->dev,
				    m1_rel_key,
				    &m1_created
				);
				m2 = alloc_or_get_state_tensor_on_device(
				    param->dims,
				    param->num_dims,
				    param->dev,
				    m2_rel_key,
				    &m2_created
				);
				timestep = alloc_or_get_state_tensor_on_device(
				    NULL, 0, tnn_get_cpu(), timestep_rel_key, &timestep_created
				);

				// zero init if newly created
				if (m1_created) {
					tnn_init_fill(m1, 0);
				}
				if (m2_created) {
					tnn_init_fill(m2, 0);
				}
				if (timestep_created) {
					tnn_init_fill(timestep, 0);
				}
			}

			// increment timestep (always on cpu)
			float t = ((float *)timestep->data)[0] + 1.0f;
			((float *)timestep->data)[0] = t;

			param->dev->_backend.adamw(
			    param->dev,
			    param->data,
			    param->grad,
			    m1->data,
			    m2->data,
			    tnn_size(param),
			    t,
			    cfg.lr,
			    cfg.b1,
			    cfg.b2,
			    cfg.eps,
			    cfg.wd
			);

			entry = entry->next;
		}
	}
}

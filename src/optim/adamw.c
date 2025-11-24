#include <tnn/tnn.h>

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./impl/param_table.h"
#include "./impl/scope_str_utils.h"

void tnn_adamw(tnn_adamw_cfg_t cfg) {
	char full_scope[TNN_PARAM_KEY_MAX_LEN];
	_tnn_cat_keys(full_scope, tnn_globals.active_scope, cfg.scope);

	for (size_t i = 0; i < TNN_PARAM_TABLE_SIZE; i++) {
		tnn_param_entry_t *entry = tnn_globals.param_table[i];
		while (entry != NULL) {
			if (!_tnn_key_in_scope(entry->key, full_scope) ||
			    entry->param->type != TNN_PARAMETER ||
			    entry->param->grad == NULL) {
				entry = entry->next;
				continue;
			}

			tnn_tensor_t *param = entry->param;

			const char *param_rel_key =
			    _tnn_relative_key(entry->key, full_scope);
			assert(param_rel_key != NULL);

			// get or create m1 and m2 for this param
			tnn_tensor_t *m1, *m2, *timestep;
			TNN_SCOPE("adamw") {
				char m1_rel_key[TNN_PARAM_KEY_MAX_LEN];
				char m2_rel_key[TNN_PARAM_KEY_MAX_LEN];
				char timestep_rel_key[TNN_PARAM_KEY_MAX_LEN];
				_tnn_cat_keys(m1_rel_key, param_rel_key, "m1");
				_tnn_cat_keys(m2_rel_key, param_rel_key, "m2");
				_tnn_cat_keys(timestep_rel_key, param_rel_key, "t");

				bool m1_created = false;
				bool m2_created = false;
				bool timestep_created = false;
				m1 = _tnn_get_or_create_param(
				    m1_rel_key,
				    param->dims,
				    param->num_dims,
				    TNN_BUFFER,
				    &m1_created
				);
				m2 = _tnn_get_or_create_param(
				    m2_rel_key,
				    param->dims,
				    param->num_dims,
				    TNN_BUFFER,
				    &m2_created
				);
				timestep = _tnn_get_or_create_param(
				    timestep_rel_key, NULL, 0, TNN_BUFFER, &timestep_created
				); // 0-dim scalar

				// zero init if newly created
				if (m1_created) {
					size_t param_size = tnn_size(param);
					memset(m1->data, 0, param_size * sizeof(float));
				}
				if (m2_created) {
					size_t param_size = tnn_size(param);
					memset(m2->data, 0, param_size * sizeof(float));
				}
				if (timestep_created) {
					timestep->data = malloc(sizeof(float));
					timestep->data[0] = 0.0f;
				}
			}

			// increment timestep
			float t = timestep->data[0] + 1.0f;
			timestep->data[0] = t;

			// pre-compute bias correction factors
			float bias1 = 1.0f - powf(cfg.b1, t);
			float bias2 = 1.0f - powf(cfg.b2, t);

			size_t param_size = tnn_size(param);
			for (size_t j = 0; j < param_size; j++) {
				float grad = param->grad[j];

				// update moments
				m1->data[j] = cfg.b1 * m1->data[j] + (1.0f - cfg.b1) * grad;
				m2->data[j] =
				    cfg.b2 * m2->data[j] + (1.0f - cfg.b2) * grad * grad;

				// correct moment biases
				float m_hat = m1->data[j] / bias1;
				float v_hat = m2->data[j] / bias2;

				// update
				param->data[j] -= cfg.lr * (m_hat / (sqrtf(v_hat) + cfg.eps) +
				                            cfg.wd * param->data[j]);
			}

			entry = entry->next;
		}
	}
}

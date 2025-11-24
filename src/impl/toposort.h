#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <tnn/tnn.h>

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
	tnn_tensor_t **visited = malloc(visited_capacity * sizeof(tnn_tensor_t *));
	size_t visited_count = 0;

	size_t result_capacity = 64;
	tnn_tensor_t **result = malloc(result_capacity * sizeof(tnn_tensor_t *));
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

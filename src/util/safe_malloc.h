#pragma once

#include <assert.h>
#include <stdlib.h>

static inline void *safe_malloc(size_t size) {
	void *ptr = malloc(size);
	assert(ptr != NULL && "malloc failed");
	return ptr;
}

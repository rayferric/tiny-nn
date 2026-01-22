#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static inline void *safe_malloc(size_t size) {
	void *ptr = malloc(size);
	if (ptr == NULL) {
		fprintf(stderr, "malloc failed\n");
		exit(1);
	}
	return ptr;
}

static inline void *safe_calloc(size_t nmemb, size_t size) {
	void *ptr = calloc(nmemb, size);
	if (ptr == NULL) {
		fprintf(stderr, "calloc failed\n");
		exit(1);
	}
	return ptr;
}

static inline void *safe_realloc(void *ptr, size_t size) {
	void *new_ptr = realloc(ptr, size);
	if (new_ptr == NULL) {
		fprintf(stderr, "realloc failed\n");
		exit(1);
	}
	return new_ptr;
}

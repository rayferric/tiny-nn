#include <memory.h>
#include <stdio.h>

#include <tnn/tnn.h>

void *cpu_buf_alloc(tnn_device_t *dev, size_t bytes) {
	void *ptr = malloc(bytes);
	if (!ptr) {
		fprintf(stderr, "cpu_alloc: out of memory\n");
		exit(1);
	}
	return ptr;
}
void cpu_buf_free(tnn_device_t *dev, void *ptr) {
	free(ptr);
}
void cpu_buf_copy(tnn_device_t *dev, void *dst, const void *src, size_t bytes) {
	memcpy(dst, src, bytes);
}

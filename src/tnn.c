#include <tnn/tnn.h>

#include <assert.h>
#include <memory.h>
#include <stdarg.h>

#include "./state.h"

#define CHAIN_INIT(fn)                                                         \
	do {                                                                       \
		int status = fn();                                                     \
		if (status != 0)                                                       \
			return status;                                                     \
	} while (0)
int tnn_init() {
	CHAIN_INIT(state_init);
	return 0;
}

void tnn_terminate() {
	state_terminate();
}

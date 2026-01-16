#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CIFAR100_NUM_IMGS 50000
#define CIFAR100_HEIGHT 32
#define CIFAR100_WIDTH 32
#define CIFAR100_CHANNELS 3
#define CIFAR100_NUM_LABELS 100
#define CIFAR100_NUM_CATEGORIES 20

typedef struct {
	float *imgs;
	float *labels;
	float *categories;
} cifar100_t;

static void cifar100_create(cifar100_t *cifar) {
	cifar->imgs = malloc(
	    CIFAR100_NUM_IMGS * CIFAR100_HEIGHT * CIFAR100_WIDTH *
	    CIFAR100_CHANNELS * sizeof(float)
	);
	cifar->labels =
	    calloc(CIFAR100_NUM_IMGS * CIFAR100_NUM_LABELS, sizeof(float));
	cifar->categories =
	    calloc(CIFAR100_NUM_IMGS * CIFAR100_NUM_CATEGORIES, sizeof(float));
}

static void cifar100_destroy(cifar100_t *cifar) {
	if (cifar->imgs) {
		free(cifar->imgs);
		cifar->imgs = NULL;
	}
	if (cifar->labels) {
		free(cifar->labels);
		cifar->labels = NULL;
	}
	if (cifar->categories) {
		free(cifar->categories);
		cifar->categories = NULL;
	}
}

enum { CIFAR100_LOAD_OK, CIFAR100_LOAD_ERROR };
static int cifar100_load(cifar100_t *cifar, const char *data_path) {
	FILE *file = fopen(data_path, "rb");
	if (!file) {
		fprintf(stderr, "could not open CIFAR-100 file: %s\n", data_path);
		return CIFAR100_LOAD_ERROR;
	}

	const size_t feat_map_size = CIFAR100_HEIGHT * CIFAR100_WIDTH;
	const size_t image_size = CIFAR100_CHANNELS * feat_map_size;
	const size_t record_size = 2 + image_size;
	// ^ 2 extra bytes per image for labels

	uint8_t *rec_buf = malloc(record_size);
	for (size_t i = 0; i < CIFAR100_NUM_IMGS; i++) {
		size_t bytes_read = fread(rec_buf, 1, record_size, file);
		if (bytes_read != record_size) {
			fprintf(
			    stderr,
			    "error reading record %zu: got %zu bytes, expected %zu\n",
			    i,
			    bytes_read,
			    record_size
			);
			fclose(file);
			free(rec_buf);
			return CIFAR100_LOAD_ERROR;
		}

		uint8_t coarse_label = rec_buf[0];
		uint8_t fine_label = rec_buf[1];

		cifar->categories[i * CIFAR100_NUM_CATEGORIES + coarse_label] = 1.0f;
		cifar->labels[i * CIFAR100_NUM_LABELS + fine_label] = 1.0f;

		uint8_t *pixels = rec_buf + 2;
		for (size_t c = 0; c < CIFAR100_CHANNELS; c++) {
			for (size_t p = 0; p < CIFAR100_HEIGHT * CIFAR100_WIDTH; p++) {
				// copy to persistent buffer, transpose, normalize values
				size_t pixel_idx = i * image_size + p * CIFAR100_CHANNELS + c;
				cifar->imgs[pixel_idx] = pixels[c * feat_map_size + p] / 255.0f;
			}
		}
	}

	free(rec_buf);
	fclose(file);
	return CIFAR100_LOAD_OK;
}
#include <tnn/tnn.h>

static void cifar100_make_batch(
    cifar100_t *cifar,
    size_t start_idx,
    size_t batch_size,
    tnn_tensor_t **img,
    tnn_tensor_t **label,
    tnn_tensor_t **category
) {
	if (start_idx + batch_size > CIFAR100_NUM_IMGS) {
		batch_size = CIFAR100_NUM_IMGS - start_idx;
	}

	if (img != NULL) {
		size_t offset =
		    start_idx * CIFAR100_HEIGHT * CIFAR100_WIDTH * CIFAR100_CHANNELS;
		*img = tnn_alloc(
		    (size_t[]){
		        batch_size, CIFAR100_HEIGHT, CIFAR100_WIDTH, CIFAR100_CHANNELS
		    },
		    4
		);
		tnn_init_from_memory(*img, &cifar->imgs[offset]);
	}
	if (label != NULL) {
		size_t offset = start_idx * CIFAR100_NUM_LABELS;
		*label = tnn_alloc((size_t[]){batch_size, CIFAR100_NUM_LABELS}, 2);
		tnn_init_from_memory(*label, &cifar->labels[offset]);
	}
	if (category != NULL) {
		size_t offset = start_idx * CIFAR100_NUM_CATEGORIES;
		*category =
		    tnn_alloc((size_t[]){batch_size, CIFAR100_NUM_CATEGORIES}, 2);
		tnn_init_from_memory(*category, &cifar->categories[offset]);
	}
}

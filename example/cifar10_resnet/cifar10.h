#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tnn/tnn.h>

#define CIFAR10_HEIGHT 32
#define CIFAR10_WIDTH 32
#define CIFAR10_CHANNELS 3
#define CIFAR10_NUM_LABELS 10

typedef struct {
	float *imgs;
	float *labels;
	size_t num_imgs;
	size_t capacity;
} cifar10_t;

static void cifar10_create(cifar10_t *cifar) {
	cifar->imgs = NULL;
	cifar->labels = NULL;
	cifar->num_imgs = 0;
	cifar->capacity = 0;
}

static void cifar10_destroy(cifar10_t *cifar) {
	free(cifar->imgs);
	free(cifar->labels);
	cifar->imgs = NULL;
	cifar->labels = NULL;
}

enum { CIFAR10_LOAD_OK, CIFAR10_LOAD_ERROR };

static int cifar10_load(cifar10_t *cifar, const char *data_path) {
	FILE *file = fopen(data_path, "rb");
	if (!file) {
		fprintf(stderr, "could not open CIFAR-10 file: %s\n", data_path);
		return CIFAR10_LOAD_ERROR;
	}

	fseek(file, 0, SEEK_END);
	long file_size = ftell(file);
	fseek(file, 0, SEEK_SET);

	const size_t feat_map_size = CIFAR10_HEIGHT * CIFAR10_WIDTH;
	const size_t image_size = CIFAR10_CHANNELS * feat_map_size;
	const size_t record_size = 1 + image_size;

	size_t batch_imgs = file_size / record_size;

	size_t new_total = cifar->num_imgs + batch_imgs;
	if (new_total > cifar->capacity) {
		float *new_imgs =
		    realloc(cifar->imgs, new_total * image_size * sizeof(float));
		float *new_labels = realloc(
		    cifar->labels, new_total * CIFAR10_NUM_LABELS * sizeof(float)
		);

		if (!new_imgs || !new_labels) {
			fprintf(stderr, "failed to allocate memory for dataset\n");
			free(new_imgs);
			free(new_labels);
			fclose(file);
			return CIFAR10_LOAD_ERROR;
		}

		memset(
		    new_labels + cifar->capacity * CIFAR10_NUM_LABELS,
		    0,
		    (new_total - cifar->capacity) * CIFAR10_NUM_LABELS * sizeof(float)
		);

		cifar->imgs = new_imgs;
		cifar->labels = new_labels;
		cifar->capacity = new_total;
	}

	uint8_t *rec_buf = malloc(record_size);
	size_t start_idx = cifar->num_imgs;

	for (size_t i = 0; i < batch_imgs; i++) {
		if (fread(rec_buf, 1, record_size, file) != record_size) {
			fprintf(stderr, "error reading record %zu\n", i);
			free(rec_buf);
			fclose(file);
			return CIFAR10_LOAD_ERROR;
		}

		uint8_t label = rec_buf[0];
		size_t img_idx = start_idx + i;
		cifar->labels[img_idx * CIFAR10_NUM_LABELS + label] = 1.0f;

		uint8_t *pixels = rec_buf + 1;
		for (size_t c = 0; c < CIFAR10_CHANNELS; c++) {
			for (size_t p = 0; p < feat_map_size; p++) {
				size_t pixel_idx =
				    img_idx * image_size + p * CIFAR10_CHANNELS + c;
				cifar->imgs[pixel_idx] = pixels[c * feat_map_size + p] / 255.0f;
			}
		}
	}

	cifar->num_imgs = new_total;

	free(rec_buf);
	fclose(file);
	return CIFAR10_LOAD_OK;
}

static void cifar10_make_batch(
    cifar10_t *cifar,
    size_t start_idx,
    size_t batch_size,
    tnn_tensor_t **img,
    tnn_tensor_t **label
) {
	if (start_idx + batch_size > cifar->num_imgs) {
		batch_size = cifar->num_imgs - start_idx;
	}

	if (img) {
		size_t offset =
		    start_idx * CIFAR10_HEIGHT * CIFAR10_WIDTH * CIFAR10_CHANNELS;
		*img = tnn_alloc(
		    (size_t[]){
		        batch_size, CIFAR10_HEIGHT, CIFAR10_WIDTH, CIFAR10_CHANNELS
		    },
		    4
		);
		tnn_init_from_memory(*img, &cifar->imgs[offset]);
	}
	if (label) {
		size_t offset = start_idx * CIFAR10_NUM_LABELS;
		*label = tnn_alloc((size_t[]){batch_size, CIFAR10_NUM_LABELS}, 2);
		tnn_init_from_memory(*label, &cifar->labels[offset]);
	}
}

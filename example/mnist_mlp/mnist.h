#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	size_t num_images;
	size_t num_rows;
	size_t num_cols;
	float *images; // flattened images data (num_images * num_rows * num_cols)
	float *labels; // one-hot encoded labels (num_images * 10)
} mnist_t;

// Helper function to read big-endian 32-bit integer
static uint32_t read_be32(FILE *fp) {
	uint8_t bytes[4];
	fread(bytes, 1, 4, fp);
	return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
	       ((uint32_t)bytes[2] << 8) | ((uint32_t)bytes[3]);
}

static void mnist_create(mnist_t *mnist) {
	mnist->num_images = 0;
	mnist->num_rows = 0;
	mnist->num_cols = 0;
	mnist->images = NULL;
	mnist->labels = NULL;
}

static void mnist_destroy(mnist_t *mnist) {
	if (mnist->images) {
		free(mnist->images);
		mnist->images = NULL;
	}
	if (mnist->labels) {
		free(mnist->labels);
		mnist->labels = NULL;
	}
}

static void
mnist_load(mnist_t *mnist, const char *images_path, const char *labels_path) {
	// IMAGES FILE

	FILE *images_file = fopen(images_path, "rb");
	if (!images_file) {
		fprintf(stderr, "could not open images file: %s\n", images_path);
		exit(1);
	}

	uint32_t magic = read_be32(images_file);
	if (magic != 0x00000803) {
		fprintf(stderr, "invalid magic number in images file: 0x%08X\n", magic);
		fclose(images_file);
		exit(1);
	}

	uint32_t num_images = read_be32(images_file);
	uint32_t num_rows = read_be32(images_file);
	uint32_t num_cols = read_be32(images_file);

	mnist->num_images = num_images;
	mnist->num_rows = num_rows;
	mnist->num_cols = num_cols;

	// read image data
	size_t image_size = num_rows * num_cols;
	size_t total_pixels = num_images * image_size;
	uint8_t *raw_images = malloc(total_pixels);
	fread(raw_images, 1, total_pixels, images_file);
	fclose(images_file);
	// convert to 0..1 floats and save to mnist->images
	mnist->images = malloc(total_pixels * sizeof(float));
	for (size_t i = 0; i < total_pixels; i++) {
		mnist->images[i] = raw_images[i] / 255.0f;
	}
	free(raw_images);

	// LABELS FILE

	FILE *labels_file = fopen(labels_path, "rb");
	if (!labels_file) {
		fprintf(stderr, "could not open labels file: %s\n", labels_path);
		exit(1);
	}

	magic = read_be32(labels_file);
	if (magic != 0x00000801) {
		fprintf(stderr, "invalid magic number in labels file: 0x%08X\n", magic);
		fclose(labels_file);
		exit(1);
	}

	uint32_t num_labels = read_be32(labels_file);
	if (num_labels != num_images) {
		fprintf(
		    stderr,
		    "number of labels (%u) does not match number of images (%u)\n",
		    num_labels,
		    num_images
		);
		fclose(labels_file);
		exit(1);
	}

	// read label data
	uint8_t *raw_labels = malloc(num_labels);
	fread(raw_labels, 1, num_labels, labels_file);
	fclose(labels_file);
	// one-hot encode and save to mnist->labels
	mnist->labels = calloc(num_labels * 10, sizeof(float));
	for (size_t i = 0; i < num_labels; i++) {
		uint8_t label = raw_labels[i];
		mnist->labels[i * 10 + label] = 1.0f;
	}
	free(raw_labels);
}

#include <tnn/tnn.h>

static tnn_tensor_t *
mnist_batch_images(mnist_t *mnist, size_t start_idx, size_t batch_size) {
	size_t image_size = mnist->num_rows * mnist->num_cols;

	// Ensure we don't go out of bounds
	if (start_idx + batch_size > mnist->num_images) {
		batch_size = mnist->num_images - start_idx;
	}

	// Create tensor directly from the offset buffer
	size_t offset = start_idx * image_size;
	tnn_tensor_t *batch = tnn_alloc((size_t[]){batch_size, image_size}, 2);
	tnn_init_from_memory(batch, &mnist->images[offset]);
	return batch;
}

static tnn_tensor_t *
mnist_batch_labels(mnist_t *mnist, size_t start_idx, size_t batch_size) {
	// Ensure we don't go out of bounds
	if (start_idx + batch_size > mnist->num_images) {
		batch_size = mnist->num_images - start_idx;
	}

	// Create tensor directly from the offset buffer (already one-hot encoded)
	size_t offset = start_idx * 10;
	tnn_tensor_t *batch = tnn_alloc((size_t[]){batch_size, 10}, 2);
	tnn_init_from_memory(batch, &mnist->labels[offset]);
	return batch;
}

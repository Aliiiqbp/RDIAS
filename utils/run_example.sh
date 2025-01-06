#!/bin/bash

# Set the directory of images to process
IMAGE_DIR="image_dir/"
IMAGE_FORMAT="jpeg"

# Set the error correction algorithm and its parameters
ERROR_CORRECTION_ALG="algorithm1"
N=15
K=11
D=3

# Set the perceptual hash algorithm and its size
HASH_ALG="phash"
HASH_SIZE=8

# Set the watermarking algorithm and its parameters
WATERMARKING_ALG="trustmark"
WATERMARKING_PARAMS='Q'

# Run the main script with the specified parameters
python3.8 main.py --image_dir $IMAGE_DIR --image_format $IMAGE_FORMAT \
               --error_correction_alg $ERROR_CORRECTION_ALG --n $N --k $K --d $D \
               --hash_alg $HASH_ALG --hash_size $HASH_SIZE \
               --watermarking_alg $WATERMARKING_ALG --watermarking_params $WATERMARKING_PARAMS

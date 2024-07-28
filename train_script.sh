#!/bin/bash

# Parameters
num_epochs=20
resize=512

# Run the training script with specified parameters
python3 finetune.py \
    --backbone mobilenet \
    --num_epochs $num_epochs \
    --resize $resize 

python3 finetune.py \
    --backbone resnet50 \
    --num_epochs $num_epochs \
    --resize $resize 


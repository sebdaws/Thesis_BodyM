#!/bin/bash

# Run the commands sequentially
python finetune.py --backbone 'mobilenet' --num_epochs 15 --resize 512 --freeze
python finetune.py --backbone 'mobilenet' --num_epochs 15 --resize 512
python finetune.py --backbone 'resnet50' --num_epochs 15 --resize 512 --freeze
python finetune.py --backbone 'resnet50' --num_epochs 15 --resize 512
python finetune.py --backbone 'resnet101' --num_epochs 15 --resize 512 --freeze
python finetune.py --backbone 'resnet101' --num_epochs 15 --resize 512


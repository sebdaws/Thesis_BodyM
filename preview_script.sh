#!/bin/bash

python previews.py --backbone mobilenet --model_path trained_models/mobilenet_f_100p_15e_512px.pt
python previews.py --backbone mobilenet --model_path trained_models/mobilenet_nf_100p_15e_512px.pt
python previews.py --backbone resnet50 --model_path trained_models/resnet50_f_100p_15e_512px.pt
python previews.py --backbone resnet50 --model_path trained_models/resnet50_nf_100p_15e_512px.pt
python previews.py --backbone resnet101 --model_path trained_models/resnet101_f_100p_15e_512px.pt
python previews.py --backbone resnet101 --model_path trained_models/resnet101_nf_100p_15e_512px.pt

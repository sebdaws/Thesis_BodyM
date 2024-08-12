#!/bin/bash

python test_model.py --backbone mobilenet
python test_model.py --backbone resnet50
python test_model.py --backbone resnet101
python test_model.py --backbone mobilenet --model_path trained_models/mobilenet_f_100p_15e_512px.pt
python test_model.py --backbone mobilenet --model_path trained_models/mobilenet_nf_100p_15e_512px.pt
python test_model.py --backbone resnet50 --model_path trained_models/resnet50_f_100p_15e_512px.pt
python test_model.py --backbone resnet50 --model_path trained_models/resnet50_nf_100p_15e_512px.pt
python test_model.py --backbone resnet101 --model_path trained_models/resnet101_f_100p_15e_512px.pt
python test_model.py --backbone resnet101 --model_path trained_models/resnet101_nf_100p_15e_512px.pt
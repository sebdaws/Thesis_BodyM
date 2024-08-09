#!/bin/bash

python tester.py --model_path './trained_models/measure_net_40e_gender_mibbn.pt' --gender
python tester.py --model_path './trained_models/measure_net_40e_gender_mimpl.pt' --gender --m_inputs
python tester.py --model_path './trained_models/measure_net_40e_mibbn.pt'
python tester.py --model_path './trained_models/measure_net_40e_mimpl.pt' --m_inputs
python tester.py --model_path './trained_models/measure_net_40e_weight_gender_mibbn.pt' --weight --gender
python tester.py --model_path './trained_models/measure_net_40e_weight_gender_mimpl.pt' --weight --gender --m_inputs
python tester.py --model_path './trained_models/measure_net_40e_weight_mibbn.pt' --weight
python tester.py --model_path './trained_models/measure_net_40e_weight_mimpl.pt' --weight --m_inputs

echo "All models have been tested."
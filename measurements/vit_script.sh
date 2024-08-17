#!/bin/bash

conda activate thesis
# Run the commands sequentially
python3 trainer.py --num_epochs 40 --m_inputs --vit
python3 trainer.py --num_epochs 40 --weight --m_inputs --vit
python3 trainer.py --num_epochs 40 --gender --m_inputs --vit
python3 trainer.py --num_epochs 40 --weight --gender --m_inputs --vit
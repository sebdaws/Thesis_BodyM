#!/bin/bash

# Run the commands sequentially
python trainer.py --num_epochs 40
python trainer.py --num_epochs 40 --weight
python trainer.py --num_epochs 40 --gender
python trainer.py --num_epochs 40 --weight --gender
python trainer.py --num_epochs 40 --m_inputs
python trainer.py --num_epochs 40 --weight --m_inputs
python trainer.py --num_epochs 40 --gender --m_inputs
python trainer.py --num_epochs 40 --weight --gender --m_inputs
@echo off
REM Activate the conda environment
call C:\Users\sebastiab\miniconda3\thesis\activate.bat your_environment_name

REM Run the Python commands
python trainer.py --num_epochs 40 --m_inputs --vit
python trainer.py --num_epochs 40 --weight --m_inputs --vit
python trainer.py --num_epochs 40 --gender --m_inputs --vit
python trainer.py --num_epochs 40 --weight --gender --m_inputs --vit

REM Deactivate the conda environment (optional)
call conda deactivate

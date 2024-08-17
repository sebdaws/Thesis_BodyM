# Activate the conda environment
& "C:\Users\Sebastian\miniconda3\Scripts\activate" "thesis"

# Run the Python commands
python trainer.py --num_epochs 40 --m_inputs --vit
python trainer.py --num_epochs 40 --weight --m_inputs --vit
python trainer.py --num_epochs 40 --gender --m_inputs --vit
python trainer.py --num_epochs 40 --weight --gender --m_inputs --vit

# Deactivate the conda environment (optional)
& "conda" deactivate
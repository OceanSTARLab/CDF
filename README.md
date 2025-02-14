# Unveiling The Spatial-Temporal Dynamics: Diffusion-Based Learning of Conditional Distribution for Range-Dependent Ocean Sound Speed Field Forecasting
This repository contains the code for the paper "Unveiling The Spatial-Temporal Dynamics: Diffusion-Based Learning of Conditional Distribution for Range-Dependent Ocean Sound Speed Field Forecasting".
# Description
About 'data': Plz click [here](https://drive.google.com/drive/my-drive) and download the dataset 'hycomdownloaddata' and copy this file folder to the folder 'data'.
About 'config': Some hyperparameters. 
About 'save': To store some results. Get more details from 'utils.py'.
# Usage
You should run these code files in a right order (1-2-3 or 1-2-4).
1. Run 'data_generate_hycom_forecast_ftr.py' to generate data in '.mat' form.
2. Run 'csv_creation_hycom_ftr.py' to generate data in '.csv' form.
3. Run 'exe_SSP_hycom_ftr.py' to generate the results for one turn.
4. Run 'exe_slidewindow_hycom_ftr.py' to generate the results for all turns.
If you want to modify the architecture of the network, try to modify the code in 'diff_models.py' and 'main_model_hycom_ftr'. 

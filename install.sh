#!/bin/bash
export ENV_NAME=securityGPT

echo Installing PyTorch...
conda install pytorch torchvision torchaudio scipy -c pytorch -c conda-forge -y
echo Installing Machine Learning frameworks...
conda install scipy pandas numpy scikit-learn matplotlib seaborn -c conda-forge -y

#installing local repo
pip3 install -e .

echo Virtual environment set up successfully!
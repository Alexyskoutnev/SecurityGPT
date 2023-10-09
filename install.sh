export ENV_NAME=securityGPT  # Replace with your desired environment name

echo Creating conda environment ${ENV_NAME}
conda create -n ${ENV_NAME} python=3.10 -y

echo Activating the environment...
conda activate ${ENV_NAME}

echo Installing PyTorch...
conda install pytorch torchvision torchaudio scipy -c pytorch -c conda-forge -y

echo Installing Machine Learning frameworks...
conda install scipy pandas numpy scikit-learn matplotlib seaborn -c conda-forge -y

echo PyTorch environment set up successfully!

#installing local repo
pip3 install -e .
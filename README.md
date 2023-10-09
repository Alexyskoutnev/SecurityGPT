# SecurityGPT
SecurityGPT is a custom GPT model designed to classify bug report documents. The model was trained on bug reports from Chromium and OpenStack.
## Installation
To automate the installation process, run the following commands: 
```console
conda create -n securityGPT python=3.10
conda activate securityGPT
chmod +x install.sh
./install.sh
```

To install SecurityGPT in your local Python environment, use the following commands:
```console
pip3 -e install .
```
To check all dependencies, run the following command:
```console
pip3 list
```
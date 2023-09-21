## Requirements installation on Mac M1

(versions aren't fixed, just what I've done)

```
pyenv install miniforge3-22.9.0-3
pyenv local miniforge3-22.9.0-3
conda create --name env_tensorflow python=3.9
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
pyenv activate miniforge3-22.9.0-3/envs/env_tensorflow
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
conda install notebook -y
pip install numpy  --upgrade
pip install pandas  --upgrade
pip install matplotlib  --upgrade
pip install scikit-learn  --upgrade
pip install scipy  --upgrade
pip install plotly  --upgrade
```

...in your resound directory:
```
pip install -r requirements.txt
```

I also set an alias that runs "pyenv activate miniforge3-22.9.0-3/envs/env_tensorflow" to get my env set up correctly
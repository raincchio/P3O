# Preconditioner Proximal Policy Optimization

Our code is based on the OpenAI's baseline.[https://github.com/openai/baselines]

## install some dependencies.

```bash
sudo apt-get install libglew-dev patchelf
```

## prepare cuda, cudnn and gcc
```bash
conda install cudatoolkit=10.0 cudnn=7
conda install -c conda-forge gcc=12.1.0
```

## install mujuco

Download MuJoCo[https://github.com/deepmind/mujoco/releases]

For example, we can downloand mujoco210-linux-x86_64.tar.gz for linux

```bash
tar -xvf  mujoco210-linux-x86_64.tar.gz
cd  mujoco210-linux-x86_64
mkdir ~/.mujoco
mv mujoco210 ~/.mujoco

```


## create virtual env with Miniconda
```bash
conda create -n p3o python=3.6
conda activate p3o
pip install -r requirements.txt
```
## install mujoco-py

```bash
pip download mujoco-py==2.0.2.13
tar -xf mujoco*
cd mujoc*
python setup.py install
```

## run experiments
For example, train P3O for HalfCheetah-v2
```bash
export PYTHONPATH=/home/*user*/workspace/P3O
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/*user*/.mujoco/mujoco210/bin:/usr/lib/nvidia
python3 -u -m baselines.run --alg=p3o --num_env=1 --seed=1 --env=HalfCheetah-v2 --num_timesteps=3e6 \
--kl_coef=0.01 --noptepochs=5 --nminibatches=64 --log_path=./HalfCheetah/p3o_s-1_no-5_minib-64_kl-0.01
```

## save model
Save the learned model after finishing the train, and you can modify the code in the baselines/run.py to save the model at an interval.
such as moving it to the baselines/p3o/p3o.py
```bash
python3 -u -m baselines.run --alg=p3o --num_env=1 --seed=1 --env=HalfCheetah-v2 --num_timesteps=3e6 \
--kl_coef=0.01 --noptepochs=5 --nminibatches=64 \
--save_path=./HalfCheetah/p3o_s-1_no-5_minib-64_kl-0.01_model
```
## test model

The learn function will load the model according to the load_path argument and set num_timesteps=0 to skip the training process.
Also, modify the code for easy use, and move the load function in the baselines/p3o/p3o.py to baselines/run.py.
```bash

python3 -u -m baselines.run --alg=p3o --num_env=1 --seed=1 --env=HalfCheetah-v2 --num_timesteps=0 \
--load_path=./HalfCheetah/p3o_s-1_no-5_minib-64_kl-0.01_model --play
```

If you found this paper or code helpful, please consider citing our work.
```
@inproceedings{chen2023sufficiency,
  title={The sufficiency of off-policyness and soft clipping: PPO is still insufficient according to an off-policy measure},
  author={Chen, Xing and Diao, Dongcui and Chen, Hechang and Yao, Hengshuai and Piao, Haiyin and Sun, Zhixiao and Yang, Zhiwei and Goebel, Randy and Jiang, Bei and Chang, Yi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={6},
  pages={7078--7086},
  year={2023}
}
```



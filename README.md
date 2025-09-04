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

For example, we can downloand mujoco200-linux-x86_64.tar.gz[https://www.roboti.us/download/mujoco200_linux.zip] for linux

```bash
tar -xvf  mujoco200-linux-x86_64.tar.gz
cd  mujoco200-linux-x86_64
mkdir ~/.mujoco
mv mujoco200 ~/.mujoco
cd ~/.mujoco

cat <<EOF > mjkey.txt
MuJoCo Pro Individual license activation key, number 7777, type 6.

Issued to Everyone.

Expires October 18, 2031.

Do not modify this file. Its entire content, including the
plain text section, is used by the activation manager.

9aaedeefb37011a8a52361c736643665c7f60e796ff8ff70bb3f7a1d78e9a605
0453a3c853e4aa416e712d7e80cf799c6314ee5480ec6bd0f1ab51d1bb3c768f
8c06e7e572f411ecb25c3d6ef82cc20b00f672db88e6001b3dfdd3ab79e6c480
185d681811cfdaff640fb63295e391b05374edba90dd54cc1e162a9d99b82a8b
ea3e87f2c67d08006c53daac2e563269cdb286838b168a2071c48c29fedfbea2
5effe96fe3cb05e85fb8af2d3851f385618ef8cdac42876831f095e052bd18c9
5dce57ff9c83670aad77e5a1f41444bec45e30e4e827f7bf9799b29f2c934e23
dcf6d3c3ee9c8dd2ed057317100cd21b4abbbf652d02bf72c3d322e0c55dcc24
EOF

```


## create virtual env with Miniconda
```bash
conda create -n p3o python=3.6
conda activate p3o
pip install -r requirements.txt
```
## install mujoco-py

```bash
pip download mujoco-py==2.0.2.13 --no-deps
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
--kl_coef=0.1 --noptepochs=5 --nminibatches=64 --log_path=./HalfCheetah/p3o_s-1_no-5_minib-64_kl-0.1
```

## save model
Save the learned model after finishing the train, and you can modify the code in the baselines/run.py to save the model at an interval.
such as moving it to the baselines/p3o/p3o.py
```bash
python3 -u -m baselines.run --alg=p3o --num_env=1 --seed=1 --env=HalfCheetah-v2 --num_timesteps=3e6 \
--kl_coef=0.1 --noptepochs=5 --nminibatches=64 \
--save_path=./HalfCheetah/p3o_s-1_no-5_minib-64_kl-0.1_model
```
## test model

The learn function will load the model according to the load_path argument and set num_timesteps=0 to skip the training process.
Also, modify the code for easy use, and move the load function in the baselines/p3o/p3o.py to baselines/run.py.
```bash

python3 -u -m baselines.run --alg=p3o --num_env=1 --seed=1 --env=HalfCheetah-v2 --num_timesteps=0 \
--load_path=./HalfCheetah/p3o_s-1_no-5_minib-64_kl-0.1_model --play
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



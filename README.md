# P3O


# install mujuco

Download MuJoCo[https://github.com/deepmind/mujoco/releases]

For example, we can downloand mujoco210-linux-x86_64.tar.gz for linux

```bash
tar -xvf  mujoco210-linux-x86_64.tar.gz
cd  mujoco210-linux-x86_64
mkdir ~/.mujoco
mv mujoco210 ~/.mujoco
```


# create virtual env using Miniconda
```bash
conda create -n p3o python=3.6
conda active p3o
conda install tensorflow-gpu==1.13.0
pip install -r requirements.txt
```
# run experiments
For example, using P3O for HalfCheetah-v2
```bash
export PYTHONPATH=/home/*user*/workspace/P3O
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/*user*/.mujoco/mujoco210/bin:/usr/lib/nvidia
python3 -u -m baselines.run --alg=ddpo --num_env=1 --seed=1 --env=HalfCheetah-v2 --num_timesteps=3e6 --kl_coef=0.01 --noptepochs=5 --nminibatches=64 --log_path=./HalfCheetah/p3o_s-1_no-5_minib-64_kl-0.01
```

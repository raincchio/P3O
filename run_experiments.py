
a = 'export PYTHONPATH=/home/chenxing/tmp/baselines'
b = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chenxing/.mujoco/mujoco200/bin'

for seed in [1,2,3,4]:

    algo = 'spg'
    game = 'HalfCheetah-v2'
    rename = 'spg+5rvkl'
    cmd=[
        "/home/chenxing/env/bin/python",
        "-m",
        "baselines.run",
        "--alg="+algo,
        "--seed="+str(seed),
        "--beta=1",
        "--env="+game,
        "--num_timesteps=3e6",
        "--log_path=/home/chenxing/tmp/logs/"+game+"/"+rename+"_"+str(seed)
    ]
    print(a)
    print(b)
    print(' '.join(cmd))
    print()

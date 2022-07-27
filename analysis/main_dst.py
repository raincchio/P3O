import os
import sys
from subprocess import Popen, PIPE, STDOUT, DEVNULL
import time

def run():
    curenv = os.environ.copy()
    cmds = []
    curenv['PYTHONPATH'] = "/home/chenxing/workspace/baselines"
    curenv['CUDA_VISIBLE_DEVICES'] = "0,1"
    # curenv['LD_LIBRARY_PATH'] = "$LD_LIBRARY_PATH:/home/chenxing/.mujoco/mujoco200/bin"
    games = ["HalfCheetah"]
    for seed in range(4):
        for env in games:
            for lr in range(1,11):
                learningrate = lr/10.0
                cmd=[
                    sys.executable,
                    '-u',
                    '-m',
                    'baselines.run',
                    '--alg=ddpo',
                    '--num_env=1',
                    '--kl_coef='+str(learningrate),
                    '--seed='+str(seed),
                    '--env='+env+'-v2',
                    '--num_timesteps=1e5',
                    '--log_path=/home/chenxing/sense_ana/'+env+'/'+str(learningrate)+'_'+str(seed)
                ]
                cmds.append(cmd)
    running = []
    while cmds:
        while len(running)<24:
            if len(cmds):
                cmd = cmds.pop(-1)
                print(cmd, 'start')
                running.append(Popen(cmd, env=curenv,stdin=PIPE, stdout=DEVNULL, stderr=STDOUT))
            else:
                break
        for idx, p in enumerate(running):
            if p.poll() is not None:
                running.pop(idx)
        time.sleep(300)


if __name__ == "__main__":
    run()
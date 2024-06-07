def mujoco():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=5,
        log_interval=1,
        ent_coef=0,
        kl_coef=0.1,
        lr=lambda f: 3e-4*f,
        cliprange=0.2,
        value_network='copy',
        tau=4
        #random seed 1-6
    )

def atari():
    return dict(
        nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        kl_coef=1,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
        tau=2,
        # value_network='copy'
    )

def retro():
    return atari()

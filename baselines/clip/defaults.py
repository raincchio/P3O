def mujoco():
    return dict(
        nsteps=4096,
        nminibatches=4096,
        lam=0.95,
        gamma=0.99,
        noptepochs=5,
        log_interval=1,
        ent_coef=0.0,
        lr=lambda f: 1e-4*f,
        cliprange=0.2,
        value_network='copy'
    )

def mujoco_bak():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=lambda f: 1e-4*f,
        cliprange=0.2,
        value_network='copy'
    )

def atarr_bak():
    return dict(
        nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
    )
def atari():
    return dict(
        nsteps=5, nminibatches=5,
        lam=0.95, gamma=0.99, noptepochs=3, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
    )

def retro():
    return atari()

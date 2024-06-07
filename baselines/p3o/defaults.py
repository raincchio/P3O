def mujoco():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=5,
        log_interval=1,
        ent_coef=0.,
        kl_coef=0.1,
        lr=lambda f: 3e-4*f,
        cliprange=0.2,
        value_network='copy'
        #random seed 4
    )
def mujoco_bak():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.01,
        kl_coef=1,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy'
    )
# def mujoco_original():
#     return dict(
#         nsteps=2048,
#         nminibatches=32,
#         lam=0.95,
#         gamma=0.99,
#         noptepochs=10,
#         log_interval=1,
#         ent_coef=0.0,
#         kl_coef=1,
#         lr=lambda f: 2.5e-4 * f,
#         cliprange=0.2,
#         value_network='copy'
#     )



# 3e-4
# best_ctn 1e4_fix best_dst 1e4?
# 2.5e-4
def atari():
    return dict(
        nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        kl_coef=1,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
        # value_network='copy'
    )

def retro():
    return atari()

import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize
from baselines.common.input import observation_placeholder
try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, nbatch_act, nbatch_train,
                nsteps, ent_coef, kl_coef, vf_coef, tau, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD
        # observation_placeholder(ob_space, batch_size=nbatch_train)

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        with tf.variable_scope("oldpi", reuse=tf.AUTO_REUSE):
            oldpi = policy(nbatch_train, nsteps, sess,observ_placeholder=train_model.X)
            # oldpi = policy(nbatch_train, nsteps, sess, observation_placeholder=)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.TAU = None

        self.assign = [tf.assign(oldv, newv)
        for (oldv, newv) in zip(get_variables("oldpi"), get_variables("model"))]

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())


        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE) + OLDVPRED
        # Unclipped value
        vf_clip_losses = tf.square(vpredclipped - R)
        # Clipped value
        vf_losses = tf.square(vpred - R)

        vf_loss = tf.reduce_mean(tf.maximum(vf_losses, vf_clip_losses))*0.5

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        fr_kl_loss = kl_coef*oldpi.pd.kl(train_model.pd)
        # fr_kl_loss = kl_coef * train_model.pd.kl(oldpi.pd)

        # tau1 need to be great than or equal to 2 to make soft-clip objective is a lower bound of the original one
        # tau2 control the effective gradient span,
        # to get accelerate effect, by setting 0.5*.5*4/tau1*tau2 >1

        pg_losses2 = -ADV*(4.0/tau)*tf.sigmoid(ratio*tau - tau) # soft clip objective

        # gradient_r_scpi = tf.reduce_mean(-ADV*4*tf.sigmoid(4*ratio - 4)*(1-tf.sigmoid(4*ratio - 4)))

        pg_loss = tf.reduce_mean(pg_losses2)

        pg_loss += tf.reduce_mean(fr_kl_loss)

        #static
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        rAt = tf.reduce_mean(-ADV * ratio)

        # tf.abs(ratio - 1.0) DEON metric
        ptadv = (tf.math.sign(ADV) + 1) / 2
        nta = (-1 * tf.math.sign(ratio -1) + 1) / 2
        ntadv = (-1*tf.math.sign(ADV) + 1) / 2
        pta = (tf.math.sign(ratio -1) + 1) / 2

        unnormal_pt = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), 0)) * ptadv*nta)
        unnormal_nt = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), 0)) * ntadv*pta)
        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('model')
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            # self.trainer = tf.train.GradientDescentOptimizer(learning_rate=LR)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'rAt',"unnormal_pt", 'unnormal_nt']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac, rAt, unnormal_pt, unnormal_nt]

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X: obs,
            self.A: actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE: cliprange,
            self.OLDNEGLOGPAC: neglogpacs,
            self.OLDVPRED: values,
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks
        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]
    def assign_v(self):
        self.sess.run(self.assign)
def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

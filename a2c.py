import time
import joblib
import numpy as np
import tensorflow as tf
import os
import time
import datetime
from setproctitle import setproctitle as ptitle



my_dt_ob  = datetime.datetime.now()
date_list = [my_dt_ob.year, my_dt_ob.month, my_dt_ob.day, my_dt_ob.hour, my_dt_ob.minute, my_dt_ob.second]
date_list = [str(i) for i in date_list]
start_time = "_".join(date_list)

ptitle('a2c.py_{}'.format(start_time))

MODEL_PATH = 'models_entropy_coeff1'
def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)


def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()


#Calculate N-steps return 
# [::-1] just treverses the list

#if (done):
#  reward 
# else:
#   reward =reward + gamma*V(s')
def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]



class Agent:
    def __init__(self, Network, ob_space, ac_space, nenvs, nsteps, nstack,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6)):
        config = tf.ConfigProto(intra_op_parallelism_threads=nenvs,
                                inter_op_parallelism_threads=nenvs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nbatch = nenvs * nsteps

        A = tf.placeholder(tf.int32, [nbatch])  #Action
        ADV = tf.placeholder(tf.float32, [nbatch]) #Advantage
        R = tf.placeholder(tf.float32, [nbatch])  # Return 
        LR = tf.placeholder(tf.float32, [])       #Learning rate can be removed
        
        #Observe that in the step model we have nsteps =1 and reuse =False
        step_model = Network(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False) 
        #Observe that in the train_model we have nsteps = nsteps and reuse =True
        #When we are training then we use the we have to collect the data from all the enviroment so that why we run nsteps time
        train_model = Network(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True) 
        print("Dictionory_of_train",train_model.__dict__) 
        #tf.nn.sparse_softmax_cross_entropy_with_logits
        # if T is one hot encoded (e.g [0,1,0,0])
        # then: -sum(T*log(Y))
        # if T is an index
        #  -logY[i]
        # We want -log pi[action]

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)  #We get policy gradient loss
        vf_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(train_model.vf), R) / 2.0) #Divide by 2
        entropy = tf.reduce_mean(cat_entropy(train_model.pi)) #Entropy loss is used for exploration
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        #Not a regular optimizer
        # Find the gradient and norm of it ...clip it to 0.5
        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_params = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads_and_params)

        
        def train(states,lstm_states, rewards, masks,actions, values):
            advs = rewards - values
            feed_dict = {train_model.X: states, A: actions, ADV: advs, R: rewards, LR: lr, train_model.S:lstm_states,train_model.M:masks}

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                feed_dict
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.initial_state = step_model.initial_state
        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

#The agent is responsible for taking each step in the enviroment.
#The runner is responsible for looping over the steps and accumulating the datameans that state, action, rewards and done flag.

class Runner:
    def __init__(self, env, agent, nsteps=5, nstack=4, gamma=0.99):
        self.env = env
        self.agent = agent
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack)
        self.state = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.update_state(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]
        self.total_rewards = [] # store all workers' total rewards
        self.real_total_rewards = []
        self.lstm_states = self.agent.initial_state
    def update_state(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce IPC overhead
        self.state = np.roll(self.state, shift=-self.nc, axis=3) #We shift the existing observation and add the new state at the end
        self.state[:, :, :, -self.nc:] = obs

    def run(self):
        mb_states, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_lstm_states =  self.lstm_states
        for n in range(self.nsteps):
            actions, values,lstm_states = self.agent.step(self.state,S_=self.lstm_states,M_=self.dones)
            mb_states.append(np.copy(self.state))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, infos = self.env.step(actions)
            for done, info in zip(dones, infos):
                if done:
                    self.total_rewards.append(info['reward'])
                    if info['total_reward'] != -1:
                        self.real_total_rewards.append(info['total_reward'])
            self.lstm_states= lstm_states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.state[n] = self.state[n] * 0
            self.update_state(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_states = np.asarray(mb_states, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.agent.value(self.state, S_=self.lstm_states, M_=self.dones).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_states,mb_lstm_states, mb_rewards,mb_masks,mb_actions, mb_values


def learn(network, env, seed, new_session=True,  nsteps=5, nstack=4, total_timesteps=int(80e6),
          vf_coef=0.5, ent_coef=0.0005, max_grad_norm=5, lr=1e-4,
          epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=250):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    env_id = env.env_id
    ob_space = env.observation_space
    ac_space = env.action_space
    agent = Agent(Network=network, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs,
                  nsteps=nsteps, nstack=nstack,
                  ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm,
                  lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps)
    
    
    if os.path.exists("models_entropy_coeff/Space_inv_A2C_LSTM_nstep8_MAX_rew_546"):
        agent.load("models_entropy_coeff/Space_inv_A2C_LSTM_nstep8_MAX_rew_546")
        print("Loaded_the_model")
    

    #run 5 step of the enviroment for each worker in parallel,collect data and pass it to the Agent for training
    #Eg : if we have 16 worker and each worker has takes 5 steps ,providing <s,a,r,s'> tuple for each step then the batch size is 16*5 =80
    runner = Runner(env, agent, nsteps=nsteps, nstack=nstack, gamma=gamma)
   
    nbatch = nenvs * nsteps
    tstart = time.time()
    # In the loop we call the runner function  and train the agent 
    max_rew = -999
    for update in range(1, total_timesteps // nbatch + 1):
        #states, rewards, actions, values = runner.run()
        obs, lstm_states, rewards, masks, actions, values = runner.run()

        
        policy_loss, value_loss, policy_entropy = agent.train(obs, lstm_states,rewards,masks,actions, values)
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            print("No_of_times_loop_run {} current_timesteps {} policy_entropy {} value_loss {} ".format(total_timesteps // nbatch,update,float(policy_entropy),float(value_loss)))
            # total reward
            r = runner.total_rewards[-100:] # get last 100
            tr = runner.real_total_rewards[-100:]
            if len(tr) == 100:
                print("avg total reward (last 100):", np.mean(tr))
                print("max (last 100):", np.max(tr))
                if(max_rew < np.mean(tr)):
                   savepath = os.path.join(MODEL_PATH,"Space_inv_A2C_LSTM_nstep8_MAX_{}".format(start_time))
                   agent.save(savepath)
                   max_rew  = np.mean(tr)
                   print("Saved_the_max_model") 
            save_name = os.path.join(MODEL_PATH,"Space_inv_A2C_LSTM_nstep8_{}".format(start_time))
            agent.save(save_name)
            #print("Saved_the_model")

    env.close()
    agent.save(save_name)

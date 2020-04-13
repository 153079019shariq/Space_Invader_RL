import numpy as np
import tensorflow as tf


def sample(logits):
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)


def conv(inputs, nf, ks, strides, gain=1.0):
    return tf.layers.conv2d(inputs=inputs, filters=nf, kernel_size=ks,
                            strides=(strides, strides), activation=tf.nn.relu,
                            kernel_initializer=tf.orthogonal_initializer(gain=gain))


def dense(inputs, n, act=tf.nn.relu, gain=1.0):
    return tf.layers.dense(inputs=inputs, units=n, activation=act,
                           kernel_initializer=tf.orthogonal_initializer(gain))


def batch_to_seq(h, nbatch, nsteps, flat=False):
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    print("nbatch {} nsteps {}".format(nbatch,nsteps))
    print("batch_to_seq {}".format(h))
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]


def seq_to_batch(h, flat = False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert(len(shape) > 1)
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def lstm(xs, ms, s, scope, nh, init_scale=1.0):
    nbatch, nin = [v.value for v in xs[0].get_shape()]
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh*4], initializer=ortho_init(init_scale))
        wh = tf.get_variable("wh", [nh, nh*4], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, (x, m) in enumerate(zip(xs, ms)):
        c = c*(1-m)
        h = h*(1-m)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s

def adjust_shape(placeholder, data):
    '''
    adjust shape of the data to the shape of the placeholder if possible.
    If shape is incompatible, AssertionError is thrown

    Parameters:
        placeholder     tensorflow input placeholder

        data            input data to be (potentially) reshaped to be fed into placeholder

    Returns:
        reshaped data
    '''

    if not isinstance(data, np.ndarray) and not isinstance(data, list):
        return data
    if isinstance(data, list):
        data = np.array(data)

    placeholder_shape = [x or -1 for x in placeholder.shape.as_list()]

    assert _check_shape(placeholder_shape, data.shape), \
        'Shape of data {} is not compatible with shape of the placeholder {}'.format(data.shape, placeholder_shape)

    return np.reshape(data, placeholder_shape)




class CNN:

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nlstm = 128
        gain = np.sqrt(2)
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        X_normal = tf.cast(X, tf.float32) / 255.0
        self.state = tf.constant([])
        self.lstm_states = tf.constant([])
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states
        with tf.variable_scope("model", reuse=reuse):
            h1 = conv(X_normal, 32, 8, 4, gain)
            h2 = conv(h1, 64, 4, 2, gain)
            h3 = conv(h2, 64, 3, 1, gain)
            h3 = tf.layers.flatten(h3)
            h4 = dense(h3, 512, gain=gain)

            
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, scope='lstm', nh=nlstm)
            h = seq_to_batch(h5)
            
            print("shape_of_snew",snew.shape)
            print("Shape_of_h4",h4.shape)
            print("Shape_of_M",M)
            print("Shape_of_S",S)

            print("Shape_of_xs",len(xs),xs[0].shape)
            print("Shape_of_ms",len(ms),ms[0].shape)
                     
            pi = dense(h, ac_space.n, act=None)
            vf = dense(h, 1, act=None)
            self.initial_state = np.zeros(S.shape.as_list(), dtype=float)
        v0 = vf[:, 0]
        a0 = sample(pi)
        # self.initial_state = []  # State reserved for LSTM

        def step(ob,S_,M_):
            #print("CNN_MODEL###################################")
            #print(S_)
            #print(M_)
            a, v,lstm_states = sess.run([a0, v0,snew], {X: ob,S:S_,M:np.array(M_)})
            return a, v,lstm_states #, []  # dummy state

        def value(ob,S_,M_):
            return sess.run(v0, {X: ob,S:S_,M:np.array(M_)})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

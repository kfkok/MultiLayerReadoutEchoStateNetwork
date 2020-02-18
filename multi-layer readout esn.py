import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.initializers import RandomNormal
import gym
import sys
import matplotlib.pyplot as plt
from simulation import Grid

# config the console to display wider output
np.set_printoptions(linewidth=1320)

# turn off eager execution
tf.compat.v1.disable_eager_execution()

class MLRESN():
    def __init__(self, state_size, action_size, reservoir_size=1000,
                 spectral_radius=0.99, n_drop=0, leak=0.1, reservoir_scale=1.2,
                 connection_probability=0.1, noise_level=0.01, print=True):

        self.input_size = state_size
        self.reservoir_size = reservoir_size
        self.action_size = action_size
        self.leak = leak
        self.reservoir_state = np.zeros(reservoir_size)
        self.reservoir_scale = reservoir_scale
        self.third_layer_buffer = np.zeros(10)

        # init input weights sampled from uniform random numbers between -1 to 1
        self.input_weights = np.random.uniform(-1, 1, size=(reservoir_size, state_size))

        # init reservoir neuron recurrent weights
        variance = 1 / (connection_probability * reservoir_size)
        self.recurrent_weights = np.random.normal(loc=0, scale=variance, size=(reservoir_size, reservoir_size))

        # init multi layer neural network (mlnn)
        input = Input(batch_shape=(None, reservoir_size+state_size))
        layer_1 = (Dense(100, activation='tanh', input_dim=reservoir_size+state_size, kernel_initializer=RandomNormal(mean=0.0, stddev=np.sqrt(0.01/100))))(input)
        layer_2 = (Dense(40, activation='tanh', kernel_initializer=RandomNormal(mean=0.0, stddev=np.sqrt(0.01/40))))(layer_1)
        layer_3 = (Dense(10, activation='tanh', name='third_layer', kernel_initializer=RandomNormal(mean=0.0, stddev=np.sqrt(0.01/10))))(layer_2)
        actions = (Dense(action_size, activation='tanh', name='actions'))(layer_3)
        value = (Dense(1, activation='linear', name='value'))(layer_3)
        actions_and_value = concatenate([actions, value])

        # we need two outputs: first, the third layer output with actor and critic and the second last layer output for feedback
        self.mlnn = Model(inputs=input, outputs=actions_and_value)
        outputs = [layer_3, actions_and_value]
        self.predict = K.function([self.mlnn.input, K.learning_phase()], outputs)

        # the train operator
        target_placeholder = Input(batch_shape=(None, action_size+1))
        loss = mean_squared_error(self.mlnn.output, target_placeholder)
        optimizer = Adam(lr=0.001)
        train_op = optimizer.get_updates(params=self.mlnn.trainable_weights, loss=loss)
        self._train = K.function(inputs=[self.mlnn.input, target_placeholder], outputs=[self.mlnn.output, loss], updates=[train_op])

        # init the feedback weights from third layer of mlnn to reservoir, uniform random numbers between 0 to 1
        self.fb_weights = np.random.uniform(-1, 1, size=(reservoir_size, 10))

    def get_output(self, input):
        # compute the internal reservoir state
        self.reservoir_state = self.compute_internal_state(input)

        # mlnn read out where mlnn input = [input, reservoir state]
        mlnn_input = np.hstack((input, np.tanh(self.reservoir_state)))
        self.third_layer_buffer, action, value = self.mlnn_readout(mlnn_input)

        # action exploration
        #explore = np.random.uniform(-1, 1, size=self.action_size)
        explore = np.random.normal(scale=0.5, size=(self.action_size))

        return mlnn_input, action, explore, value

    def compute_internal_state(self, input):
        rec = self.recurrent_weights.dot(np.tanh(self.reservoir_state))
        inp = self.input_weights.dot(input)
        fb = self.fb_weights.dot(self.third_layer_buffer)
        return (1 - self.leak) * self.reservoir_state + self.leak * (self.reservoir_scale * rec + inp + fb)

    def mlnn_readout(self, input):
        if input.ndim == 1:
            input = input[np.newaxis]

        third_layer_output, output = self.predict([input])
        action = output[:, 0:-1]
        value = output[:, -1]
        return third_layer_output.flatten(), action, value

    def train_mlnn(self, mlnn_input, action, value):
        mlnn_input = mlnn_input[np.newaxis]
        value = np.array([value])
        target = np.hstack((action, value))
        return self._train([mlnn_input, target])

    def test_mlnn(self):
        env = gym.make("Pendulum-v0")
        n_samples = 3
        input_size = self.mlnn.input_shape[-1]
        input = np.random.rand(n_samples, input_size)
        # target_action = np.asarray([env.action_space.sample() for i in range(n_samples)])
        target_action = np.random.uniform(-1, 1, size=(n_samples, 2))
        target_value = np.random.uniform(-3, 5, size=n_samples)

        for i in range(400):
            for t in range(n_samples):
                pred, loss = esn.train_mlnn(input[t], target_action[t], target_value[t])
                print("prediction:", pred, "loss:", loss)

        print("target action:", target_action, "target value:", target_value)

        print("testing predicting")
        _, act, val = self.mlnn_readout(input)
        print("act:", act, "val", val)

    def reset(self):
        self.reservoir_state.fill(0)
        self.third_layer_buffer.fill(0)

    def save(self, filename):
        np.save(filename +"_input_weights", self.input_weights)
        np.save(filename +"_recurrent_weights", self.recurrent_weights)
        np.save(filename +"_fb_weights", self.fb_weights)
        self.mlnn.save(filename+".h5")
        print("model saved successfully..")

    def load(self, filename):
        try:
            self.input_weights = np.load(filename + "_input_weights.npy")
            self.recurrent_weights = np.load(filename + "_recurrent_weights.npy")
            self.fb_weights = np.load(filename + "_fb_weights.npy")
            self.mlnn = load_model(filename + '.h5', compile=False)
            print("model loaded successfully..")
        except IOError:
            print("File " + filename + " not available")

    def print_weights(self):
        print("ESN weights:")
        print("input weights:", self.input_weights)
        print("recurrent weights:", self.recurrent_weights)
        print("fb weights:", self.fb_weights)
        print("mlnn:", self.mlnn.get_weights())


# env = gym.make("Pendulum-v0"), env.observation_space.shape[0],env.action_space.shape[0]
env = Grid()
esn = MLRESN(state_size=env.state_size, action_size=env.action_size)

# esn.test_mlnn()
# exit()

# load the previously trained model
esn.load("sample")

N_EPISODES = 1
render = True
discount_rate = 0.99
online_learning = True

for e in range(N_EPISODES):
    total_rewards = 0
    done = False
    state = env.reset()
    t = 0

    value_trace = []
    target_value_trace = []
    action_trace = []
    target_action_trace = []

    # Stores all information from the network and the received reward on a particular state
    # so network can be trained on next state
    previous_mlnn_input = None
    previous_value = 0
    previous_action = None
    previous_explore = 0
    previous_reward = None

    # reset the reservoir state
    esn.reset()

    while not done:
        mlnn_input, action, explore, value = esn.get_output(state)
        print("action:", action, "val:", value, "explore:", explore)
        next_state, reward, done = env.step(action + explore)
        total_rewards += reward

        # train if t more than 0
        if t > 0 and online_learning:
            # previous reward is the reward for reaching the current state
            td_error = previous_reward + discount_rate * value - previous_value
            if done:
                target_value = previous_value
            else:
                target_value = previous_value + td_error

            target_action = previous_action + td_error * previous_explore
            target_action = previous_action
            if (td_error > 0):
                target_action += previous_explore

            # target_action = np.array([[0.33, 0.33]])
            # target_value = [0.88]

            esn.train_mlnn(previous_mlnn_input, target_action, target_value)

        # store this information about this state for use in next state
        previous_mlnn_input = mlnn_input
        previous_value = value
        previous_action = action
        previous_explore = explore
        previous_reward = reward

        t += 1
        state = next_state.flatten()

        if done:
            print("ep:", e, "total rewards:", total_rewards)
            #env.render()
            if total_rewards >= -1:
                env.save_figure("Results/" + str(np.round(total_rewards, 2)).replace('.', '_'))

            # save the network every 500 episodes
            if ((e+1) % 500 == 0) & (e > 0):
                esn.save("sample")



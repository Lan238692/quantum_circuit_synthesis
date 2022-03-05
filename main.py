# %%
import numpy as np
from learn import learn
from environment import CompEnv
from simulator import SIMULATOR
from model import Policy
import time
import tensorflow_probability as tfp
import winsound
# %%
# learning setup
network_size = 256
batch_size = 64
epochs = 40
lr = 1e-4
num_qubit = 4
circuit_length = 20
# Specification of gates to be used for synthesis
gates = ["Controlled-X", "Controlled-V", "Controlled-V+"]
sim = SIMULATOR(num_qubit, circuit_length, gates=gates)
env = CompEnv(sim)
# Number of training data to be generated
num_circuit = 100000
# name of train data and model
train_data = f"q{sim.n}_cl{sim.m}_n{num_circuit//1000}k"
model_name = f"./model/q{sim.n}_cl{sim.m}_ns{network_size}_bs{batch_size}_{epochs}epochs"
# %%
# learn network
policy = learn(
    model_name=model_name,
    train_data=train_data,
    sim=sim,
    network_size=network_size,
    batch_size=batch_size,
    epochs=epochs,
    lr=lr)
winsound.Beep(800,1000)  #ビープ音（800Hzの音を1000msec流す)
# %%
class Node:
    def __init__(self, state, priority) -> None:
        self.state = state
        self.priority = priority
        self.next_node = [None for _ in range(priority.shape[1])]

def equal(sv1, sv2):
    epsilon = 1e-8
    real = ((sv1.real-sv2.real)**2 < epsilon).all()
    imag = ((sv1.imag-sv2.imag)**2 < epsilon).all()
    return real and imag

def search(sim, policy, env):
    eye = np.eye(2**sim.n, dtype=np.complex64)
    state = env.get_state()
    root = Node(state, policy(env.encode_state(state)))
    searched = []
    # while True:
    for _ in range(1000):
        node = root
        p = []
        for i in range(sim.m):
            cdist = tfp.distributions.Categorical(probs=node.priority)
            action = cdist.sample().numpy()[0]
            p.append(action)
            if node.next_node[action] == None:
                next_state = env.calc([action], node.state)
                priority = policy(env.encode_state(next_state))
                node.next_node[action] = Node(next_state, priority)
            node = node.next_node[action]
            if equal(node.state, eye):
                return p[::-1], len(searched)
        x = [p[i]*(sim.m**i) for i in range(len(p))]
        btinsert(searched, sum(x))

def btinsert(a, x):
    hi = len(a)
    if hi == 0:
        a.insert(0, x)
        return True
    lo = 0
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid] < x: lo = mid+1
        elif x < a[mid]: hi = mid
        else: return False
    a.insert(lo, x)
    return True
# %%
# load the learned model
env.reset([0])
dummy = env.encode_state(env.get_state())
try:
    policy(dummy)
except NameError:
    policy = Policy(sim.ACTION_SPACE, network_size=network_size)
    policy(dummy)
    policy.load_weights(model_name + ".h5")
# %%
# test
# read the test_cl*.py and test it
from test_cl20 import gates, circuits
test_num = 100
with open("./result/model20_tsg.txt", mode="w") as f:
    for name, program in circuits.items():
        length = len(program)
        sim = SIMULATOR(num_qubit, length, gates=gates)
        program = [sim.gate2gn(name, idx) for name, idx in program]
        env.reset(program)
        time_sum = []
        synthesized_circuit = []
        searched_space = []
        for _ in range(test_num):
            try:
                t = time.time()
                p, s = search(sim, policy, env)
                time_sum.append(time.time() - t)
                synthesized_circuit.append(p)
                searched_space.append(s)
            except TypeError:
                pass
        time_sum = np.array(time_sum)
        searched_space = np.array(searched_space)
        f.write(f"{name}\n")
        f.write(f"time_average:{np.mean(time_sum):.2f}[sec], time_std:{np.std(time_sum):.2f}[sec]\n")
        f.write(f"searched_apace_average:{np.mean(searched_space):.2f}, searched_space_std:{np.std(searched_space):.2f}\n")
        for p in synthesized_circuit:
            f.write(str(p)+"\n")
            f.write(str([sim.gn2gate_name(g) for g in p])+"\n")
        f.write("\n\n")
        print(name, " finished")
winsound.Beep(800,1000)  #ビープ音（800Hzの音を1000msec流す)
# %%
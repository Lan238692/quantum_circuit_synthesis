import numpy as np
import qulacs
from dataclasses import dataclass


class CompEnv():

    def __init__(self, sim) -> None:

        self.sim = sim
    
    def reset(self, program):
        self.history = []
        self.program = program

        # 全基数で初期化
        qs = qulacs.QuantumState(self.sim.n)
        self.state = []
        for i in range(2**self.sim.n):
            qs.set_computational_basis(i)
            self.state.append(qs.get_vector())
        self.state = np.array(self.state)

        for gate in program:
            self.history.append(self.state)
            self.state = np.array([self.sim.calc([gate], x) for x in self.state])
            for h in self.history:
                if equal(h, self.state):
                    return False

        # self.state = [self.sim.calc(program, x) for x in self.state]
        return True

    def update(self, action):
        #: 変数を更新する
        self.state = [self.sim.calc(action, x, T=True) for x in self.state]
        return np.array(self.state)

    def step(self, action):
        #: 変数を更新しない
        tmp = np.array([self.sim.calc(action, x, T=True) for x in self.state])
        return tmp

    def calc(self, action, state):
        tmp = np.array([self.sim.calc(action, x, T=True) for x in state])
        return tmp

    def get_state(self):
        return np.array(self.state)

    def encode_state(self, state):
        return np.array([state.real, state.imag]).reshape(1, 2, -1)



class Qstate:

    def __init__(self, sim, source, target):
        self.sim = sim
        self.source = source[:, 0, :] + 1j*source[:, 1, :]
        self.target = target[:, 0, :] + 1j*target[:, 1, :]


    def step(self, action_for, action_back):
        f = np.array(
            [self.sim.calc([action_for], x) for x in self.source]
        )
        b = np.array(
            [self.sim.calc([action_back], x) for x in self.target]
        )

        return self.encode_state(f, b)


    def encode_state(self, source, target):
        forward = np.concatenate([source, target], axis=1)
        forward = np.array([[x.real, x.imag] for x in forward])
        backward = np.concatenate([target, source], axis=1)
        backward = np.array([[x.real, x.imag] for x in backward])

        return forward, backward


    def get_state(self):
        return self.encode_state(self.source, self.target)


@dataclass
class Node:
    target: int
    controll: int

class DAG:

    def __init__(self, sim, program) -> None:
        self.sim = sim
        self.n = sim.n
        self.program = program
        self.idx = list(range(len(program)))
        self.dag = [[] for _ in range(self.n)]
        self.nodes = []
        for i in range(len(program)):
            name, idx = sim.gn2gate_name(program[i])

            if len(self.dag[idx[0]]) == 0:
                t = None
            else:
                t = self.dag[idx[0]][-1]
            if len(self.dag[idx[1]]) == 0:
                c = None
            else:
                c = self.dag[idx[1]][-1]
            self.nodes.append(Node(t, c))

            self.dag[idx[0]].append(i)
            self.dag[idx[1]].append(i)

    def get_root(self):
        subset = set(self.idx)
        for i in self.idx:
            node = self.nodes[i]
            t = node.target
            c = node.controll
            subset.discard(t)
            subset.discard(c)
        for g in subset:
            self.idx.remove(g)
        return list(subset)


def equal(sv1, sv2):
    epsilon = 1e-8
    real = ((sv1.real-sv2.real)**2 < epsilon).all()
    imag = ((sv1.imag-sv2.imag)**2 < epsilon).all()
    return real and imag


# %%
import numpy as np
import random
import itertools
import tensorflow as tf

from simulator import SIMULATOR
from environment import CompEnv, DAG
# %%
def make_data(sim, num, data_name):
    action_space = sim.ACTION_SPACE
    env = CompEnv(sim)
    X, Y = [], []
    created = []
    for i in range(num):
        #: 回路をランダム生成する
        # p = [random.randrange(action_space) for _ in range(sim.m)]
        while True:
            t = random.randrange(action_space**sim.m)
            while not btinsert(created, t) and len(created) < action_space**sim.m:
                t = random.randrange(action_space**sim.m)
            if len(created) >= action_space**sim.m:
                break
            p = []
            for i in range(sim.m):
                p.append(t % action_space)
                t = t // action_space
            if env.reset(p):
                break

        #: 全ての基数に回路を適用して初期化
        dag = DAG(sim, p)
        length = len(p)
        #: トポロジーソートに基づいて入力と出力を生成してリストに入れる
        while length > 0:
            #: Xに現在の状態を加える
            x = env.get_state()
            # x = env.encode_state(x)
            X.append(env.encode_state(x).reshape(2, -1))
            #: 正解になるゲートを探す
            root = dag.get_root()
            # root = [p[j] for j in root]
            y = sum([tf.one_hot(p[r], action_space, dtype=tf.float32) for r in root])
            Y.append(y)
            #: 全ての組み合わせを考える
            for i in range(1, len(root)):
                for comb in itertools.combinations(root, i):
                    x = env.step([p[j] for j in comb])
                    y = list(set(root) - set(comb))
                    # y = [p[j] for j in y]
                    y = sum([tf.one_hot(p[r], action_space, dtype=tf.float32) for r in y])
                    # X = np.concatenate([X, env.encode_state(x)], axis=0)
                    X.append(env.encode_state(x).reshape(2, -1))
                    Y.append(y)
            #: 状態を更新する
            env.update([p[r] for r in root])
            length -= len(root)
    #: 全てのパターンを入れ終えたら学習
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    np.save(f"./data/X_"+data_name, X)
    np.save(f"./data/Y_"+data_name, Y)

    return X, Y


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
gates = ["Controlled-X", "Controlled-V", "Controlled-V+"]
sim = SIMULATOR(4, 20, gates=gates)
num_circuit = 100000
train_data = f"q{sim.n}_cl{sim.m}_n{num_circuit//1000}k"
X, Y = make_data(sim, num_circuit, train_data)
# %%
print(Y.shape)
# %%
import winsound
winsound.Beep(800,1000)  #ビープ音（800Hzの音を1000msec流す)

# %%

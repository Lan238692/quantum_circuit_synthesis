from pathlib import Path
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.python.eager.backprop import GradientTape
from tqdm import tqdm

from model import Policy




def learn(model_name, train_data, sim, network_size=256, batch_size=256, epochs=50, lr=1e-4):

    action_space = sim.ACTION_SPACE

    policy = Policy(action_space=action_space, network_size=network_size)
    optimizer = tf.keras.optimizers.Adam(lr=lr)

    logdir = Path(__file__).parent / ("log/" + model_name)
    if logdir.exists():
        shutil.rmtree(logdir)
    summary_writer = tf.summary.create_file_writer(str(logdir))

    X = np.load("./data/X_" + train_data + ".npy")
    Y = np.load("./data/Y_" + train_data + ".npy")

    num_data = X.shape[0]
    iteration = 0

    for n in tqdm(range(epochs)):

        #: prepare data
        sff_idx = np.random.permutation(num_data)


        for idx in range(0, num_data, batch_size):
            batch_x = X[sff_idx[idx: idx+batch_size if idx+batch_size < num_data else num_data]]
            batch_y = Y[sff_idx[idx: idx+batch_size if idx+batch_size < num_data else num_data]]

            with GradientTape() as tape:

                action_probs = policy(batch_x)

                loss = -1 * tf.reduce_sum(
                    batch_y * tf.math.log(action_probs + 1e-5),
                    axis=1, keepdims=True
                )
                loss = tf.reduce_mean(loss)


            grads = tape.gradient(loss, policy.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, policy.trainable_variables)
            )

            with summary_writer.as_default():
                tf.summary.scalar("loss", loss, step=iteration)

            iteration += 1

            if iteration % 10000 == 0:
                print(f"iter:{iteration}, loss:{loss}")

    policy.save_weights(model_name+".h5", save_format="h5")

    return policy
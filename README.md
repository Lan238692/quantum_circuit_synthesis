# Quantum Circuit Synthesis Using Neurak Network.

Automatic synthesis of quantum circuits from specific quantum gates and input/output specifications of quantum circuits.
Note that this is not good code as it was haphazardly expanded (especially simulator.py).
Currently, the supported gates are quantum gates with two arguments, such as CX gates and CV gates.

dependency library:
* numpy
* tensorflow
* tensorflow_probability
* qulacs
* dataclasses
* tqdm

The executable files are make_train_data.py and main.py. 
The former produces training data, the latter contains code for training and testing the network.

simulator.py:
This file is an interface for quantum computation and is also used to generate quantum gates. SIMULATOR class specifies the number of qubits in the first argument, the length of the quantum circuit in the second argument, and the type of quantum gate to use in the third argument ( The default is H-gate, T-gate, and CX-gate).

Components.py:
Specifies the minimum unit for generating a matrix of quantum gates in SIMULATOR class.

environment.py:
CompEnv class extends the SIMULATOR calculation to 2^n.
DAG class is used to generate training data.
Qstate class should not be used at this time.

model.py:
Policy class describes the structure of the network.
Cx2DAffine class defines a complex neural network.

learn.py:
Functions for learning networks.

How to describe quantum circuits:
examples are test_cl*.py.
HNG circuit is described as follows:
[
        ("Controlled-V", (3, 0)),
        ("Controlled-V", (3, 1)),
        ("Controlled-V", (3, 2)),
        ("Controlled-X", (2, 0)),
        ("Controlled-X", (2, 1)),
        ("Controlled-V+", (3, 2))
]
Each element of the list is (type of gate, specification of gate arguments). The first gate argument is the target bit and the second is the control bit.

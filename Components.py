import numpy as np

GateUnit = {}

# H
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
GateUnit["H"] = (0, H)

# T
T = np.array([[1, 0], [0, np.e**(1j*np.pi/4)]], dtype=np.complex128)
GateUnit["T"] = (0, T)

# Z
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
GateUnit["Z"] = (0, Z)

# CNOT
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
GateUnit["Controlled-X"] = (1, X)

# CV
plus = np.array([[1, 1], [1, 1]], dtype=np.complex128) / 2
minus = np.array([[1, -1], [-1, 1]], dtype=np.complex128) / 2
V = plus + 1j*minus
GateUnit["Controlled-V"] = (1, V)

# CV+
Vdg = plus + ((-1)**(-1/2))*minus
GateUnit["Controlled-V+"] = (1, Vdg)

# CH
GateUnit["Controlled-H"] = (1, H)

# CZ
GateUnit["Controlled-Z"] = (1, Z)

# SWAP
GateUnit["SWAP"] = (2, None)
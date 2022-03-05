import numpy as np
import bisect

class SIMULATOR:
    def __init__(self, n, m, gates=["H", "T", "Controlled-X"]) -> None:
        self.n = n
        self.m = m
        self.gates = gates
        self.btree = []
        self.matrices = []
        s = 0
        for gate in self.gates:
            matrix = self.load_gate_matrix(gate)
            self.matrices.append(matrix)
            s += len(matrix)
            self.btree.append(s)
        self.ACTION_SPACE = self.btree[-1]

    def load_gate_matrix(self, gate):
        path = "./matrices/n{}_{}.npy".format(self.n, gate)
        try:
            matrix = np.load(path)
        except FileNotFoundError:
            print("Not found file: ", path)
            matrix = self.generate_matrix(gate)
        return matrix

    def generate_matrix(self, gate):
        # generate gate matrix and save npy file
        # search gate name "gate" from Components.py 
        # GateUnit["gate"] = (prtcl, gate_unit)
        # prtcl: determine how to generate
        # gate_unit: matrix for minimum qubit
        print("generate {} gate for {} qubit".format(gate, self.n))

        def tensordot(l):
            matrix = l[0]
            for m in l[1:]:
                matrix = np.tensordot(matrix, m, axes=0)
            return matrix

        def reshape(matrix, n):
            if n == 1:
                return matrix

            else:
                lt = reshape(matrix[0, 0], n-1)
                lb = reshape(matrix[0, 1], n-1)
                rt = reshape(matrix[1, 0], n-1)
                rb = reshape(matrix[1, 1], n-1)

                l = np.vstack([lt, lb])
                r = np.vstack([rt, rb])
                return np.hstack([l, r])

        from Components import GateUnit
        prtcl, gu = GateUnit[gate]
        matrices = []

        if prtcl == 0:
            # single qubit gate
            # H, T, X
            for i in range(self.n):
                matrix = []
                for j in range(self.n):
                    if i == j:
                        matrix.append(gu)
                    else:
                        matrix.append(np.eye(2, dtype=np.complex128))
                matrix = tensordot(matrix)
                matrix = reshape(matrix, self.n)
                matrices.append(matrix)

        elif prtcl == 1:
            # multiple qubit gate
            # Controlled-Gate
            # CNOT, CV
            for i in range(self.n*(self.n-1)):
                target, controll = self.num2comb(i)
                m1 = []
                m2 = []
                for j in range(self.n):
                    if j == controll:
                        m1.append(np.array([[1, 0], [0, 0]], dtype=np.complex128))
                        m2.append(np.array([[0, 0], [0, 1]], dtype=np.complex128))
                    elif j == target:
                        m1.append(np.eye(2, dtype=np.complex128))
                        m2.append(gu)
                    else:
                        m1.append(np.eye(2, dtype=np.complex128))
                        m2.append(np.eye(2, dtype=np.complex128))
                m1 = tensordot(m1)
                m2 = tensordot(m2)
                matrices.append(reshape(m1 + m2, self.n))

        elif prtcl == 2:
            # SWAP
            CNOT = self.load_gate_matrix("Controlled-X")
            for i in range(self.n):
                for j in range(i+1, self.n):
                    num = self.comb2num(i, j)
                    num2 = self.comb2num(j, i)
                    matrix = np.dot(CNOT[num], CNOT[num2])
                    matrix = np.dot(matrix, CNOT[num])
                    matrices.append(matrix)

        matrices = np.array(matrices)
        np.save("./matrices/n{}_{}".format(self.n, gate), matrices)

        return matrices

    def comb2num(self, target, controll):
        num = target * (self.n - 1) + controll
        if target <= controll:
            num -= 1
        return num

    def num2comb(self, num):
        target = num // (self.n - 1)
        controll = num % (self.n - 1)
        if target <= controll:
            controll += 1
        return target, controll

    def comb2num_swap(self, target, controll):
        num = controll - (target +1)
        tmp = self.n-1
        while target > 0:
            num += tmp
            tmp -= 1
            target -= 1
        return num

    def num2comb_swap(self, num):
        target = 0
        tmp = self.n - 1
        while num >= tmp:
            target += 1
            num -= tmp
            tmp -= 1
        return target, num+target+1

    def gn2gate(self, gn):
        if gn >= self.ACTION_SPACE:
            raise ValueError(f"gn: {gn}")
        gate_idx = bisect.bisect_right(self.btree, gn)
        if gate_idx == 0:
            idx = gn
        else:
            idx = gn -self.btree[gate_idx-1]
        return gate_idx, idx

    def gn2gate_name(self, gn):
        gate_idx, idx = self.gn2gate(gn)
        name = self.gates[gate_idx]
        if "Controlled" in name:
            idx = self.num2comb(idx)
        elif "SWAP" in name:
            idx = self.num2comb_swap(idx)
        if type(idx) == int:
            idx = [idx]
        return name, idx

    def gate2gn(self, gate_name, idx):
        gate_idx = self.gates.index(gate_name)
        if "Controlled" in gate_name:
            idx = self.comb2num(idx[0], idx[1])
        elif "SWAP" in gate_name:
            idx = self.comb2num_swap(idx[0], idx[1])
        else:
            idx = idx[0]
        gn = idx
        if gate_idx > 0:
            gn += self.btree[gate_idx-1]
        return gn

    def check(self, sv1, sv2):
        # check equality of state vector sv1 and sv2
        val = np.sum(sv1 == sv2) == len(sv1)
        return val

    def calc(self, program, sv, T=False):
        for gn in program:
            gate, idx = self.gn2gate(gn)
            mat = self.matrices[gate][idx]
            if T:
                mat = np.conjugate(mat)
            sv = np.dot(mat, sv)
        return sv
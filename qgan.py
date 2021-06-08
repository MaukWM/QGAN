import math

import matplotlib.pyplot as plt

from data import OneClusters

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
import numpy as np


def derivative(f, theta, epoch):
    return 0.5 * (f * (theta + math.pi / (2 * math.sqrt(epoch + 1))) - f * (theta - math.pi / (2 * math.sqrt(epoch + 1))))

derivatives = np.vectorize(derivative)

class QGAN:

    def __init__(self, epochs):
        self.data_generator = OneClusters()

        self.data = self.data_generator.generate(250)

        self.d_thetas = np.random.random_sample((4,)) * math.pi
        self.g_thetas = np.random.random_sample((4,)) * math.pi

        self.epochs = epochs

        self.alpha = 0.1

        # print(self.thetas)

    # In order to train with real data
    def build_discriminator(self, real_x, real_y):
        q = QuantumRegister(5)
        c = ClassicalRegister(1)
        circuit = QuantumCircuit(q, c)

        circuit.initialize([math.sqrt(1 - real_x**2), real_x], 3)
        circuit.initialize([math.sqrt(1 - real_y**2), real_y], 4)

        circuit.h(q[0])

        circuit.ry(self.d_thetas[0], q[1])
        circuit.ry(self.d_thetas[1], q[2])

        circuit.ryy(self.d_thetas[2], q[1], q[2])
        circuit.cry(self.d_thetas[3], q[1], q[2])

        circuit.cswap(q[0], q[1], q[3])
        circuit.cswap(q[0], q[2], q[4])

        circuit.h(q[0])
        circuit.measure(q[0], c[0])

        # Draw the circuit
        # print(circuit.draw())
        # plot = circuit.draw()
        # plot.show()

        return circuit

    def build_full_circuit(self):
        # q0 is for reading result
        # q1, q2 are the discriminator
        # q3, q4 are the generator
        q = QuantumRegister(5)
        c = ClassicalRegister(1)
        circuit = QuantumCircuit(q, c)

        # Randomly initialize q3 and q4 for the generator
        # theta1 = np.random.rand(1)[0]
        # initial_vector1 = np.array([np.cos(theta1), np.sin(theta1)])
        # theta2 = np.random.rand(1)[0]
        # initial_vector2 = np.array([np.cos(theta2), np.sin(theta2)])
        #
        # circuit.initialize(initial_vector1, 3)
        # circuit.initialize(initial_vector2, 4)

        circuit.h(q[0])

        circuit.ry(self.d_thetas[0], q[1])
        circuit.ry(self.d_thetas[1], q[2])
        circuit.ry(self.g_thetas[0], q[3])
        circuit.ry(self.g_thetas[1], q[4])

        circuit.ryy(self.d_thetas[2], q[1], q[2])
        circuit.ryy(self.g_thetas[2], q[3], q[4])

        circuit.cry(self.d_thetas[3], q[1], q[2])
        circuit.cry(self.g_thetas[3], q[3], q[4])

        circuit.cswap(q[0], q[1], q[3])
        circuit.cswap(q[0], q[2], q[4])

        circuit.h(q[0])
        circuit.measure(q[0], c[0])

        # Draw the circuit
        # plot = circuit.draw(output='mpl')
        # plot.show()

        return circuit

    def build_generator(self):
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        circuit = QuantumCircuit(q, c)

        # Randomly initialize q3 and q4 for the generator
        # theta1 = np.random.rand(1)[0]
        # initial_vector1 = np.array([np.cos(theta1), np.sin(theta1)])
        # theta2 = np.random.rand(1)[0]
        # initial_vector2 = np.array([np.cos(theta2), np.sin(theta2)])
        #
        # circuit.initialize(initial_vector1, 0)
        # circuit.initialize(initial_vector2, 1)

        circuit.ry(self.g_thetas[0], q[0])
        circuit.ry(self.g_thetas[1], q[1])

        circuit.ryy(self.g_thetas[2], q[0], q[1])

        circuit.cry(self.g_thetas[3], q[0], q[1])

        circuit.measure(q[0], c[0])
        circuit.measure(q[1], c[1])

        # plot = circuit.draw(output='mpl')
        # plot.show()

        return circuit

    @staticmethod
    def run_simulation(circuit, shot_count):
        backend = Aer.get_backend('qasm_simulator')
        job = execute(backend=backend, experiments=circuit, shots=shot_count)
        counts = job.result().get_counts()
        # if shot_count == 1:
        #     return int(list(counts.keys())[0])
        prob = QGAN.counts_to_prob(counts)
        return prob

    def cost(self, n):
        real_data = [(0, 0), (1, 1)]
        seeds = [0.5, 0.7]
        result = np.mean([np.log(self.discriminator(x)) for x in real_data]) + np.mean([np.log(1-self.discriminator(self.generator(z))) for z in seeds])
        return 0

    def real_cost(self, circuit):
        runs = 30
        result = self.run_simulation(circuit, runs)
        result_prob = np.abs((result / runs - 0.5) / 0.5)

        return np.log(result_prob), result

        # if result == 0:
        #     return 0, result
        # elif result == 1:
        #     #TODO: fix this it's bad
        #     return 1, result
        # else:

    def fake_cost(self, circuit):
        runs = 30
        result = self.run_simulation(circuit, runs)
        result_prob = np.abs((result / runs - 0.5) / 0.5)
        # if result == 0:
        #     return 0, result
        # elif result == 1:
        #     #TODO: fix this it's bad
        #     return 1, result
        # else:
        return np.log(1-result_prob), result

    def train(self):
        pass

if __name__ == '__main__':
    qgan = QGAN(epochs=10)
    qgan.train()
    qgan.plot_generator_distribution()

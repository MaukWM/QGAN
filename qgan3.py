import math

import matplotlib.pyplot as plt
from qiskit.algorithms.optimizers import COBYLA

from data import OneClusters, TwoClusters

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
import numpy as np


def derivative(f, theta, epoch):
    return 0.5 * (f * (theta + math.pi / (2 * math.sqrt(epoch + 1))) - f * (theta - math.pi / (2 * math.sqrt(epoch + 1))))

derivatives = np.vectorize(derivative)


class QGAN2:

    def __init__(self, epochs):
        self.data_generator = OneClusters()

        self.data = self.data_generator.generate(250)

        self.d_thetas = np.random.random_sample((4,)) * math.pi
        self.g_thetas = np.random.random_sample((4,)) * math.pi

        self.epochs = epochs

        self.alpha = 0.005

        self.steps_per_epoch = 20

        # print(self.thetas)

    @staticmethod
    def run_simulation(circuit, shot_count):
        backend = Aer.get_backend('qasm_simulator')
        job = execute(backend=backend, experiments=circuit, shots=shot_count)
        counts = job.result().get_counts()
        # if shot_count == 1:
        #     return int(list(counts.keys())[0])
        prob = QGAN2.counts_to_prob(counts)
        return prob

    # In order to train with real data
    def build_discriminator(self, real_x, real_y):
        q = QuantumRegister(3)
        c = ClassicalRegister(1)
        circuit = QuantumCircuit(q, c)

        circuit.initialize([math.sqrt(1 - real_x**2), real_x], 1)
        circuit.initialize([math.sqrt(1 - real_y**2), real_y], 2)

        circuit.h(q[0])

        circuit.ry(self.d_thetas[0], q[1])
        circuit.ry(self.d_thetas[1], q[2])

        circuit.ryy(self.d_thetas[2], q[1], q[2])
        circuit.cry(self.d_thetas[3], q[1], q[2])

        circuit.cnot(q[0], q[1])
        circuit.cnot(q[0], q[2])

        circuit.h(q[0])
        circuit.measure(q[0], c[0])

        # Draw the circuit
        # print(circuit.draw())
        # plot = circuit.draw()
        # plot.show()

        return circuit

    def build_generator(self):
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        circuit = QuantumCircuit(q, c)

        circuit.ry(self.g_thetas[0], q[0])
        circuit.ry(self.g_thetas[1], q[1])

        circuit.ryy(self.g_thetas[2], q[0], q[1])

        circuit.cry(self.g_thetas[3], q[0], q[1])

        circuit.measure(q[0], c[0])
        circuit.measure(q[1], c[1])

        # plot = circuit.draw(output='mpl')
        # plot.show()

        return circuit

    def objective_function_generator(self, g_thetas):
        self.g_thetas = g_thetas
        cost = 0
        for i in range(26):
            (gen_x, gen_y) = self.generate_point()
            d_circuit = self.build_discriminator(gen_x, gen_y)
            prob = self.run_simulation(d_circuit, 30)

            # cost += abs(0.75 - gen_x) + abs(0.25 - gen_y)

            cost += (1 - prob) ** 2

        self.plot_generator_distribution(1)
        print("G", cost, g_thetas)

        return cost

    def objective_function_discriminator(self, d_thetas):
        self.d_thetas = d_thetas
        cost = 0
        for i in range(13):
            (gen_x, gen_y) = self.generate_point()
            d_circuit = self.build_discriminator(gen_x, gen_y)
            prob = self.run_simulation(d_circuit, 30)

            cost += prob ** 2

        idx = np.random.randint(len(self.data), size=26)
        xs = self.data[idx]

        for x in xs:
            d_circuit = self.build_discriminator(x[0], x[1])
            prob = self.run_simulation(d_circuit, 30)

            cost += (1 - prob) ** 2

        self.generate_discriminator_heatmap()
        print("D", cost, d_thetas)
        return cost

    def shared_objective_function(self, parameters):
        self.d_thetas = parameters[:len(parameters) // 2]
        self.g_thetas = parameters[len(parameters) // 2:]

        cost = 0

        # Generator cost
        for i in range(50):
            (gen_x, gen_y) = self.generate_point()
            d_circuit = self.build_discriminator(gen_x, gen_y)
            prob = self.run_simulation(d_circuit, 30)

            cost += (1 - prob)

        # Dicriminator cost
        for i in range(25):
            (gen_x, gen_y) = self.generate_point()
            d_circuit = self.build_discriminator(gen_x, gen_y)
            prob = self.run_simulation(d_circuit, 30)

            cost += prob

        idx = np.random.randint(len(self.data), size=25)
        xs = self.data[idx]

        for x in xs:
            d_circuit = self.build_discriminator(x[0], x[1])
            prob = self.run_simulation(d_circuit, 30)

            cost += (1 - prob)

        # self.plot_generator_distribution(0)
        # print(cost)

        return cost

    def shared_train(self):
        optimizer = COBYLA(maxiter=100, tol=0.001)

        opt_param = optimizer.optimize(num_vars=(len(self.d_thetas) + len(self.g_thetas)),
                                       objective_function=self.shared_objective_function,
                                       initial_point=np.concatenate((self.d_thetas, self.g_thetas), axis=None))

        print("done", opt_param)

        self.d_thetas = opt_param[0][:len(opt_param[0]) // 2]
        self.g_thetas = opt_param[0][len(opt_param[0]) // 2:]

    def train(self):
        asd = 0
        for i in range(25):
            asd += 1
            optimizer = COBYLA(maxiter=10, tol=0.001)

            opt_param = optimizer.optimize(num_vars=len(self.d_thetas),
                                           objective_function=self.objective_function_discriminator,
                                           initial_point=self.d_thetas)

            self.d_thetas, d_cost, _ = opt_param
            # print("D", opt_param)

            opt_param = optimizer.optimize(num_vars=len(self.g_thetas),
                                           objective_function=self.objective_function_generator,
                                           initial_point=self.g_thetas)

            self.g_thetas, g_cost, _ = opt_param
            # print("G", opt_param)

            # # Add some shared training?
            # opt_param = optimizer.optimize(num_vars=(len(self.d_thetas) + len(self.g_thetas)),
            #                                objective_function=self.shared_objective_function,
            #                                initial_point=np.concatenate((self.d_thetas, self.g_thetas), axis=None))
            # self.d_thetas = opt_param[0][:len(opt_param[0]) // 2]
            # self.g_thetas = opt_param[0][len(opt_param[0]) // 2:]

            # print("G", opt_param)
            # d_cost = 0
            # g_cost = 0
            print(f"Epoch {i}, G_cost: {g_cost:.2f}, D_cost: {d_cost:.3f}")

            # if asd % 5 == 0:
            self.plot_generator_distribution(i)
            #     self.generate_discriminator_heatmap()

    def generate_discriminator_heatmap(self):
        xs = np.arange(0, 1, 0.05)
        ys = np.arange(0, 1, 0.05)

        heatmap = np.zeros(shape=(20, 20))

        for x_i in range(len(xs)):
            for y_i in range(len(ys)):
                d_circuit = self.build_discriminator(xs[x_i], ys[y_i])
                prob = self.run_simulation(d_circuit, 30)
                #TODO: Check heatmap, inverted probabilities?
                heatmap[x_i][y_i] = prob

        heatmap = heatmap.T

        plt.imshow(heatmap)
        plt.colorbar()
        plt.gca().invert_yaxis()

        plt.show()

    def generate_point(self):
        g_circuit = self.build_generator()
        shots = 30

        backend = Aer.get_backend('qasm_simulator')
        job = execute(backend=backend, experiments=g_circuit, shots=shots)
        result = job.result()

        counts = result.get_counts(g_circuit)
        #counts = QGAN.run_simulation(g_circuit, 30)

        (x, y) = self.counts_to_point(counts, shots)

        # print(x, y)
        # print(counts)
        return x, y

    def plot_generator_distribution(self, epoch):
        # First plot the desired distribution
        plt.scatter(*map(list, zip(*self.data)), color='red')

        # Plot generated points
        points = []
        for i in range(50):
            points.append(self.generate_point())
        plt.scatter(*map(list, zip(*points)), color='blue')

        # Plot disc points

        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.title("Epoch {}".format(epoch + 1))

        plt.show()

    @staticmethod
    def counts_to_prob(counts):
        if '0' in counts.keys():
            x_0 = counts['0']
        else:
            x_0 = 0
        if '1' in counts.keys():
            x_1 = counts['1']
        else:
            x_1 = 0
        return x_1 / (x_0 + x_1)

    def counts_to_point(self, counts, shots):
        # For each point: x_i/N where N=shots from generator
        x_0 = 0
        x_1 = 0
        y_0 = 0
        y_1 = 0

        for key in counts.keys():
            if key[0] == '0':
                x_0 += counts[key]
            else:
                x_1 += counts[key]
            if key[1] == '0':
                y_0 += counts[key]
            else:
                y_1 += counts[key]
        # print(x_0, x_1, y_0, y_1)

        return x_1/shots, y_1/shots


if __name__ == '__main__':
    qgan = QGAN2(epochs=150)
    # qgan.build_generator().draw(output='mpl').show()
    # qgan.build_discriminator(0.5, 0.7).draw(output='mpl').show()
    # qgan.plot_generator_distribution()
    qgan.train()
    # qgan.generate_discriminator_heatmap()

    # Below is to only optimize the Generator

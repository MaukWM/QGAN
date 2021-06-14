import math

import matplotlib.pyplot as plt
from qiskit.algorithms.optimizers import COBYLA

#from data import OneClusters, TwoClusters

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
import numpy as np

from data import Distribution

def derivative(f, theta, epoch):
    return 0.5 * (f * (theta + math.pi / (2 * math.sqrt(epoch + 1))) - f * (theta - math.pi / (2 * math.sqrt(epoch + 1))))

derivatives = np.vectorize(derivative)


class QGAN4:

    def __init__(self, num_of_dimensions, epochs=20):
        self.epochs = epochs
        self.steps_per_epoch = 10
        self.alpha = 0.0025

        self.num_of_dimensions = num_of_dimensions
        self.g_layers = 1
        self.d_layers = 1

        # self.g_thetas_j = np.random.random_sample((g_n_qubits + self.k * g_n_qubits,)) * math.pi
        self.g_thetas = np.random.random_sample((self.g_layers + 1, self.num_of_dimensions)) * math.pi
        self.d_thetas = np.random.random_sample((self.d_layers + 1, self.num_of_dimensions)) * math.pi

        self.distribution_generator = Distribution(self.num_of_dimensions, type="normal")
        self.distribution = self.distribution_generator.distribution
        self.distribution_dict = self.distribution_generator.distribution_dict

        self.data = np.random.choice(2 ** self.num_of_dimensions, 2000, p=self.distribution)

    # def build_generator_j(self):
    #     q = QuantumRegister(self.g_n_qubits)
    #     c = ClassicalRegister(self.g_n_qubits)
    #     circuit = QuantumCircuit(q, c)
    #
    #     j = 0
    #
    #     # Kelbi method of 2D Array?
    #     for i in range(j, j + self.g_n_qubits):
    #         circuit.ry(self.g_thetas_j[i], q[i])
    #     j += self.g_n_qubits
    #
    #     for iter in range(self.k):
    #         circuit.cz(q[1:], q[:-1])
    #         circuit.cz(q[0], q[-1])
    #
    #         for i in range(j, j + self.g_n_qubits):
    #             circuit.ry(self.g_thetas_j[j+i], q[i])
    #         j += self.g_n_qubits
    #         circuit.barrier()
    #
    #     circuit.measure(q[:], c[:])
    #
    #     plot = circuit.draw(output='mpl')
    #     plot.show()
    #
    #     return circuit

    def build_generator(self):
        q = QuantumRegister(self.num_of_dimensions)
        c = ClassicalRegister(self.num_of_dimensions)
        circuit = QuantumCircuit(q, c)

        for i in range(self.num_of_dimensions):
            circuit.ry(self.g_thetas[0][i], q[i])

        circuit.barrier()

        for iter in range(self.g_layers):
            circuit.cz(q[1:], q[:-1])
            circuit.cz(q[0], q[-1])
            circuit.barrier()
            for i in range(self.num_of_dimensions):
                circuit.ry(self.g_thetas[iter+1][i], q[i])
            circuit.barrier()

        circuit.measure(q[:], c[:])

        # plot = circuit.draw(output='mpl')
        # plot.show()

        return circuit

    def build_discriminator(self, array):
        q = QuantumRegister(self.num_of_dimensions+1)
        c = ClassicalRegister(1)
        circuit = QuantumCircuit(q, c)

        while len(array) != 3:
            array.insert(0, 0)

        for i in range(self.num_of_dimensions):
            circuit.initialize((1-array[i], array[i]), q[i])
            circuit.ry(self.g_thetas[0][i], q[i])

        circuit.barrier()

        for iter in range(self.d_layers):
            circuit.cz(q[1:-1], q[:-2])
            circuit.cz(q[0], q[-2])
            circuit.barrier()
            for i in range(self.num_of_dimensions):
                circuit.ry(self.g_thetas[iter+1][i], q[i])
            circuit.barrier()

        circuit.h(q[self.num_of_dimensions])
        circuit.cnot(q[self.num_of_dimensions], q[:-1])
        circuit.h(q[self.num_of_dimensions])

        circuit.measure(q[self.num_of_dimensions], c[0])

        # plot = circuit.draw(output='mpl')
        # plot.show()

        return circuit

    def generate_point(self):
        g_circuit = self.build_generator()

        backend = Aer.get_backend('qasm_simulator')
        job = execute(backend=backend, experiments=g_circuit, shots=1)
        counts = job.result().get_counts()

        return counts

    @staticmethod
    def get_discriminator_judgement(d_circuit):
        backend = Aer.get_backend('qasm_simulator')
        job = execute(backend=backend, experiments=d_circuit, shots=1)
        counts = job.result().get_counts()
        return counts

    def get_generator_distribution(self):
        shots = 1000
        g_circuit = self.build_generator()

        backend = Aer.get_backend('qasm_simulator')
        job = execute(backend=backend, experiments=g_circuit, shots=shots)
        counts = job.result().get_counts()

        probability_distribution = QGAN4.convert_dict_binary_to_int_and_normalize(counts, shots)

        return probability_distribution

    @staticmethod
    def calculate_hellinger_distance(true_distribution, generated_distribution):
        # Distributions must come in dict form
        result = 0

        # print(true_distribution, generated_distribution)

        for key in true_distribution.keys():
            f1 = true_distribution.get(key)
            # print(key, generated_distribution, generated_distribution.get(key))
            f2 = generated_distribution.get(key)
            if f1 is None:
                f1 = 0
            if f2 is None:
                f2 = 0
            result += (math.sqrt(f1) - math.sqrt(f2)) ** 2

        return (1 / math.sqrt(2)) * math.sqrt(result)

    @staticmethod
    def convert_dict_binary_to_int_and_normalize(dict, shots):
        result_dict = {}
        for key in dict.keys():
            result_dict[str(int(key, 2))] = dict[key] / shots

        return result_dict

    def plot_generator_distribution(self):
        gen_dist = self.get_generator_distribution()
        dist_arr = np.zeros(2 ** self.num_of_dimensions)

        for key in gen_dist.keys():
            dist_arr[int(key)] = gen_dist.get(key)

        plt.bar(np.arange(len(dist_arr)), dist_arr)
        plt.show()

    def plot_real_and_generated_distribution(self):
        gen_dist = self.get_generator_distribution()
        gen_dist_arr = np.zeros(2 ** self.num_of_dimensions)

        for key in gen_dist.keys():
            gen_dist_arr[int(key)] = gen_dist.get(key)

        #plt.plot(np.arange(len(gen_dist_arr)), gen_dist_arr, label='Generated', color='orange')
        plt.bar(np.arange(len(self.distribution)), self.distribution, label='Real', color='blue', width=0.5)
        plt.bar(np.arange(len(gen_dist_arr)), gen_dist_arr, label='Fake', color='red', width=0.2)

        plt.xlabel("x")
        plt.ylabel("p(x)")

        plt.grid(color="grey", linestyle="--", linewidth=0.5, zorder=-1)
        plt.legend()

        plt.show()

    @staticmethod
    def bitfield(n):
        return [int(digit) for digit in bin(n)[2:]]

    def train(self):
        d_loss_on_real = []
        d_loss_on_generated = []
        g_loss = []
        accuracies = []
        h_dists = []

        for e in range(self.epochs):
            d_real_total_cost = 0
            n_correct = 0
            n_incorrect = 0
            # First train the Discriminator on real data
            idx = np.random.randint(len(self.data), size=self.steps_per_epoch)
            xs = self.data[idx]
            for x in xs:
                d_circuit = self.build_discriminator(QGAN4.bitfield(x))
                d_prediction = int(list(QGAN4.get_discriminator_judgement(d_circuit).keys())[0])

                d_cost = 1 - d_prediction

                # print('real', d_cost)
                grad = derivatives(d_cost, self.d_thetas, e)
                # print(grad)

                if d_prediction > 0.5:
                    n_correct += 1
                else:
                    n_incorrect += 1

                # Update the thetas
                self.d_thetas -= self.alpha * grad

                d_real_total_cost += d_cost

            d_loss_on_real.append(d_real_total_cost)

            d_generated_total_cost = 0
            # Then train the Discriminator on generated data
            for i in range(self.steps_per_epoch):
                gen = self.generate_point()
                # Convert binary string to list of binary digits
                bin_gen = []
                for bin in list(gen.keys())[0]:
                    bin_gen.append(int(bin))

                d_circuit = self.build_discriminator(bin_gen)
                d_prediction = int(list(QGAN4.get_discriminator_judgement(d_circuit).keys())[0])

                d_cost = d_prediction

                grad = derivatives(d_cost, self.d_thetas, e)

                if d_prediction < 0.5:
                    n_correct += 1
                else:
                    n_incorrect += 1

                # Update the thetas
                self.d_thetas -= self.alpha * grad

                d_generated_total_cost += d_cost

            d_loss_on_generated.append(d_generated_total_cost)

            g_total_cost = 0
            # # Finally train the Generator on the Discriminator
            for i in range(self.steps_per_epoch):
                gen = self.generate_point()
                # Convert binary string to list of binary digits
                bin_gen = []
                for bin in list(gen.keys())[0]:
                    bin_gen.append(int(bin))

                d_circuit = self.build_discriminator(bin_gen)
                d_prediction = int(list(QGAN4.get_discriminator_judgement(d_circuit).keys())[0])

                g_cost = -(1 - d_prediction)
                grad = derivatives(g_cost, self.g_thetas, e)

                # Update the thetas
                self.g_thetas -= self.alpha * grad

                g_total_cost += g_cost
            g_loss.append(g_total_cost)

            h_dist = QGAN4.calculate_hellinger_distance(self.distribution_dict, self.get_generator_distribution())
            h_dists.append(h_dist)

            print(f"Epoch: {e}, G_loss: {g_total_cost:.3f}, D_loss_real: {d_real_total_cost:.3f}, D_loss_gen: {d_generated_total_cost:.3F}, Accuracy D: {n_correct / (n_incorrect + n_correct)}, Hellinger Distance: {h_dist:.3F}")

            self.plot_real_and_generated_distribution()
        self.plot_hellinger_distance(h_dists)

    @staticmethod
    def plot_hellinger_distance(h_dists):
        plt.title("Hellinger Distance per Epoch")
        plt.xlabel("Hellinger Distance")
        plt.ylabel("Epoch")
        plt.plot(h_dists, label="Hellinger Distance")
        plt.show()


if __name__ == '__main__':
    qgan = QGAN4(num_of_dimensions=3, epochs=100)
    # qgan.build_discriminator([1])
    qgan.train()

    print("Optimized parameters:", qgan.d_thetas, qgan.g_thetas)
    # qgan.plot_real_and_generated_distribution()
    # print(qgan.g_thetas[0])
    # qgan.build_discriminator()
    # print(qgan.distribution_dict)
    # print(QGAN4.convert_dict_binary_to_int_and_normalize(qgan.get_generator_distribution(), 1000))

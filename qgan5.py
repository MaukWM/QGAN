import math

import matplotlib.pyplot as plt
from qiskit.algorithms.optimizers import COBYLA

#from data import OneClusters, TwoClusters

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
import numpy as np

from data import Distribution

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.models import Model

def derivative(f, theta, epoch):
    return 0.5 * (f * (theta + math.pi / (2 * math.sqrt(epoch + 1))) - f * (theta - math.pi / (2 * math.sqrt(epoch + 1))))

derivatives = np.vectorize(derivative)


class QGAN4:

    def __init__(self, num_of_dimensions, epochs=20):
        self.epochs = epochs
        self.steps_per_epoch = 10
        self.alpha = 0.005

        self.disc_batch_size = 64
        self.disc_learning_rate = 0.0015

        optimizer = tf.keras.optimizers.Adam(self.disc_learning_rate)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.num_of_dimensions = num_of_dimensions
        self.g_layers = 2
        self.d_layers = 1

        # self.g_thetas_j = np.random.random_sample((g_n_qubits + self.k * g_n_qubits,)) * math.pi
        self.g_thetas = np.random.random_sample((self.g_layers + 1, self.num_of_dimensions)) * math.pi
        self.d_thetas = np.random.random_sample((self.d_layers + 1, self.num_of_dimensions)) * math.pi

        self.distribution_generator = Distribution(self.num_of_dimensions, type="single_valued")
        self.distribution = self.distribution_generator.distribution
        self.distribution_dict = self.distribution_generator.distribution_dict

        self.data = np.random.choice(2 ** self.num_of_dimensions, 2000, p=self.distribution) / self.distribution_generator.n

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

    def build_discriminator(self):
        model = Sequential()

        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(1,)))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.summary()

        disc_input = Input(1)
        disc_output = model(disc_input)

        return Model(disc_input, disc_output)

    def generate_point(self):
        g_circuit = self.build_generator()

        backend = Aer.get_backend('qasm_simulator')
        job = execute(backend=backend, experiments=g_circuit, shots=1)
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

    def get_discriminator_distribution(self):
        dist = np.zeros(self.distribution_generator.n)
        for i in range(self.distribution_generator.n):
            dist[i] = (self.discriminator.predict([i / self.distribution_generator.n]) - 0.5)
        return dist / np.linalg.norm(dist)

    def plot_real_and_generated_distribution(self, e=-1, plot_disc_dist=False):
        gen_dist = self.get_generator_distribution()
        gen_dist_arr = np.zeros(2 ** self.num_of_dimensions)

        for key in gen_dist.keys():
            gen_dist_arr[int(key)] = gen_dist.get(key)

        #plt.plot(np.arange(len(gen_dist_arr)), gen_dist_arr, label='Generated', color='orange')
        plt.bar(np.arange(len(self.distribution)), self.distribution, label='Real', color='blue', width=0.5)
        plt.bar(np.arange(len(gen_dist_arr)), gen_dist_arr, label='Fake', color='red', width=0.2)

        if plot_disc_dist:
            disc_dist = self.get_discriminator_distribution()
            plt.bar(np.arange(len(gen_dist_arr)), disc_dist, label='Disc_probs', color='green', width=0.1)

        plt.xlabel("x")
        plt.ylabel("p(x)")

        plt.grid(color="grey", linestyle="--", linewidth=0.5, zorder=-1)
        plt.legend()

        if e > -1:
            plt.title("Epoch " + str(e))

        plt.show()

    @staticmethod
    def bitfield(n):
        return [int(digit) for digit in bin(n)[2:]]

    def objective_function_generator(self, g_thetas):
        self.g_thetas = g_thetas.reshape((self.g_thetas.shape[0], self.g_thetas.shape[1]))
        cost = 0
        for i in range(13):
            generated_sample = int(list(self.generate_point().keys())[0], 2)
            # generated_sample = generated_sample / self.distribution_generator.n
            d_prediction = self.discriminator.predict([generated_sample])

            # cost += abs(0.75 - gen_x) + abs(0.25 - gen_y)

            if d_prediction < 0.5:
                cost += 1
            # if generated_sample != 0:
            #     cost += 1
            # cost += ((1 - d_prediction) * 2) ** 2

        return cost

    def train(self):
        d_loss_on_real = []
        d_loss_on_generated = []
        g_loss = []
        accuracies = []
        h_dists = []

        # Adversarial ground truths
        valid = np.ones((self.disc_batch_size, 1))
        fake = np.zeros((self.disc_batch_size, 1))

        asd = 0

        for e in range(self.epochs):
            d_real_total_cost = 0
            n_correct = 0
            n_incorrect = 0

            # First train the Discriminator on real data
            # Select a random batch of real data points
            idx = np.random.randint(0, self.data.shape[0], self.disc_batch_size)
            real_data_points = self.data[idx]

            generated_data_points = []
            for i in range(self.disc_batch_size):
                generated_data_points.append(int(list(self.generate_point().keys())[0], 2) / self.distribution_generator.n)
            generated_data_points = np.stack(generated_data_points)

            # Train the discriminator
            for i in range(20):
                d_loss_real = self.discriminator.train_on_batch(real_data_points, valid)
                d_loss_fake = self.discriminator.train_on_batch(generated_data_points, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                d_loss_on_real.append(d_loss_real)
                d_loss_on_generated.append(d_loss_fake)

            g_total_cost = 0

            g_dist = self.get_generator_distribution()

            # print(g_dist)
            # Get probs from discriminator 0 through 7
            # apply loss function in discord

            # # Finally train the Generator on the Discriminator
            # for i in range(self.steps_per_epoch):
            #     # Convert binary string to list of binary digits
            #     generated_sample = int(list(self.generate_point().keys())[0], 2)
            #     d_prediction = self.discriminator.predict([generated_sample])
            #
            #     g_cost = -(1 - d_prediction)
            #
            #     g_total_cost += g_cost
            #
            # grad = derivatives(g_total_cost, self.g_thetas, e)
            #
            # # Update the thetas
            # self.g_thetas -= self.alpha * grad
            #
            # g_loss.append(g_total_cost)

            optimizer = COBYLA(maxiter=10, tol=0.001)

            opt_param = optimizer.optimize(num_vars=self.g_thetas.size,
                                           objective_function=self.objective_function_generator,
                                           initial_point=self.g_thetas.flatten())

            self.g_thetas, g_cost, _ = opt_param
            self.g_thetas = self.g_thetas.reshape((self.g_layers + 1, self.num_of_dimensions))

            h_dist = QGAN4.calculate_hellinger_distance(self.distribution_dict, self.get_generator_distribution())
            h_dists.append(h_dist)

            acc = 100*d_loss[1]
            # print(e, acc, g_total_cost, d_loss_real, d_loss_fake, h_dist)
            print(f"Epoch: {e}, G_loss: {g_cost:.3f}, D_loss_real: {d_loss_real[0]:.3f}, D_loss_gen: {d_loss_fake[0]:.3F}, Accuracy D: {100 * d_loss[1]}%, Hellinger Distance: {h_dist:.3F}")

            # if asd % 25 == 0:
            self.plot_real_and_generated_distribution(e, plot_disc_dist=True)
            g_loss.append(g_cost)
            accuracies.append(acc)
            # asd += 1
        self.plot_hellinger_distance(h_dists)

        plt.plot(accuracies, label='acc')
        plt.legend()
        plt.show()

        plt.plot(g_loss, label='g_loss')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_hellinger_distance(h_dists):
        plt.title("Hellinger Distance per Epoch")
        plt.xlabel("Hellinger Distance")
        plt.ylabel("Epoch")
        plt.plot(h_dists, label="Hellinger Distance")
        plt.savefig("plots/h_dists.png")
        plt.show()


if __name__ == '__main__':
    qgan = QGAN4(num_of_dimensions=3, epochs=150)
    qgan.build_generator().draw(output='mpl').show()
    # qgan.build_discriminator([1])
    qgan.train()

    print("Optimized parameters:", qgan.d_thetas, qgan.g_thetas)
    # qgan.plot_real_and_generated_distribution()
    # print(qgan.g_thetas[0])
    # qgan.build_discriminator()
    # print(qgan.distribution_dict)
    # print(QGAN4.convert_dict_binary_to_int_and_normalize(qgan.get_generator_distribution(), 1000))

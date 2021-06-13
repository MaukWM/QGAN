import copy
import math

import matplotlib.pyplot as plt

from data import OneClusters

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

        self.opt_d_thetas = copy.copy(self.d_thetas)
        self.opt_g_thetas = copy.copy(self.g_thetas)
        self.opt_h_dist = 1

        self.epochs = epochs

        self.alpha = 0.01

        self.steps_per_epoch = 20

        self.resolution = 0.05

        self.true_distribution = self.calculate_true_distribution()

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

    def calculate_true_distribution(self):
        # Initialize values in order to calculate distribution
        # Resolution must divide 1
        xs = np.arange(0, 1, self.resolution)
        ys = np.arange(0, 1, self.resolution)

        total_data_points = len(self.data)

        probabilities = np.zeros(shape=(int(1 / self.resolution), int(1 / self.resolution)))

        #     print(data.shape)
        #     print(data)

        for x_i in range(len(xs)):
            for y_i in range(len(ys)):
                filtered_x = self.data[((self.resolution * (x_i + 1) > self.data[:, 0]) & (self.data[:, 0] > self.resolution * x_i))]
                filtered_y = filtered_x[
                    ((self.resolution * (y_i + 1) > filtered_x[:, 1]) & (filtered_x[:, 1] > self.resolution * y_i))]
                probability = len(filtered_y) / total_data_points
                probabilities[x_i][y_i] = probability

        # Transpose the probabilities so that the results represent the data set.
        probabilities = probabilities.T

        return probabilities

    def calculate_generated_distribution(self):
        g_circuit = self.build_generator()
        # Initialize values in order to calculate distribution
        # Resolution must divide 1
        xs = np.arange(0, 1, self.resolution)
        ys = np.arange(0, 1, self.resolution)

        total_data_points = 250

        probabilities = np.zeros(shape=(int(1 / self.resolution), int(1 / self.resolution)))

        def clamp(n, minn, maxn):
            return max(min(maxn, n), minn)

        for i in range(total_data_points):
            x, y = self.generate_point()
            probabilities[clamp(int(x / self.resolution), 0, (int(1/self.resolution) - 1))][clamp(int(y / self.resolution), 0, (int(1/self.resolution) - 1))] += 1 / total_data_points

        # Transpose the probabilities so that the results represent the data set.
        probabilities = probabilities.T

        return probabilities

    def calculate_hellinger_distance(self, true_distribution, generated_distribution):
        result = 0

        for x_i in range(len(true_distribution)):
            for y_i in range(len(true_distribution[x_i])):
                result += (math.sqrt(true_distribution[x_i][y_i]) - math.sqrt(generated_distribution[x_i][y_i])) ** 2

        return (1 / math.sqrt(2)) * math.sqrt(result)

    def train(self):
        opt_d_thetas, opt_g_thetas, opt_h_dist = self.d_thetas, self.g_thetas, 1
        # Train for e amount of epochs

        d_loss_on_real = []
        d_loss_on_generated = []
        g_loss = []
        accuracies = []

        asd = 0

        for e in range(self.epochs):
            asd+=1
            # TODO: Print loss D and G and accuracy of D
            # TODO: Add losses of D and G to list in order to plot

            d_real_total_cost = 0
            n_correct = 0
            n_incorrect = 0

            # First train the Discriminator on real data
            idx = np.random.randint(len(self.data), size=self.steps_per_epoch)
            xs = self.data[idx]
            for x in xs:
                d_circuit = self.build_discriminator(x[0], x[1])
                d_prediction = QGAN2.run_simulation(d_circuit, 30)
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
                (gen_x, gen_y) = self.generate_point()
                d_circuit = self.build_discriminator(gen_x, gen_y)
                d_prediction = QGAN2.run_simulation(d_circuit, 30)

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
                (gen_x, gen_y) = self.generate_point()
                d_circuit = self.build_discriminator(gen_x, gen_y)
                d_prediction = QGAN2.run_simulation(d_circuit, 30)
                # print('gen', g_cost)
                g_cost = -(1 - d_prediction)
                grad = derivatives(g_cost, self.g_thetas, e)

                # Update the thetas
                self.g_thetas -= self.alpha * grad

                g_total_cost += g_cost
            g_loss.append(g_total_cost)

            h_dist = self.calculate_hellinger_distance(self.true_distribution, self.calculate_generated_distribution())

            if h_dist < opt_h_dist:
                self.opt_d_thetas, self.opt_g_thetas, self.opt_h_dist = copy.copy(self.d_thetas), copy.copy(self.g_thetas), copy.copy(h_dist)
                print("Updating optimal g_thetas to:", opt_g_thetas)

            if asd % 5 == 0:
                self.plot_generator_distribution(epoch=e)
                # self.generate_discriminator_heatmap()

            accuracies.append(n_correct/(n_incorrect + n_correct))

            print(f"Epoch: {e}, G_loss: {g_total_cost:.3f}, D_loss_real: {d_real_total_cost:.3f}, D_loss_gen: {d_generated_total_cost:.3F}, Accuracy D: {n_correct / (n_incorrect + n_correct)}, Hellinger Distance: {h_dist:.3F}")

        # plt.ylim(0)
        print(d_loss_on_real)
        print(d_loss_on_generated)
        print(g_loss)
        plt.plot(d_loss_on_real, label='d_loss_on_real', color='blue')
        plt.plot(d_loss_on_generated, label='d_loss_on_generated', color='green')
        plt.plot(g_loss, label='g_loss', color='orange')
        plt.legend()
        plt.show()

        plt.plot(accuracies, label='accuracy discriminator')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()

        return g_loss, d_loss_on_real, d_loss_on_generated

    def generate_discriminator_heatmap(self):
        xs = np.arange(0, 1, 0.05)
        ys = np.arange(0, 1, 0.05)

        heatmap = np.zeros(shape=(20, 20))

        for x_i in range(len(xs)):
            for y_i in range(len(ys)):
                d_circuit = self.build_discriminator(xs[x_i], ys[y_i])
                prob = self.run_simulation(d_circuit, 30)
                heatmap[x_i][y_i] = prob

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

from data import OneClusters


class QGAN:

    def __init__(self):
        self.data_generator = OneClusters()

        self.q_circuit = self.build_circuit()

    def build_full_circuit(self):
        pass

    def build_generator_circuit(self):
        pass

    def build_discriminator(self):
        pass

    def train(self):
        pass


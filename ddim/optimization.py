import tensorflow as tf
import os

class Optimizer(object):
    def __init__(self, lr=8.75e-4):
        self.lr = lr
        self.lr_func = lambda x: x # a crude learning rate scheduler thingy
        self.ema = {}

    def update_variable(self, gradient: tf.Tensor, variable: tf.Variable): # subclasses will implement
        pass

    def apply_gradients(self, grads_and_vars: zip):
        for grad, var in grads_and_vars:
            if grad is not None:
                if isinstance(grad, tf.IndexedSlices):
                    grad = tf.convert_to_tensor(grad)
                self.update_variable(grad, var)

# adam optimizer
class Adam(Optimizer):
    def __init__(self, lr=8.75e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        super().__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.moment1, self.moment2, self.updates = {}, {}, {}

    # a quick simple implementation of the adam algorithm, don't need any advanced features from the keras optimizers
    def update_variable(self, gradient: tf.Tensor, variable: tf.Variable):
        if variable.name not in self.updates:
            self.moment1[variable.name] = tf.zeros_like(variable)
            self.moment2[variable.name] = tf.zeros_like(variable)
            self.updates[variable.name] = 0
        self.moment1[variable.name] = self.beta_1 * self.moment1[variable.name] + gradient * (1.0 - self.beta_1)
        self.moment2[variable.name] = self.beta_2 * self.moment2[variable.name] + tf.square(gradient) * (1.0 - self.beta_2)    
        corrected_moment1 = self.moment1[variable.name] / (1.0 - self.beta_1 ** (self.updates[variable.name] + 1))
        corrected_moment2 = self.moment2[variable.name] / (1.0 - self.beta_2 ** (self.updates[variable.name] + 1))
        variable.assign_sub(self.lr * corrected_moment1 / (tf.sqrt(corrected_moment2) + self.epsilon))
        self.updates[variable.name] += 1

# sgd with momentum
class Sgd(Optimizer):
    def __init__(self, lr=8.75e-4, momentum=0.9, nesterov=False):
        super().__init__(lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = {}
    # a minimal implementation of the sgd algorithm
    def update_variable(self, gradient: tf.Tensor, variable: tf.Variable):
        if variable.name not in self.velocities:
            self.velocities[variable.name] = tf.zeros_like(variable)
        if self.momentum > 0.0:
            self.velocities[variable.name] = self.velocities[variable.name] * self.momentum - gradient * self.lr
            if self.nesterov:
                update = self.velocities[variable.name] * self.momentum - gradient * self.lr
                variable.assign_add(update)
            else:
                variable.assign_add(self.velocities[variable.name])
        else:
            # this boils down into regular sgd
            variable.assign_sub(gradient*self.lr)

class SGLD(Optimizer): # stochastic gradient descent with langevin dynamics
    def __init__(self, lr=8.75e-4, momentum=0.9, noise_factor=0.01, temp=1.0, nesterov=False):
        super().__init__(lr)
        self.momentum = momentum
        self.noise_factor = noise_factor
        self.temp = temp
        self.nesterov = nesterov
        self.velocities = {}

    def update_variable(self, gradient: tf.Tensor, variable: tf.Variable):
        if variable.name not in self.velocities:
            self.velocities[variable.name] = tf.zeros_like(variable)
        if self.momentum > 0.0:
            self.velocities[variable.name] = self.velocities[variable.name] * self.momentum - gradient * self.lr
            if self.nesterov:
                update = self.velocities[variable.name] * self.momentum - gradient * self.lr
            else:
                update = self.velocities[variable.name]
        else:
            update = -gradient * self.lr
        noise = tf.random.normal(variable.shape, mean=0.0, stddev=tf.sqrt(self.lr * self.temp))
        langevin_update = update/2.0 + self.noise_factor * noise
        variable.assign_add(langevin_update)

# rmsprop algorithm
class RMSProp(Optimizer):
    def __init__(self, lr=8.75e-4, rho=0.9, epsilon=1.0e-7):
        super().__init__(lr)
        self.rho = rho
        self.epsilon = epsilon
        self.grad_sq_avg = {}
    # a minimal implementation of the rmsprop algorithm
    def update_variable(self, gradient: tf.Tensor, variable: tf.Variable):
        if variable.name not in self.grad_sq_avg:
            self.grad_sq_avg[variable.name] = tf.zeros_like(variable)
        self.grad_sq_avg[variable.name] = self.grad_sq_avg[variable.name] * self.rho + tf.square(gradient) * (1.0 - self.rho)
        step = gradient * self.lr / tf.sqrt(self.grad_sq_avg[variable.name] + self.epsilon)
        variable.assign_sub(step)
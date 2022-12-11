import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class VAE(keras.Model):
    def __init__(self, encoder, decoder, kl_beta = 1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.kl_beta_tracker = keras.metrics.Mean(name="kl_beta")

        self.kl_beta = kl_beta

    def call(self, input):

        mean_log_var = self.encoder(input)
        z_mean, z_log_var = tf.split(mean_log_var, 2, axis=-1)
        z = self.sample_z(z_mean, z_log_var)
        reconstruction = self.decoder(z)

        return reconstruction

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.kl_beta_tracker,

        ]

    def sample_z(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


    def train_step(self, data):
        with tf.GradientTape() as tape:

            mean_log_var = self.encoder(data)
            z_mean, z_log_var = tf.split(mean_log_var, 2, axis=-1)
            z = self.sample_z(z_mean, z_log_var)
            
            reconstruction = self.decoder(z)
            reconstruction = tf.squeeze(reconstruction)

            reconstruction_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data, reconstruction)
            )
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

            kl_loss = tf.reduce_mean(kl_loss)

            kl_loss = kl_loss

            total_loss = reconstruction_loss + self.kl_beta*kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_beta_tracker.update_state(self.kl_beta)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            'kl_beta': self.kl_beta_tracker.result(),
        }

class KL_Callback(tf.keras.callbacks.Callback):
        def __init__(self, kl_beta, minimum, maximum, interval) -> None:
            super().__init__()

            self.min = minimum
            self.max = maximum
            self.interval = interval

            self.kl_beta = kl_beta

        def on_epoch_begin(self, epoch, logs=None):
            if epoch == 0:
                self.kl_beta.assign(self.min)

            else:
                if self.kl_beta  >= self.max:
                    self.kl_beta.assign(self.min)
                else:
                    self.kl_beta.assign(self.kl_beta + (self.max-self.min)/self.interval)

            print(f'kl_beta: {float(self.kl_beta.numpy()):.05f}')
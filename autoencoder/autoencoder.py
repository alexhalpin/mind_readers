import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

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

            print(kl_loss.shape)
            kl_loss = tf.reduce_mean(kl_loss)

            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class KL_Callback(tf.keras.callbacks.Callback):
        def __init__(self, kl_beta=None) -> None:
            super().__init__()

            if kl_beta is None:
                self.kl_beta = tf.Variable(1.0, trainable=False)
            else:
                self.kl_beta = kl_beta

        def on_epoch_begin(self, epoch, logs=None):
            print(f'kl_beta: {self.kl_beta}')
            if epoch == 1:
                self.kl_beta.assign(1.0)
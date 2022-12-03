import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

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

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)

            reconstruction = self.decoder(z)
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction)
                )
            )
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

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

class Encoder(keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.name = "encoder"
        self.latent_dim = latent_dim
        
        self.conv1 = layers.Conv2D(64, kernel_size=4, strides=2, activation="relu", padding="same")
        self.conv2 = layers.Conv2D(128, kernel_size=4, strides=2, activation="relu", padding="same")
        self.conv3 = layers.Conv2D(512, kernel_size=4, strides=2, activation="relu", padding="same")
        self.flatten = layers.Flatten()

        self.dense_mean = layers.Dense(latent_dim, name="z_mean")
        self.dense_log_var = layers.Dense(latent_dim, name="z_log_var")
        self.sampling = Sampling()

    def __call__(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        z_mean = self.dense_mean(x)
        z_log_var = self.dense_mean(x)
        z = self.sampling([z_mean, z_log_var])

        return [z_mean, z_log_var, z]

class Decoder(keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.name = "decoder"
        self.latent_dim = latent_dim

        self.dense = layers.Dense((4* 4* 512), activation="relu")
        self.reshape = layers.Reshape((4, 4, 512))
        self.conv1 = layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, activation="relu", padding='same')
        self.conv2 = layers.Conv2DTranspose(filters=64 , kernel_size=4, strides=2, activation="relu", padding='same')
        self.conv3 = layers.Conv2DTranspose(filters=3  , kernel_size=4, strides=2, activation="relu", padding='same')
        self.conv4 = layers.Conv2DTranspose(filters=3  , kernel_size=3, activation="sigmoid", padding="same")

    def __call__(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x

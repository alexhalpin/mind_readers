import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square

class CVAE(tf.keras.Model):

    def __init__(self, encoder, decoder, latent_dim, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.kld_tracker = tf.keras.metrics.Mean(name="kld") 
        self.rec_tracker = tf.keras.metrics.Mean(name="rec") 

        self.encoder = encoder
        self.decoder = decoder

    def call(self, data):
      image, label = data
      #label = tf.one_hot(label,10)
      z_param = self.encoder(image)
      mean, logvar, z = self.latent_ops(z_param)
      cond_z = tf.concat([z, label], axis=1)
      recon = self.decoder(cond_z) #Change if conditioning
      return recon, mean, logvar, z    


    def latent_ops(self, logits):
        mean, logvar = tf.split(logits, num_or_size_splits=2, axis=1)
        z= self.sample_z(mean,logvar) 
        return mean, logvar, z
    
    def sample_z(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean  
 
 
    def compile(self, *args, **kwargs ): 
        super().compile( *args, )
        self.kld_loss = kwargs['kld_loss']
        self.rec_loss = kwargs['rec_loss'] 


    def train_step(self, data): return self.batch_step(data, training=True)
    def test_step(self, data):  return self.batch_step(data, training=False)

    def batch_step(self, data, training=True):
        image, label = data

        with tf.GradientTape() as tape:
          output, mean, logvar, z = self(image)
          rec_loss = self.rec_loss(label, output)
          kld_loss = self.kld_loss(logvar, mean)
          loss = rec_loss + kld_loss

        if training:
          gradients = tape.gradient(loss, self.trainable_variables)
          self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.kld_tracker.update_state(kld_loss)
        self.rec_tracker.update_state(rec_loss)
    
        return {"kld_loss": self.kld_tracker.result(), "recon_loss": self.rec_tracker.result()}

# ml_generator.py (v1.5 - Moderni Keras API pristup)

import numpy as np
import pandas as pd
import sqlite3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MultiLabelBinarizer
import os
import time

# Konstante
MODEL_PATH = 'loto_decoder_model.keras'
LATENT_DIM = 4 
BROJ_BROJEVA = 39 

def pripremi_podatke():
    print("ML Motor: Pripremam podatke za trening...")
    if not os.path.exists('loto_baza.db'):
        return None, "Baza podataka 'loto_baza.db' nije pronađena."
    try:
        conn = sqlite3.connect('loto_baza.db')
        df = pd.read_sql_query("SELECT b1, b2, b3, b4, b5, b6, b7 FROM istorijski_rezultati", conn)
        conn.close()
        kombinacije = df.values.tolist()
        mlb = MultiLabelBinarizer(classes=range(1, BROJ_BROJEVA + 1))
        podaci_za_trening = mlb.fit_transform(kombinacije)
        print(f"ML Motor: Podaci uspešno pripremljeni. Ukupno uzoraka: {len(podaci_za_trening)}")
        return np.array(podaci_za_trening, dtype='float32'), None
    except Exception as e:
        print(f"ML Motor: Greška pri pripremi podataka - {e}")
        return None, str(e)

# --- NOVA, KOMPLETNA KLASA ZA VAE MODEL ---
class VAE(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        original_dim = BROJ_BROJEVA
        intermediate_dim = 64
        
        # Enkoder
        encoder_inputs = keras.Input(shape=(original_dim,))
        x = layers.Dense(intermediate_dim, activation="relu")(encoder_inputs)
        x = layers.Dense(intermediate_dim // 2, activation="relu")(x)
        z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
        z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
        
        # Reparametrization Trick
        def sampling(args):
            z_mean_samp, z_log_var_samp = args
            batch = tf.shape(z_mean_samp)[0]
            dim = tf.shape(z_mean_samp)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean_samp + tf.exp(0.5 * z_log_var_samp) * epsilon
        
        z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # Dekoder
        latent_inputs = keras.Input(shape=(LATENT_DIM,))
        x = layers.Dense(intermediate_dim // 2, activation="relu")(latent_inputs)
        x = layers.Dense(intermediate_dim, activation="relu")(x)
        decoder_outputs = layers.Dense(original_dim, activation="sigmoid")(x)
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=-1
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
        
        return {m.name: m.result() for m in self.metrics}
    
    def call(self, inputs):
        # Definišemo kako se ceo model ponaša pri pozivu
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

def treniraj_i_sacuvaj_model():
    """Pokreće ceo proces treninga i čuva model."""
    podaci, greska = pripremi_podatke()
    if greska: return f"Neuspeh: {greska}"
    
    print("ML Motor: Gradim i kompajliram VAE model...")
    vae = VAE()
    vae.compile(optimizer=keras.optimizers.Adam())
    
    print("ML Motor: Počinjem trening... Ovo može potrajati nekoliko minuta.")
    try:
        start_time = time.time()
        # Trening se sada poziva na 'podaci', bez drugog argumenta jer se loss računa unutar modela
        vae.fit(podaci, epochs=100, batch_size=32, shuffle=True, verbose=1)
        end_time = time.time()
        
        # Čuvamo samo dekoder jer nam je on potreban za generisanje
        vae.decoder.save(MODEL_PATH)
        
        vreme_treninga = end_time - start_time
        print(f"ML Motor: Trening završen za {vreme_treninga:.2f} sekundi. Dekoder sačuvan u '{MODEL_PATH}'.")
        return f"Trening uspešno završen za {vreme_treninga:.2f}s! Model je sačuvan."
    except Exception as e:
        return f"Greška tokom treninga: {e}"

def generisi_kombinacije(broj_predloga=10):
    """Učitava sačuvani dekoder i generiše nove kombinacije."""
    if not os.path.exists(MODEL_PATH):
        return [f"GREŠKA: Model '{MODEL_PATH}' nije pronađen. Molim Vas, prvo istrenirajte model."], True
    try:
        decoder = keras.models.load_model(MODEL_PATH)
        print(f"ML Motor: Generišem {broj_predloga} novih predloga...")
        
        random_latent_vectors = np.random.normal(size=(broj_predloga * 3, LATENT_DIM))
        generisani_vektori = decoder.predict(random_latent_vectors, verbose=0)
        
        predlozi = set()
        for vektor in generisani_vektori:
            indeksi_brojeva = np.argsort(vektor)[-7:] + 1
            predlozi.add(tuple(sorted(indeksi_brojeva)))
        
        finalni_predlozi = [tuple(int(x) for x in k) for k in sorted(list(predlozi))[:broj_predloga]]
        print(f"ML Motor: Generisano {len(finalni_predlozi)} jedinstvenih predloga.")
        return finalni_predlozi, False
    except Exception as e:
        return [f"GREŠKA pri generisanju: {e}"], True
import pickle
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

'''
Modelo de VAE com modificações pessoais, baseado nas fontes:

https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/
https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py

'''


# Dimensão da camada latente (médias e variâncias) e shape das imagens ===========================================
latent_dim = 40 
shape_img = (64, 64, 3)


# Função custo de reconstrução da entrada ========================================================================
def nll(y_true, y_pred):
    '''
    negative log likelihood
    '''
    return K.sum(keras.losses.binary_crossentropy(y_true, y_pred), axis=(1,2))


# Camada de amostragem e integração do custo KLDivergence ========================================================
class Sampling(layers.Layer):
    
    def __init__(self, *args, **kwargs):
        super(Sampling, self).__init__(*args, *kwargs)
        
        self.is_placeholder = True
        
        
    def call(self, inputs):
        
        mean, log_var = inputs
        
        # Maneira para adicionar custo KL
        kl_batch = -0.5*K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
        
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        
        # Processo de amostragem
        batch = K.shape(mean)[0]
        dim = K.shape(mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        
        return mean + epsilon*K.exp(0.5*log_var)


# Rede encoder: ==================================================================================================

encoder_inputs = keras.Input(shape=shape_img)

x = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(encoder_inputs)

x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)

x = layers.Flatten()(x)

x = layers.Dense(64, activation='relu')(x)

z_mean = layers.Dense(latent_dim, activation='linear', name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, activation='linear', name='z_log_var')(x)

z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var, z], name='Encoder')

print(encoder.summary())


# Rede decoder: ==================================================================================================

latent_inputs = keras.Input(shape=(latent_dim,))

x = layers.Dense(16*16*128, activation='relu')(latent_inputs)

x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(x)

x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)

decoder_output = layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)

decoder = keras.Model(inputs=latent_inputs, outputs=decoder_output, name='Decoder')

print(decoder.summary())


# Variational Auto Encoder: ======================================================================================

vae_input = keras.Input(shape=shape_img)
_, _, samples = encoder(vae_input)
vae_output = decoder(samples)

vae = keras.Model(inputs = vae_input, outputs = vae_output, name='VAE')

print(vae.summary())


# Carregando, normalizando as imagens e dividindo em treino e teste: =============================================
data = np.load('cats_rgb.npy')

data = data.astype('float32')/data.max()

X_train, X_test, y_train, y_test = train_test_split(data, data, test_size=0.2, random_state=42)

plt.figure()
plt.subplot(121)
plt.imshow(X_train[3])
plt.subplot(122)
plt.imshow(y_train[3])
plt.show()


# Compilando e treinando o VAE =======================================================================================

vae.compile(optimizer='rmsprop', loss=nll)

vae.fit(x=X_train, y=y_train, batch_size=64, epochs=20, validation_data=(X_test, y_test))

# Testando alguns resultados =========================================================================================

preds = vae.predict(X_test[:10])

plt.figure()
plt.subplot(221)
plt.imshow(X_test[5])
plt.subplot(222)
plt.imshow(preds[5])
plt.subplot(223)
plt.imshow(X_test[8])
plt.subplot(224)
plt.imshow(preds[8])
plt.show()

# Aplicando PCA: =================================================================================

pca = PCA()

means, _, _ = encoder.predict(data)

pca.fit(means)

# Teste rápido PCA: ============================================================================
# Provável que a direção de maior variação presente entre as imagens seja a cor da pelagem,
# já que vemos que o primeiro componente modifica a cor da pelagem do gato (branco <-> escuro).

mean_test0 = 3*np.ones(40)
mean_test0[1:] = 0

img_test0 = decoder.predict(np.expand_dims(pca.inverse_transform(mean_test0), 0))

mean_test1 = -3*np.ones(40)
mean_test1[1:] = 0

img_test1 = decoder.predict(np.expand_dims(pca.inverse_transform(mean_test1), 0))

plt.subplot(121)
plt.imshow(img_test0[0])
plt.subplot(122)
plt.imshow(img_test1[0])

# Salvando a rede decoder e PCA ==================================================================

decoder.save('decoder_model.h5')
pickle.dump(pca, open('pca.pkl', 'wb'))
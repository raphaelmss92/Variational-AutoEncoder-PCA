# Variational-AutoEncoder-PCA
Aplicação da rede VAE + PCA para geração de faces de gatos

 - Dados: 15.7k imagens de faces de gatos de dimensões 64x64


 - Fonte dos dados: https://www.kaggle.com/spandan2/cats-faces-64x64-for-generative-models


 - Fontes do algoritmo e entendimento sobre redes VAE's:

    https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73

    https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-autoencoder-with-keras/

    https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py


 - Arquivos do repositório:
    
    cats_rgb.npy: Arquivo numpy de transformação das imagens carregadas em arrays. Mais rápido e fácil para carregar.
    
    decoder_model.h5: Rede decoder treinada salva. Modelo construido no tensorflow-keras.
    
    pca.pkl: Arquivo PCA do sklearn já adaptado às médias de todas imagens passadas pela rede encoder.
    
    training.py: Código de treinamento da rede VAE e adaptação do PCA.
    
    testing.py: Construção da GUI para exibição dos efeitos das 10 primeiras componentes principais na formação das faces.
    

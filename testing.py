import pickle
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

import PySimpleGUI as sg

decoder = load_model('decoder_model.h5')
pca = pickle.load(open('pca.pkl','rb'))

# Iniciando GUI
# Coluna com as barras deslizantes do PCA

sg.theme('Dark')

sliders_column = [
    [sg.Text("Componentes Principais")],
    [sg.Slider(range=(-3.5, 3.5), default_value=0, resolution=0.25, orientation='h', size=(30, 10), key="PC1")],
    [sg.Slider(range=(-3.5, 3.5), default_value=0, resolution=0.25, orientation='h', size=(30, 10), key="PC2")],
    [sg.Slider(range=(-3.5, 3.5), default_value=0, resolution=0.25, orientation='h', size=(30, 10), key="PC3")],
    [sg.Slider(range=(-3.5, 3.5), default_value=0, resolution=0.25, orientation='h', size=(30, 10), key="PC4")],
    [sg.Slider(range=(-3.5, 3.5), default_value=0, resolution=0.25, orientation='h', size=(30, 10), key="PC5")],
    [sg.Slider(range=(-3.5, 3.5), default_value=0, resolution=0.25, orientation='h', size=(30, 10), key="PC6")],
    [sg.Slider(range=(-3.5, 3.5), default_value=0, resolution=0.25, orientation='h', size=(30, 10), key="PC7")],
    [sg.Slider(range=(-3.5, 3.5), default_value=0, resolution=0.25, orientation='h', size=(30, 10), key="PC8")],
    [sg.Slider(range=(-3.5, 3.5), default_value=0, resolution=0.25, orientation='h', size=(30, 10), key="PC9")],
    [sg.Slider(range=(-3.5, 3.5), default_value=0, resolution=0.25, orientation='h', size=(30, 10), key="PC10")]
]

# Coluna de exibição da imagem
image_column = [
    [sg.Text(text="Saída da VAE")],
    [sg.Image(key="IMAGE")]
]

# Layout completo
layout = [
    [
    sg.Column(sliders_column),
    sg.VSeparator(),
    sg.Column(image_column)
    ]
]

# Definindo janela da GUI
window = sg.Window("VAE - Cats", layout)

means_input = np.zeros(40)
last_values = None

# Iniciando loop de leitura da janela
while True:
    
    event, values = window.read(timeout=20)

    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    
    if values != last_values:
            
        means_input[:10] = np.array([values[f"PC{i}"] for i in range(1, 11)])
        
        img = decoder.predict(np.expand_dims(pca.inverse_transform(means_input), 0))[0]
        
        img = Image.fromarray((img*255).astype(np.uint8))
        
        img = img.resize((128, 128))
        
        img = ImageTk.PhotoImage(img)
        
        window["IMAGE"].update(data=img)

        last_values = values

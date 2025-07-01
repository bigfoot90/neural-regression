import math
import random
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim

save_path = 'weights.ai'

def normalize_input(degrees):
    # Converte i gradi a valori normalizzati tra -1 e 1
    return degrees / 180 - 1


"""
Detect hardware accelerator
"""
device = torch.device("cpu")
#device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
print(f"Using {device} acceleration")

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, layer_configs):
        """
        Inizializza una rete neurale flessibile.
        
        Args:
            input_dim (int): Dimensione dell'input
            layer_configs (list): Lista di tuple (hidden_dim, activation_function)
                                 dove activation_function può essere una stringa o una funzione
            output_dim (int): Dimensione dell'output
        
        Esempio:
            layer_configs = [
                (128, 'relu'),
                (64, 'tanh'),
                (32, torch.sigmoid),
                (16, 'leaky_relu')
            ]
        """
        super(NeuralNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        self.activations = []
        
        # Dizionario delle funzioni di attivazione disponibili
        self.activation_functions = {
            'relu': torch.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': lambda x: torch.nn.functional.leaky_relu(x),
            'elu': torch.nn.functional.elu,
            'gelu': torch.nn.functional.gelu,
            'swish': lambda x: x * torch.sigmoid(x),
            'linear': lambda x: x,  # Nessuna attivazione
            None: lambda x: x       # Nessuna attivazione
        }
        
        # Costruzione dei layer
        prev_dim = input_dim
        
        for hidden_dim, activation in layer_configs:
            # Aggiungi il layer lineare
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Gestisci la funzione di attivazione
            if isinstance(activation, str):
                if activation.lower() in self.activation_functions:
                    self.activations.append(self.activation_functions[activation.lower()])
                else:
                    raise ValueError(f"Funzione di attivazione '{activation}' non riconosciuta")
            elif callable(activation):
                self.activations.append(activation)
            else:
                self.activations.append(self.activation_functions[None])
            
            prev_dim = hidden_dim
        
        # Inizializzazione opzionale dei pesi (commentata come nell'originale)
        # self._initialize_weights()
    
    def _initialize_weights(self):
        """Inizializza i pesi usando Kaiming uniform initialization"""
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight)
        nn.init.kaiming_uniform_(self.output_layer.weight)
    
    def forward(self, x):
        # Aggiungi dimensione se necessario (come nell'originale)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        # Passa attraverso tutti i layer nascosti
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            x = activation(x)
        
        return x

    def get_activations(self, x):
        activations = []
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            x = layer(x)
            x = activation(x)
            activations.append(x.detach().numpy()) # Salva le attivazioni
        return activations

"""
Create Neural Network
"""
input_dim = 1
layer_configs = [
    (10, 'tanh'), # Primo layer nascosto: 10 neuroni, Tanh
    (10, 'tanh'), # Secondo layer nascosto: 10 neuroni, Tanh
    (2, None)  # Output layer: 2 neuroni, Nessuna funzione lineare
]

model = NeuralNetwork(input_dim, layer_configs)
model = model.to(device)
print("\nModello della rete neurale:")
print(model)

if os.path.isfile(save_path):
    model.load_state_dict(torch.load(save_path))
    print("Weights loaded")

else:

    def dataset_learning(size):
        # Dataset 1: a incrementale da 0 a 360, x=cos(a), y=sin(a)
        dataset = []
        for i in range(size):
            degrees = (360 / (size - 1)) * i  # Incrementa A da 0 a 360
            radians = math.radians(degrees)
            x = math.cos(radians)
            y = math.sin(radians)
            dataset.append({'A': normalize_input(degrees), 'X': x, 'Y': y})
        
        return dataset

    def dataset_testing(size):
        # Dataset 2: a casuale da 0 a 360, x=cos(a), y=sin(a)
        dataset = []
        for _ in range(size):
            degrees = random.uniform(0, 360)
            radians = math.radians(degrees)
            x = math.cos(radians)
            y = math.sin(radians)
            dataset.append({'A': normalize_input(degrees), 'X': x, 'Y': y})

        return dataset

    class Data(Dataset):
        def __init__(self, data):
            # Estrai x e y dal dataset e convertili in array numpy
            input = np.array([item['A'] for item in data])
            output = np.array([[item['X'], item['Y']] for item in data])
            
            # Converti in tensori PyTorch
            self.input = torch.from_numpy(input.astype(np.float32))
            self.output = torch.from_numpy(output.astype(np.float32))
            self.len = self.input.shape[0]
        
        def __getitem__(self, index):
            return self.input[index], self.output[index]
        
        def __len__(self):
            return self.len

    def visualise_datasets(dataset_a, dataset_b):
        plt.figure(figsize=(12, 6))

        # Plot Dataset 1
        plt.subplot(1, 2, 1) # 1 riga, 2 colonne, primo plot
        x_vals1 = [item['X'] for item in dataset_a]
        y_vals1 = [item['Y'] for item in dataset_a]
        plt.plot(x_vals1, y_vals1, 'o', markersize=4)
        plt.title('Learning (incrementale)')
        plt.xlabel('cos(a)')
        plt.ylabel('sin(a)')
        plt.grid(True)
        plt.axis('equal')

        # Plot Dataset 2
        plt.subplot(1, 2, 2) # 1 riga, 2 colonne, secondo plot
        x_vals2 = [item['X'] for item in dataset_b]
        y_vals2 = [item['Y'] for item in dataset_b]
        plt.plot(x_vals2, y_vals2, 'o', markersize=4, alpha=0.6)
        plt.title('Testing (casuale)')
        plt.xlabel('cos(a)')
        plt.ylabel('sin(a)')
        plt.grid(True)
        plt.axis('equal')

        plt.tight_layout()
        plt.show()
    


    """
    Run
    """
    size = 100
    batch_size = 100
    learning_data = dataset_learning(size)
    testing_data = dataset_testing(size)
    #visualise_datasets(learning_data, testing_data)

    # Crea i DataLoader
    train_data = Data(learning_data)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = Data(testing_data)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # Verifica il funzionamento
    print("Verifica del DataLoader:")
    for batch, (input, output) in enumerate(train_dataloader):
        print(f"Batch {batch+1}:")
        print(f"Input shape: {input.shape}")
        print(f"Output shape: {output.shape}")
        break


    """
    Train Neural Network
    """
    def train(dataset: DataLoader, model, num_epochs: int):
        learning_rate = 0.1
        #loss_fn = nn.BCELoss()
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        loss_values = []

        for epoch in range(num_epochs):
            for batch, (input, output) in enumerate(dataset):
                # zero the parameter gradients
                optimizer.zero_grad()
            
                # forward + backward + optimize
                pred = model(input.to(device))
                loss = loss_fn(pred, output.to(device))
                #pred = model(input)
                #loss = loss_fn(pred, output)
                loss.backward()
                optimizer.step()

                loss_values.append(loss.item())

            # print training progress
            if epoch % 1000 == 0:
                loss_value = loss.item()
                perc = (epoch + 1) / num_epochs * 100
                print(f"loss: {loss_value:>7f}      [{perc:>6.2f}%]")

        print("Training Complete")
        return loss_values

    num_epochs = 100000
    loss_values = train(train_dataloader, model, num_epochs)
    torch.save(model.state_dict(), save_path)

    """
    Visualize learning loss
    """
    step = range(len(loss_values))

    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(step, np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    """
    Predict and Visualize Random Values
    """

    # Prepare input tensor
    test_angles = torch.tensor([item['A'] for item in testing_data], dtype=torch.float32)
    test_angles = test_angles.to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(test_angles)

    # Extract true values and predictions
    true_values = np.array([[item['X'], item['Y']] for item in testing_data])
    predicted_values = predictions.cpu().numpy()

    # Visualize results
    plt.figure(figsize=(8, 8)) # Modificato per un singolo grafico quadrato

    # Plot true values
    plt.scatter(true_values[:, 0], true_values[:, 1], c='blue', alpha=0.5, label='True Values')

    # Plot predicted values
    plt.scatter(predicted_values[:, 0], predicted_values[:, 1], c='red', alpha=0.5, label='Predictions')

    plt.title('True Values vs Predicted Values') # Titolo combinato
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

    plt.tight_layout()
    plt.show()

    """
    Calculate Accuracy Metrics
    """
    # Calculate MSE for each component
    mse_x = np.mean((true_values[:, 0] - predicted_values[:, 0])**2)
    mse_y = np.mean((true_values[:, 1] - predicted_values[:, 1])**2)
    total_mse = (mse_x + mse_y) / 2

    # Calculate average Euclidean distance
    euclidean_distances = np.sqrt(np.sum((true_values - predicted_values)**2, axis=1))
    mean_distance = np.mean(euclidean_distances)

    print(f"\nAccuracy Metrics:")
    print(f"MSE (X coordinate): {mse_x:.6f}")
    print(f"MSE (Y coordinate): {mse_y:.6f}")
    print(f"Average MSE: {total_mse:.6f}")
    print(f"Mean Euclidean Distance: {mean_distance:.6f}")


def create_dynamic_neuron_visualization(model, device, normalize_input):
    """
    Crea una visualizzazione dinamica delle attivazioni dei neuroni con slider interattivo
    """
    model.eval()
    
    # Configurazione della figura
    fig = plt.figure(figsize=(12, 8))
    
    # Calcola il numero di layer per determinare il layout
    with torch.no_grad():
        dummy_input = torch.tensor([0.0], dtype=torch.float32).to(device)
        dummy_activations = model.get_activations(dummy_input)
        num_layers = len(dummy_activations)
    
    # Crea i subplot per ogni layer
    axes = []
    for i in range(num_layers):
        ax = plt.subplot(num_layers, 1, i + 1)
        axes.append(ax)
    
    # Spazio per lo slider
    plt.subplots_adjust(bottom=0.15)
    
    # Crea lo slider
    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    angle_slider = widgets.Slider(
        slider_ax, 
        'Angolo (°)', 
        0, 360, 
        valinit=0, 
        valstep=1,
        valfmt='%d°'
    )
    
    # Inizializza i grafici a barre
    bars = []
    for i in range(num_layers):
        # Calcola le attivazioni iniziali
        with torch.no_grad():
            normalized_input = normalize_input(0)
            input_tensor = torch.tensor([normalized_input], dtype=torch.float32).to(device)
            activations = model.get_activations(input_tensor)
        
        # Crea i grafici a barre iniziali
        layer_activations = activations[i].flatten()
        bar = axes[i].bar(range(len(layer_activations)), layer_activations, alpha=0.7)
        bars.append(bar)
        
        # Configurazione degli assi
        axes[i].set_title(f'Attivazioni Layer {i+1}')
        axes[i].set_xlabel('Indice Neurone')
        axes[i].set_ylabel('Valore Attivazione')
        axes[i].grid(True, alpha=0.3)
    
    # Funzione di aggiornamento per lo slider
    def update_visualization(val):
        angle = int(angle_slider.val)
        
        with torch.no_grad():
            normalized_input = normalize_input(angle)
            input_tensor = torch.tensor([normalized_input], dtype=torch.float32).to(device)
            activations = model.get_activations(input_tensor)
        
        # Aggiorna ogni layer
        for i, layer_activations in enumerate(activations):
            layer_vals = layer_activations.flatten()
            
            # Aggiorna le altezze delle barre
            for bar, height in zip(bars[i], layer_vals):
                bar.set_height(height)
            
            # Aggiorna i limiti dell'asse Y per una migliore visualizzazione
            if len(layer_vals) > 0:
                y_min = min(0, float(layer_vals.min()) * 1.1)
                y_max = float(layer_vals.max()) * 1.1
                axes[i].set_ylim(y_min, y_max)
        
        # Aggiorna il titolo principale con l'angolo corrente
        fig.suptitle(f'Attivazioni Neuroni - Angolo: {angle}°', fontsize=14, fontweight='bold')
        
        # Ridisegna la figura
        fig.canvas.draw()
    
    # Collega la funzione di aggiornamento allo slider
    angle_slider.on_changed(update_visualization)
    
    # Aggiornamento iniziale
    update_visualization(0)
    
    # Aggiungi istruzioni per l'utente
    plt.figtext(0.5, 0.02, 'Usa lo slider per cambiare l\'angolo di input (0-360°)', 
                ha='center', va='bottom', fontsize=10, style='italic')
    
    plt.show()
    
    return fig, angle_slider

fig, slider = create_dynamic_neuron_visualization(model, device, normalize_input)
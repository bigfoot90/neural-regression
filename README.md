# Neural Network Regression Example

This project implements a neural network in PyTorch to learn sine and cosine functions from an angular input. The model is trained to map a (normalized) angle to its corresponding (cos(angle), sin(angle)) coordinates on a unit circle. The project includes functionalities for model training, performance visualization, and an interactive visualization of neuron activations.

## Features

- **Flexible Neural Network**: A customizable `NeuralNetwork` class that allows defining the input size, hidden layer configuration (number of neurons and activation function), and output size.
- **Training**: The model is trained using a dataset of incremental angles and tested on random angles. Mean Squared Error (MSE) loss function and Stochastic Gradient Descent (SGD) optimizer are used.
- **Weight Saving/Loading**: Model weights are saved to a `weights.ai` file after training and automatically loaded if the file exists.
- **Visualization**:
    - **Training Loss**: Graph of loss over time during training.
    - **Predictions**: Scatter plot comparing true values (cos, sin) with predicted values from the neural network.
    - **Accuracy Metrics**: Calculation of MSE for X and Y coordinates, and the average Euclidean distance between true and predicted values.
    - **Dynamic Neuron Visualization**: An interactive visualization showing neuron activations in each layer of the neural network and the model's output in real-time, controllable via an input angle slider.

## Requirements
Install the necessary dependencies:
```sh
pip install -r requirements.txt
```

## Useage
To run the script, execute the following command in your terminal:
```sh
python3 main.py
```

At the beginning the script will check if the file `weights.ai` exists. If it does, the script will load the weights from the file and skip the training process. If the file does not exist, the script will train the model and save the weights to the file.

After the training process, the script will show the following plots:
- Training Loss: A plot that shows the training loss over time.
- Predictions: A scatter plot that shows the true values (cos, sin) and the predicted values from the neural network.
- Accuracy Metrics: The Mean Squared Error (MSE) for the X and Y coordinates, and the average Euclidean distance between the true and predicted values.

Finally, the script will show an interactive plot that allows you to control the input angle and see the activations of the neurons in each layer of the neural network and the output of the model in real time.

### Prediction values
![Prediction](https://github.com/bigfoot90/neural-regression/blob/main/doc/prediction.jpg)

### Dynamic dashboard for neuron activations
![Prediction](https://github.com/bigfoot90/neural-regression/blob/main/doc/neuron-activation-dynamic.png)

## Contributions
Contributions are welcome! If you have suggestions for improving the project, please open an issue or submit a pull request.
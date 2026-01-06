# RNN Chatbot Project

This project implements a Recurrent Neural Network (RNN) chatbot from scratch using pure Python. It demonstrates the fundamentals of RNNs, including forward propagation, backpropagation through time (BPTT), and stochastic gradient descent (SGD), without relying on deep learning frameworks like TensorFlow or PyTorch.

## Group Members

*   **Abdul Rehman** (22sp-052-cs)
*   **Afnan** (22sp-051-cs)
*   **Hamza** (22sp-045-cs)
*   **Muhamil** (22sp-011-cs)

## Project Description

The `RNN_chatbot.py` script contains a custom implementation of a simple RNN class. It performs intent classification to understand user inputs and provides appropriate responses based on a predefined dataset.

Key features:
*   **Pure Python Implementation**: No external heavy ML libraries.
*   **Custom RNN Architecture**: Implements standard RNN cells with hidden states.
*   **Training Mode**: Learns from `Data.json` using backpropagation.
*   **Chat Mode**: Interactive command-line interface to chat with the trained bot.

## How to Run

### Prerequisites
*   Python 3.x

### Setup
*   Clone the repository
*   cd into the repository
*   Run `python RNN_chatbot.py train` to train the model
*   Run `python RNN_chatbot.py` to start the chatbot

### 1. Training the Model
Before you can chat with the bot, you need to train it. The training process loads data from `Data.json`, trains the RNN for a fixed number of epochs, and saves the model weights to `model.json`.

Run the following command in your terminal:

```bash
python RNN_chatbot.py train
```

### 2. Running the Chatbot
Once the model is trained and `model.json` is created, you can start the chat interface.

Run the command:

```bash
python RNN_chatbot.py
```

### Usage
*   Type your message and press Enter.
*   The bot will classify your intent and respond.
*   Type `quit` to exit the chat.

Example interactions:
> You: hi  
> Bot: Hello! How can I help you?
>
> You: tell me a joke  
> Bot: Why did the Python programmer need glasses? Because he couldn't C#.

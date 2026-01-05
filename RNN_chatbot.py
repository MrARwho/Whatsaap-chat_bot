import random
import math
import json
import sys
import os

def rand_matrix(rows, cols, scale=0.1):
    """Creates a matrix of random weights."""
    matrix = []

    for r in range(rows):
        
        single_row = []
        
        for c in range(cols):
            num = random.random()
            num = num * 2
            num = num - 1
            final_value = num * scale
            single_row.append(final_value)
        
        matrix.append(single_row)
    return matrix

def zeros(rows, cols):
    """Creates a matrix of zeros."""
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def mat_vec_mul(W, x):
    """Multiplies matrix W by vector x."""
    # W is (rows, cols), x is (cols) -> output is (rows)
    out = []
    for i in range(len(W)):
        val = 0
        for j in range(len(x)):
            weight = W[i][j]  
            input_value = x[j]
            val = val + (weight * input_value)
        
        out.append(val)
    return out

def element_add(v1, v2):
    """Adds two vectors element-wise."""
    result = []
    for i in range(len(v1)):
        val1 = v1[i]
        val2 = v2[i]
        total = val1 + val2
        result.append(total)
    return result

def element_tanh(v):
    """Applies tanh to a vector."""
    result = []
    for i in range(len(v)):
        val = v[i]
        result.append(math.tanh(val))
    return result

def softmax(v):
    """Computes softmax probabilities for a vector."""
    exps = []
    for i in range(len(v)):
        val = v[i]
        exps.append(math.exp(val - max(v)))
    sum_exps = sum(exps)
    result = []
    for i in range(len(exps)):
        val = exps[i]
        result.append(val / sum_exps)
    return result

def outer_product(v1, v2):
    """Calculates outer product of two vectors (v1 column, v2 row)."""
    result = []
    for v1_i in v1:
        current_row = []
        for v2_j in v2:
            val = v1_i * v2_j
            current_row.append(val)
        result.append(current_row)

return result


# def get_training_data():
#     # Training data: (Sentence, Intent_ID)
#     train_data = [
#         ("hi", 0), ("hello", 0), ("hey there", 0),  # 0: Greeting
#         ("bye", 1), ("goodbye", 1), ("see you", 1), # 1: Goodbye
#         ("name", 2), ("who are you", 2),            # 2: Identity
#         ("joke", 3), ("tell me a joke", 3),         # 3: Joke
#     ]

#     # Responses map
#     intents = {0: "greeting", 1: "goodbye", 2: "identity", 3: "joke"}
#     responses = {
#         "greeting": "Hello! How can I help you?",
#         "goodbye": "Goodbye! Have a nice day.",
#         "identity": "I am a pure Python RNN chatbot.",
#         "joke": "Why did the Python programmer need glasses? Because he couldn't C#."
#     }
#     return train_data, intents, responses

def build_vocab(train_data):
    # Build Vocabulary
    vocab = set()
    for sent, _ in train_data:
        for word in sent.split():
            vocab.add(word)
    vocab = sorted(list(vocab))
    word_to_ix = {w: i for i, w in enumerate(vocab)}
    return vocab, word_to_ix

def one_hot(word, word_to_ix, vocab_size):
    """Converts a word to a one-hot vector."""
    vec = [0] * vocab_size
    if word in word_to_ix:
        vec[word_to_ix[word]] = 1
    return vec

# ==========================================
# 3. RNN MODEL FROM SCRATCH
# ==========================================
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        # Initialize Weights (Xavier-like initialization)
        # Wxh: Input -> Hidden
        self.Wxh = rand_matrix(hidden_size, input_size)
        # Whh: Hidden -> Hidden
        self.Whh = rand_matrix(hidden_size, hidden_size)
        # Why: Hidden -> Output
        self.Why = rand_matrix(output_size, hidden_size)
        
        # Biases
        self.bh = [0] * hidden_size
        self.by = [0] * output_size

    def forward(self, inputs):
        """
        inputs: list of one-hot vectors (one for each word in sentence)
        Returns: output probabilities, list of hidden states
        """
        h = [0] * self.hidden_size # Initial hidden state
        self.last_inputs = inputs
        self.hs = { -1: h } # Dictionary to store hidden states for BPTT

        # Loop through time steps (words)
        for t, x in enumerate(inputs):
            # h[t] = tanh(Wxh * x + Whh * h[t-1] + bh)
            term1 = mat_vec_mul(self.Wxh, x)
            term2 = mat_vec_mul(self.Whh, self.hs[t-1])
            combined = element_add(element_add(term1, term2), self.bh)
            self.hs[t] = element_tanh(combined)

        # Output computed only at the last step (Many-to-One)
        last_h = self.hs[len(inputs) - 1]
        logits = element_add(mat_vec_mul(self.Why, last_h), self.by)
        probs = softmax(logits)
        return probs, self.hs

    def backprop(self, probs, target_class, learning_rate=0.1):
        """Calculates gradients and updates weights."""
        # 1. Calculate Output Error (dy)
        # Gradient of Cross-Entropy Loss w.r.t Softmax input is simply (prob - 1) for the correct class
        dy = list(probs)
        dy[target_class] -= 1
        
        # Initialize Gradients
        dWhy = zeros(len(self.Why), len(self.Why[0]))
        dby = [0] * len(self.by)
        dWxh = zeros(len(self.Wxh), len(self.Wxh[0]))
        dWhh = zeros(len(self.Whh), len(self.Whh[0]))
        dbh = [0] * len(self.bh)
        
        # Calculate Output Gradients
        last_t = len(self.last_inputs) - 1
        dWhy = outer_product(dy, self.hs[last_t])
        dby = dy
        
        # Backpropagate through time
        dh_next = [0] * self.hidden_size # Gradient flowing back from next step
        
        # Gradient flowing from Output layer to last hidden state
        # dh = W_hy.T * dy
        dh = [0] * self.hidden_size
        for i in range(len(self.Why)):
            for j in range(self.hidden_size):
                dh[j] += self.Why[i][j] * dy[i]

        for t in reversed(range(len(self.last_inputs))):
            x = self.last_inputs[t]
            
            # Add gradient from future step
            dh = element_add(dh, dh_next)
            
            # Backprop through tanh: dtanh = (1 - h^2) * dh
            dtanh = [(1 - self.hs[t][i]**2) * dh[i] for i in range(self.hidden_size)]
            
            # Gradients for weights
            dbh = element_add(dbh, dtanh)
            
            # dWxh += dtanh * x.T
            dWxh_step = outer_product(dtanh, x)
            for i in range(len(dWxh)):
                for j in range(len(dWxh[0])):
                    dWxh[i][j] += dWxh_step[i][j]
            
            # dWhh += dtanh * h_prev.T
            dWhh_step = outer_product(dtanh, self.hs[t-1])
            for i in range(len(dWhh)):
                for j in range(len(dWhh[0])):
                    dWhh[i][j] += dWhh_step[i][j]

            # Calculate dh for next iteration (previous time step)
            # dh_next = Whh.T * dtanh
            dh_next = [0] * self.hidden_size
            for i in range(len(self.Whh)):
                for j in range(self.hidden_size):
                    dh_next[j] += self.Whh[i][j] * dtanh[i]
            
            # Reset dh for strict BPTT flow (in simple version)
            dh = [0] * self.hidden_size

        # Update Weights (SGD)
        def update_weights(W, dW):
            for i in range(len(W)):
                for j in range(len(W[0])):
                    W[i][j] -= learning_rate * dW[i][j]
        
        def update_bias(b, db):
            for i in range(len(b)):
                b[i] -= learning_rate * db[i]

        update_weights(self.Why, dWhy)
        update_weights(self.Wxh, dWxh)
        update_weights(self.Whh, dWhh)
        update_bias(self.by, dby)
        update_bias(self.bh, dbh)

def load_training_data_from_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    raw_data = [] # List of (sentence, tag_index)
    classes = {}  # Map index to tag name
    responses = {} # Map tag name to response
    
    for idx, intent in enumerate(data['intents']):
        tag = intent['tag']
        classes[idx] = tag
        responses[tag] = intent['response']
        
        for pattern in intent['patterns']:
            # Basic preprocessing (remove punctuation)
            clean_pattern = "".join([c for c in pattern if c.isalnum() or c.isspace()])
            raw_data.append((clean_pattern.lower(), idx))
            
    return raw_data, classes, responses

def save_model(filename, model, word_to_ix, intents, responses):
    """Saves model weights and vocabulary to a JSON file."""
    checkpoint = {
        # 1. Model Weights
        'hidden_size': model.hidden_size,
        'Wxh': model.Wxh,
        'Whh': model.Whh,
        'Why': model.Why,
        'bh': model.bh,
        'by': model.by,
        # 2. Vocabulary & Config
        'word_to_ix': word_to_ix,
        'intents': intents,
        'responses': responses
    }
    
    with open(filename, 'w') as f:
        json.dump(checkpoint, f)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Loads a trained model from a JSON file."""
    with open(filename, 'r') as f:
        checkpoint = json.load(f)
    
    # Reconstruct Vocabulary
    word_to_ix = checkpoint['word_to_ix']
    intents = {int(k): v for k, v in checkpoint['intents'].items()} # JSON converts int keys to strings
    responses = checkpoint['responses']
    vocab_size = len(word_to_ix)
    num_classes = len(intents)
    
    # Re-initialize the RNN
    model = SimpleRNN(vocab_size, checkpoint['hidden_size'], num_classes)
    
    # Load Weights
    model.Wxh = checkpoint['Wxh']
    model.Whh = checkpoint['Whh']
    model.Why = checkpoint['Why']
    model.bh = checkpoint['bh']
    model.by = checkpoint['by']
    
    return model, word_to_ix, intents, responses

# ==========================================
# 4. TRAINING FUNCTION
# ==========================================
def train_mode():
    print("Starting Training Mode...")
    
    data_file = 'Data.json'
    if os.path.exists(data_file):
        print(f"Loading training data from {data_file}...")
        train_data, classes, responses = load_training_data_from_file(data_file)
        # Convert classes dict to match expected format (int -> tag) if needed
        # load_training_data_from_file returns:
        # raw_data: [(text, label_idx), ...], classes: {idx: tag}, responses: {tag: response}
        # get_training_data returns:
        # train_data: [(text, label_idx)], intents: {idx: tag}, responses: {tag: response}
        # The formats align perfectly.
        intents = classes
    else:
        print(f"'{data_file}' not found. Using hardcoded training data.")
        train_data, intents, responses = get_training_data()

    vocab, word_to_ix = build_vocab(train_data)
    vocab_size = len(vocab)
    num_classes = len(intents)

    hidden_size = 16
    rnn = SimpleRNN(vocab_size, hidden_size, num_classes)
    
    print(f"Training on {len(train_data)} sentences...")
    epochs = 1500
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        # Shuffle data
        random.shuffle(train_data)
        
        for text, label in train_data:
            # Prepare input
            tokens = text.split()
            inputs = [one_hot(w, word_to_ix, vocab_size) for w in tokens]
            
            if not inputs: continue
            
            # Forward
            probs, _ = rnn.forward(inputs)
            
            # Calculate Loss (Cross Entropy)
            # Loss = -ln(probability of correct class)
            safe_prob = max(0.0000001, probs[label])
            total_loss += -math.log(safe_prob)
            
            # Accuracy
            prediction = probs.index(max(probs))
            if prediction == label:
                correct += 1
                
            # Backward
            rnn.backprop(probs, label, learning_rate=0.01)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {total_loss:.4f} | Acc: {correct}/{len(train_data)}")

    save_model('model.json', rnn, word_to_ix, intents, responses)

# ==========================================
# 5. CHAT FUNCTION
# ==========================================
def chat_mode():
    if not os.path.exists('model.json'):
        print("Model file 'model.json' not found.")
        print("Please run with 'train' argument first: python RNN_chatbot.py train")
        return

    print("Loading model...")
    rnn, word_to_ix, intents, responses = load_model('model.json')
    vocab_size = len(word_to_ix)
    
    print("\nModel Loaded! Chat with me (type 'quit' to exit)")
    print("Try saying: 'hi', 'who are you', 'tell me a joke', 'bye'")
    
    while True:
        try:
            user_input = input("You: ").lower().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
            
        if user_input == 'quit':
            break
        
        # Preprocess
        tokens = user_input.split()
        inputs = []
        for w in tokens:
            if w in word_to_ix:
                inputs.append(one_hot(w, word_to_ix, vocab_size))
        
        if not inputs:
            print("Bot: I don't know those words yet.")
            continue

        # Predict
        probs, _ = rnn.forward(inputs)
        
        if not probs:
             print("Bot: Error in prediction.")
             continue

        pred_idx = probs.index(max(probs))
        pred_intent = intents[pred_idx]

        print(f"\n(Debug) Confidence: {max(probs):.2f}")
        # Show top 3 guesses
        top_3 = sorted(zip(probs, intents.values()), reverse=True)[:3]
        for p, tag in top_3:
            print(f"(Debug) {tag}: {p:.4f}")
        
        # Confidence threshold
        if max(probs) < 0.8:
            print("Bot: I'm not sure what you mean.")
        else:
            print(f"Bot: {responses[pred_intent]}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'train':
        train_mode()
    else:
        chat_mode()
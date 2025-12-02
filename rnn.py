import numpy as np

# Toy Dataset: Simple sequence
data = "hello"
chars = list(set(data))
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}
input_size = len(chars)
hidden_size = 10
output_size = len(chars)

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, hprev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        
        for t in range(len(inputs)):
            xs[t] = np.zeros((input_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) 
            
        return ps, hs

# Execution
rnn = RNN(input_size, hidden_size, output_size)
inputs = [char_to_ix[ch] for ch in data]
hprev = np.zeros((hidden_size, 1))
probs, _ = rnn.forward(inputs, hprev)
print("\nRNN Output Probabilities for last character:\n", probs[len(inputs)-1])
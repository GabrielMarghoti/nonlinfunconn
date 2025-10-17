import numpy as np
import matplotlib.pyplot as plt

# --- 1. RNN Model Definition ---
class SimpleRNN:
    """A simple Recurrent Neural Network implemented from scratch."""
    def __init__(self, input_size, hidden_size, output_size):
        # Model dimensions
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        # Weight matrices
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01

        # Bias vectors
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        """
        Performs the forward pass of the RNN.
        Returns outputs, hidden states, and pre-activations for backprop.
        """
        h = np.zeros((self.hidden_size, 1))
        hidden_states, pre_activations, outputs = {}, {}, {}
        hidden_states[-1] = np.copy(h)

        for t in range(len(inputs)):
            net_h = np.dot(self.Wxh, inputs[t]) + np.dot(self.Whh, h) + self.bh
            pre_activations[t] = net_h
            h = np.tanh(net_h)
            hidden_states[t] = h
            outputs[t] = np.dot(self.Why, h) + self.by
        
        return outputs, hidden_states, pre_activations

    def train(self, inputs, targets, learning_rate=0.001):
        """
        Performs one step of training with backpropagation through time (BPTT).
        """
        # Forward pass
        outputs, hidden_states, _ = self.forward(inputs)
        
        # Initialize gradients
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        
        loss = 0
        dh_next = np.zeros_like(hidden_states[0])

        # Backward pass (BPTT)
        for t in reversed(range(len(inputs))):
            # Loss and output gradient
            error = outputs[t] - targets[t]
            loss += 0.5 * np.sum(error**2)
            
            # Gradients for output layer
            dWhy += np.dot(error, hidden_states[t].T)
            dby += error
            
            # Backpropagate through hidden layer
            dh = np.dot(self.Why.T, error) + dh_next
            dtanh = (1 - hidden_states[t]**2) * dh # Derivative of tanh
            
            # Gradients for hidden layer
            dbh += dtanh
            dWxh += np.dot(dtanh, inputs[t].T)
            dWhh += np.dot(dtanh, hidden_states[t-1].T)
            
            # Pass gradient to next time step
            dh_next = np.dot(self.Whh.T, dtanh)
            
        # Clip gradients to prevent exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
            
        # Update weights and biases
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby
        
        return loss

# --- 2. Data Generation & Training ---
def generate_data(total_len=1000, noise_factor=0.1):
    """Generates a noisy sine wave."""
    t = np.linspace(0, 100, total_len)
    data = np.sin(t) + np.random.randn(total_len) * noise_factor
    inputs = [data[i:i+1].reshape(1,1) for i in range(total_len - 1)]
    targets = [data[i+1:i+2].reshape(1,1) for i in range(total_len - 1)]
    return inputs, targets

# Hyperparameters
input_size = 1
hidden_size = 50
output_size = 1
epochs = 200
seq_length = 1000

# Initialize model and data
rnn = SimpleRNN(input_size, hidden_size, output_size)
inputs, targets = generate_data(seq_length)

print("Starting RNN training...")
loss_history = []
for epoch in range(epochs):
    loss = rnn.train(inputs, targets)
    loss_history.append(loss)
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# --- 3. Test the Model and Plot Results ---
print("\nTesting the trained model...")
test_inputs, test_targets = generate_data(200, noise_factor=0.05)
predictions, _, _ = rnn.forward(test_inputs)

# Extract numerical values for plotting
true_values = np.array([t.item() for t in test_targets])
predicted_values = np.array([p.item() for p in predictions.values()])

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(true_values, label='True Signal', color='blue', alpha=0.7)
plt.plot(predicted_values, label='RNN Prediction', color='red', linestyle='--')
plt.title("RNN Prediction vs. True Signal")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 4. NEGF Computation ---
def compute_g_and_G(rnn_model, inputs, max_path_length, noise_std=0.0):
    """
    Computes the state-dependent direct (g) and effective (G) Green's functions.
    'g' is the one-step Jacobian.
    'G' is the product of Jacobians over a path length 'l'.
    """
    # Run a forward pass to get all hidden states and pre-activations
    _, hidden_states, pre_activations = rnn_model.forward(inputs)

    # Add noise to hidden states for trial-to-trial variability
    if noise_std > 0.0:
        for t in hidden_states:
            if t >= 0:
                hidden_states[t] += np.random.randn(*hidden_states[t].shape) * noise_std
    
    # 1. Compute the sequence of one-step Jacobians (our 'g')
    # g_ij(t, t-1) = ∂h_i(t) / ∂h_j(t-1)
    jacobians = {}
    for t in range(len(inputs)):
        # Derivative of tanh is (1 - h(t)^2)
        dtanh = 1 - hidden_states[t]**2
        # Element-wise multiplication with the weights
        jacobians[t] = rnn_model.Whh * dtanh.T
    
    # g is the dictionary of jacobians, representing the direct one-step response
    g = jacobians

    # 2. Compute the effective Green's function G over different path lengths
    # G(t, t-l) = J(t) * J(t-1) * ... * J(t-l+1)
    effective_G = {} # G[l] will store G for path length l
    for l in range(1, max_path_length + 1):
        G_l = {}
        for t in range(l - 1, len(inputs)):
            # Chain rule: multiply the Jacobians backward in time
            G_t_l = np.identity(rnn_model.hidden_size)
            for step in range(l):
                G_t_l = np.dot(jacobians[t - step], G_t_l)
            G_l[t] = G_t_l
        effective_G[l] = G_l
        
    return g, effective_G

# --- 5. Variance Analysis ---
print("\nComputing NEGF and analyzing variance...")
num_trials = 20
max_path_length = 15
process_noise_std = 0.01 # Small noise to introduce trial variability

# Store G for a specific element (e.g., neuron 0 influencing itself) across trials
G_element_trials = {l: [] for l in range(1, max_path_length + 1)}

for trial in range(num_trials):
    # Use the same input sequence for all trials
    _, G_trial = compute_g_and_G(rnn, inputs, max_path_length, noise_std=process_noise_std)
    
    for l in range(1, max_path_length + 1):
        # We analyze the influence at a fixed time point, e.g., t=500
        t_fixed = 500
        if t_fixed in G_trial[l]:
            # Get the influence of neuron 0 on neuron 0 after l steps
            g_00_val = G_trial[l][t_fixed][0, 0]
            G_element_trials[l].append(g_00_val)

# Calculate the variance for each path length
variances = []
path_lengths = []
for l, values in G_element_trials.items():
    if values: # Ensure we have data for this path length
        variances.append(np.var(values))
        path_lengths.append(l)

# --- 6. Plot the Final Result ---
plt.figure(figsize=(8, 6))
plt.plot(path_lengths, variances, marker='o', linestyle='-')
plt.title("Variance of Effective Green's Function ($G_{00}$) vs. Path Length")
plt.xlabel("Path Length (scale l)")
plt.ylabel("Variance of $G_{00}(t, t-l)$ across trials")
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.show()
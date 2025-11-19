"""
Spiking Neural Network with STDP and Winner-Take-All (WTA)
-----------------------------------------------------------------
This script builds a 2-layer SNN from scratch using only NumPy.
It learns to detect *two* separate synchronous patterns using a
competitive learning rule (WTA + STDP).

This demonstrates how a network can develop a Rank-2 representation
to classify two independent features.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Simulation Hyperparameters ---
num_steps     = 4000   # Total simulation time
num_inputs    = 100    # Size of the input layer
num_outputs   = 10     # 10 output neurons compete
v_thresh      = 1.0    # Neuron spike threshold
beta          = 0.9    # Membrane leak rate (closer to 1 = slower leak)
tau_pre       = 20.0   # Pre-synaptic trace time constant (ms)
tau_post      = 20.0   # Post-synaptic trace time constant (ms)

# STDP Learning rates
lr_ltp        = 0.01   # Potentiation (strengthening)
lr_ltd        = 0.005  # Depression (weakening)

# Pre-calculate decay factors for efficiency
pre_decay     = np.exp(-1.0 / tau_pre)
post_decay    = np.exp(-1.0 / tau_post)

# --- 2. Define the Input Spike Patterns ---
prob_noise    = 0.02   # Background random spike probability
prob_pattern  = 0.5    # Pattern spike probability
pattern_a_indices = range(10, 20) # Pattern A
pattern_b_indices = range(30, 40) # Pattern B

# Generate the full spike train
spike_data = np.zeros((num_steps, num_inputs))
# Store pattern times for plotting
pattern_a_times = []
pattern_b_times = []

for i in range(num_steps):
    # Background noise
    spike_data[i] = (np.random.rand(num_inputs) < prob_noise).astype(float)
    
    # Insert Pattern A (e.g., every 100 steps)
    if i % 100 == 0 and i > 0:
        pattern_spikes_a = (np.random.rand(len(pattern_a_indices)) < prob_pattern).astype(float)
        spike_data[i, pattern_a_indices] = np.maximum(spike_data[i, pattern_a_indices], pattern_spikes_a)
        pattern_a_times.append(i)

    # Insert Pattern B (e.g., every 170 steps - out of phase)
    if i % 170 == 0 and i > 0:
        pattern_spikes_b = (np.random.rand(len(pattern_b_indices)) < prob_pattern).astype(float)
        spike_data[i, pattern_b_indices] = np.maximum(spike_data[i, pattern_b_indices], pattern_spikes_b)
        pattern_b_times.append(i)

print("--- Running on NumPy with WTA ---")

# --- 3. Define the Network and Dynamics ---
class SNN_WTA_STDP_Learner:
    def __init__(self):
        # 3.1 Network Parameters
        # Weights are [num_outputs, num_inputs]
        self.weights = np.random.rand(num_outputs, num_inputs) * 0.1
        self.mem = np.zeros(num_outputs)  # Membrane potential
        
        # 3.2 STDP Trace Variables
        self.pre_trace = np.zeros(num_inputs)
        self.post_trace = np.zeros(num_outputs)

    def step(self, spk_in):
        """
        Runs one time step of the LIF neuron dynamics WITH
        a Winner-Take-All (WTA) competitive mechanism.
        """
        
        # 1. Calculate synaptic input current (10x100 @ 100x1 -> 10x1)
        cur_in = self.weights @ spk_in
        
        # 2. Update membrane potential (Leaky Integrate)
        self.mem = self.mem * beta + cur_in
        
        # 3. Winner-Take-All (WTA)
        spk_out = np.zeros(num_outputs)
        
        # Find all neurons that *could* spike
        potential_spikers = np.where(self.mem >= v_thresh)[0]
        
        if len(potential_spikers) > 0:
            # Find the *single winner* (neuron with highest voltage)
            winner_idx = potential_spikers[np.argmax(self.mem[potential_spikers])]
            
            # Only that winner gets to spike
            spk_out[winner_idx] = 1.0
            
            # Reset *only* the winner's membrane
            self.mem[winner_idx] = 0.0
        
        # Reset any other neurons that were above threshold but lost
        self.mem[np.logical_and(self.mem >= v_thresh, spk_out == 0)] = v_thresh
        
        return spk_out, self.mem

    def stdp_update(self, spk_in, spk_out):
        """
        Applies the STDP learning rule to the weights.
        This rule is identical to the previous script, but now
        it only strengthens the weights of the *winning* neuron.
        """
        
        # 1. Update traces
        self.pre_trace = self.pre_trace * pre_decay + spk_in
        self.post_trace = self.post_trace * post_decay + spk_out
        
        # 2. Potentiation (LTP)
        # (10, 1) @ (1, 100) -> (10, 100) matrix
        delta_w_ltp = lr_ltp * (spk_out.reshape(-1, 1) @ self.pre_trace.reshape(1, -1))
        
        # 3. Depression (LTD)
        # (10, 1) @ (1, 100) -> (10, 100) matrix
        delta_w_ltd = -lr_ltd * (self.post_trace.reshape(-1, 1) @ spk_in.reshape(1, -1))

        # 4. Apply changes
        self.weights += delta_w_ltp
        self.weights += delta_w_ltd
            
        # 5. Clamp weights
        self.weights = np.clip(self.weights, 0, 1.0)
        
    def get_weights(self):
        return self.weights.copy()


# --- 4. Simulation Loop ---
net = SNN_WTA_STDP_Learner()

# Storage for plotting
mem_rec = []
spk_rec = []
initial_weights = net.get_weights()
weight_history = [initial_weights]
s_initial = np.linalg.svd(initial_weights, compute_uv=False)
svd_history = [s_initial] 

print("--- Starting STDP + WTA simulation ---")
for step in range(num_steps):
    spk_in = spike_data[step]
    spk_out, mem = net.step(spk_in)
    net.stdp_update(spk_in, spk_out)
    
    mem_rec.append(mem.copy())
    spk_rec.append(spk_out.copy())
    
    if (step + 1) % 100 == 0: # Store weights and SVD every 100 steps
        weights_now = net.get_weights()
        weight_history.append(weights_now)
        s = np.linalg.svd(weights_now, compute_uv=False)
        svd_history.append(s)

print("--- Simulation complete ---")

# Convert lists to NumPy arrays
mem_rec = np.array(mem_rec)
spk_rec = np.array(spk_rec).T # Transpose (neurons x time)
weight_history = np.array(weight_history)
svd_history = np.array(svd_history)
weight_time = np.arange(0, num_steps + 1, 100)

# --- 5. Visualize Results ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 18), sharex=True, 
                                        gridspec_kw={'height_ratios': [1, 1, 2, 2]})
plt.xlim(0, 2000) # Zoom in on the first 2000 steps

# Plot 1: Input Spikes (Raster Plot)
ax1.imshow(spike_data[:2000, :].T, aspect='auto', cmap='binary', origin='lower')
ax1.set_title('Input Spike Raster')
ax1.set_ylabel('Input Neuron')
# Add lines for both patterns
for t in pattern_a_times:
    if t < 2000: ax1.vlines(t, 0, num_inputs, colors='cyan', linestyles='dashed', lw=1)
for t in pattern_b_times:
    if t < 2000: ax1.vlines(t, 0, num_inputs, colors='magenta', linestyles='dashed', lw=1)

# Plot 2: Output Neuron Spikes
ax2.imshow(spk_rec[:, :2000], aspect='auto', cmap='binary', origin='lower')
ax2.set_title('Output Neuron Firing (WTA Competition)')
ax2.set_ylabel('Output Neuron')
ax2.set_yticks(range(num_outputs))

# Plot 3: Weight Evolution
# Avg. weight for Pattern A inputs vs. Pattern B inputs
pattern_a_weights = weight_history[:, :, 10:20]
pattern_b_weights = weight_history[:, :, 30:40]

# Plot average weight *across all output neurons*
ax3.plot(weight_time, np.mean(pattern_a_weights, axis=(1, 2)), 
         label='Avg. Weight for Pattern A (10-20)', lw=2, color='cyan')
ax3.plot(weight_time, np.mean(pattern_b_weights, axis=(1, 2)), 
         label='Avg. Weight for Pattern B (30-40)', lw=2, color='magenta')
ax3.set_title('Synaptic Weight Learning (STDP + WTA)')
ax3.set_ylabel('Synaptic Weight')
ax3.legend()
ax3.grid(True)

# Plot 4: SVD SPECTRUM EVOLUTION
# Plot the top 5 singular values
ax4.plot(weight_time, svd_history[:, 0], label='$\sigma_1$ (Singular Value 1)', lw=2)
ax4.plot(weight_time, svd_history[:, 1], label='$\sigma_2$ (Singular Value 2)', lw=2)
ax4.plot(weight_time, svd_history[:, 2], label='$\sigma_3$ (Singular Value 3)', linestyle=':')
ax4.plot(weight_time, svd_history[:, 3], label='$\sigma_4$ (Singular Value 4)', linestyle=':')
ax4.plot(weight_time, svd_history[:, 4], label='$\sigma_5$ (Singular Value 5)', linestyle=':')

ax4.set_title('Evolution of Singular Values of Weight Matrix')
ax4.set_xlabel('Time Step')
ax4.set_ylabel('Singular Value ($\sigma$)')
ax4.legend(loc='upper left')
ax4.grid(True)

plt.tight_layout()
plt.show()
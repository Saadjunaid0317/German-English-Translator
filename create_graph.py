import matplotlib.pyplot as plt
import numpy as np

# --- Parameters to define the shape of the graph ---
EPOCHS = 30
initial_loss = 1.8
final_loss_floor = 0.45
noise_level = 0.05

# --- Generate realistic-looking data ---
# Create a decreasing curve using a mathematical function
epochs = np.arange(1, EPOCHS + 1)
# Create an exponential decay curve
loss_curve = (initial_loss - final_loss_floor) * np.exp(-epochs / (EPOCHS / 3)) + final_loss_floor
# Add some random noise to make it look more authentic
noise = np.random.normal(0, noise_level, EPOCHS) * (1 - epochs / (EPOCHS * 1.5)) # Noise decreases over time
final_loss_data = loss_curve + noise

# Ensure the loss doesn't dip below the floor
final_loss_data = np.maximum(final_loss_data, final_loss_floor + np.random.uniform(0, 0.05, EPOCHS))


# --- Plotting the graph ---
print("Generating illustrative training loss plot...")

plt.figure(figsize=(10, 6))
plt.plot(epochs, final_loss_data, marker='o', linestyle='-', color='#007acc', label='Training Loss')

# --- Adding titles and labels for a professional look ---
plt.title('Illustrative Training Loss per Epoch', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(np.arange(0, EPOCHS + 1, 2)) # Show ticks every 2 epochs
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout() # Adjusts plot to prevent labels from being clipped

# --- Save the plot as an image file ---
output_filename = 'illustrative_loss_plot.png'
plt.savefig(output_filename)

print(f"Graph saved successfully as '{output_filename}'")

# --- Display the plot ---
plt.show()
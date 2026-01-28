import matplotlib.pyplot as plot

def draw_neural_network_diagram(layer_sizes, layer_labels):
    fig, ax = plot.subplots(figsize=(10, 6))
    ax.axis('off')
	
    x_space = 2.0
    y_space = 1.0
	
    for layer_indx, layer_size in enumerate(layer_sizes):
        x = layer_indx * x_space
        y_top = (layer_size - 1) * y_space / 2
        for neuron_index in range(layer_size):
            y = y_top - neuron_index * y_space
            circle = plot.Circle((x, y), 0.15, fill=True)
            ax.add_patch(circle)
            # Draw network
            if layer_indx > 0:
                prev_layer_size = layer_sizes[layer_indx - 1]
                prev_y_top = (prev_layer_size - 1) * y_space / 2
                for prev_neuron in range(prev_layer_size):
                    prev_y = prev_y_top - prev_neuron * y_space
                    ax.plot([x - x_space, x], [prev_y, y], 'gray', linewidth=0.5)
        # label
        ax.text(x, y_top + 0.1, layer_labels[layer_indx],
                ha='center', fontsize=10, fontweight='bold')
    plot.title("Neural Network Architecture for Credit Card Fraud Detection")
    plot.show()
 
 
layers = [6, 8, 4, 1]
labels = [
    "Input Layer(6 Features)",
    "Hidden Layer 1 - (8 Neurons)",
    "Hidden Layer 2 - (4 Neurons)",
    "Output Layer\n(Fraud / Genuine)"
]
draw_neural_network_diagram(layers, labels)
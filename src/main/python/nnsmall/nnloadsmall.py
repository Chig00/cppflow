import tensorflow
from . import nnsavesmall

TEST_SIZE = 100

def main():
    """Loads the model from disk and displays its information."""
    
    # Loads the model and displays its summary.
    model = tensorflow.keras.models.load_model(nnsavesmall.FILE_NAME)
    print("\nSummary:")
    model.summary()
    
    # Runs a quick test on the neural network.
    print("\nTesting...")
    (test_data, test_labels) = nnsavesmall.make_data(TEST_SIZE)
    model.evaluate(test_data, test_labels)
    
    # Displays in-depth information for each layer.
    print("\nLayer Analysis:", end = "")
    
    for layer in model.layers:
        print("\nConfiguration:\n", layer.get_config(), "\n\nWeights:\n", layer.get_weights(), sep = "")

# Allows import without run.
if __name__ == "__main__":
    main()

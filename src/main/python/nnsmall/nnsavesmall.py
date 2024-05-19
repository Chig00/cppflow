import random
import tensorflow

# Constants
TRAIN_SIZE = 10000
TEST_SIZE = 2000
THRESHOLDS = [0.5, 0.5]
CLASSES = 2
NEURONS = 4
ACTIVATION = "relu"
OPTIMISER = "adam"
METRICS = ["accuracy"]
EPOCHS = 10
FILE_NAME = "models/small"

def make_data(size):
    """Return a pair of data and labels of the given size."""
    
    data = []
    labels = []
    
    for i in range(size):
        data_point = [random.random(), random.random()]
        data.append(data_point)
        labels.append(1 if data_point[0] >= THRESHOLDS[0] and data_point[1] >= THRESHOLDS[1] else 0)
    
    return (data, labels)

def main():
    """Train a small simple model on fake data and store it.
    
    The inputs are points in 2-dimensional space with co-ordinates in the range [0, 1].
    The positive sets have both co-ordinates in the range [0.5, 1] (i.e. their in the top-right quadrant).
    """
    
    print("\nGenerating Data...\n")
    
    # Randomly produce training and test data.
    random.seed()
    (train_data, train_labels) = make_data(TRAIN_SIZE)
    (test_data, test_labels) = make_data(TEST_SIZE)
    
    # Define the neural network's architecture.
    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Dense(NEURONS, activation = ACTIVATION),
        tensorflow.keras.layers.Dense(CLASSES)
    ])
    
    # Compile the model.
    model.compile(
        loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = METRICS
    )
    
    # Train and test the model.
    print("\nTraining...")
    model.fit(train_data, train_labels, epochs = EPOCHS)
    print("\nTesting...")
    model.evaluate(test_data, test_labels)
    
    # Save the model on disk.
    print("\nSaving model to", FILE_NAME)
    model.save(FILE_NAME)

# Allows import without run.
if __name__ == "__main__":
    main()

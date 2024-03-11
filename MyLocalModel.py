import tensorflow as tf
import numpy as np
from clearml import PipelineDecorator, Task
import os

# Set ClearML API host and API key
os.environ['CLEARML_API_HOST'] = 'https://api.community.clear.ml'
os.environ['CLEARML_API_KEY'] = 'Y0ELY1U3XT27XIVQVZIQ'

# Initialize ClearML task
task = Task.init(project_name="MLOps Example", task_name="MLOps with ClearML")

# Define data loading function
@PipelineDecorator.component(cache=True)
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, y_train, x_test, y_test

# Define data preprocessing function
@PipelineDecorator.component(cache=True)
def preprocess_data(data):
    x_train, y_train, x_test, y_test = data
    # Reshape data to 2D arrays
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    return x_train, y_train, x_test, y_test

# Define feature engineering function (dummy function for demonstration)
@PipelineDecorator.component(cache=True)
def feature_engineering(data):
    # Dummy feature engineering: Add noise to the data
    x_train, y_train, x_test, y_test = data
    x_train += np.random.normal(loc=0, scale=0.1, size=x_train.shape)
    x_test += np.random.normal(loc=0, scale=0.1, size=x_test.shape)
    return x_train, y_train, x_test, y_test

# Define data transformation function (dummy function for demonstration)
@PipelineDecorator.component(cache=True)
def data_transform(data):
    # Dummy data transformation: Apply min-max scaling
    x_train, y_train, x_test, y_test = data
    x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
    x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))
    return x_train, y_train, x_test, y_test

# Define model selection function
@PipelineDecorator.component(cache=True)
def select_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Define model evaluation function
@PipelineDecorator.component(cache=True)
def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')
    return loss, accuracy, model

# Define the pipeline logic
@PipelineDecorator.pipeline(name='MLOps Pipeline', project='MLOps Example', version='1.0')
def mlops_pipeline():
    data = load_data()
    data = preprocess_data(data)
    data = feature_engineering(data)
    data = data_transform(data)
    best_accuracy = 0
    best_model = None
    for _ in range(3):  # Evaluate three different models
        model = select_model()
        _, accuracy, model = evaluate_model(model, data[2], data[3])  # Passing x_test, y_test for evaluation
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    return data, best_model, best_accuracy

# Run the pipeline locally
if __name__ == '__main__':
    # Run the pipeline on the current machine, for local debugging
    PipelineDecorator.run_locally()

    # Execute the pipeline logic
    data, best_model, best_accuracy = mlops_pipeline()

    # Log metrics to ClearML
    task.upload_artifact('model.h5', best_model)  # Upload best model artifact
    task.get_logger().report_scalar("best_accuracy", "scalar", best_accuracy, step=0)

    # Close ClearML task
    task.close()

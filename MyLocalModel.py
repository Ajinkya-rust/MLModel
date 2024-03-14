import os
import tensorflow as tf
import numpy as np
from clearml import PipelineDecorator, Task


# Define data loading function
@PipelineDecorator.component(cache=True, execution_queue="default")
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Use a subset of the dataset for debugging
    sample_size = 1000
    x_train, y_train = x_train[:sample_size], y_train[:sample_size]
    x_test, y_test = x_test[:sample_size], y_test[:sample_size]
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, y_train, x_test, y_test

# Define data preprocessing function
@PipelineDecorator.component(cache=True, execution_queue="default")
def preprocess_data(data):
    x_train, y_train, x_test, y_test = data
    # Reshape data to 2D arrays
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    return x_train, y_train, x_test, y_test

# Define feature engineering function (dummy function for demonstration)
@PipelineDecorator.component(cache=True, execution_queue="default")
def feature_engineering(data):
    # Dummy feature engineering: Add noise to the data
    x_train, y_train, x_test, y_test = data
    x_train += np.random.normal(loc=0, scale=0.1, size=x_train.shape)
    x_test += np.random.normal(loc=0, scale=0.1, size=x_test.shape)
    return x_train, y_train, x_test, y_test

# Define data transformation function (dummy function for demonstration)
@PipelineDecorator.component(cache=True, execution_queue="default")
def data_transform(data):
    # Dummy data transformation: Apply min-max scaling
    x_train, y_train, x_test, y_test = data
    x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
    x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))
    
    # Reshape data to match the expected input shape of the model
    x_train = x_train.reshape(x_train.shape[0], 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 28, 28)
    
    return x_train, y_train, x_test, y_test

@PipelineDecorator.component(cache=True, execution_queue="default")
def select_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Define the pipeline logic
@PipelineDecorator.pipeline(name='MLOps Pipeline', project='MLOps Example', version='1.0')
def mlops_pipeline(do_stuff: bool):
    if do_stuff:
        data = load_data()
        data = preprocess_data(data)
        data = feature_engineering(data)
        data = data_transform(data)
        best_model = select_model()
        
        # Save best model locally
        model_path = os.path.join(os.getcwd(), 'best_model.h5')
        best_model.save(model_path)
        
        return data, model_path
    else:
        print("Not doing anything in the pipeline as 'do_stuff' is set to False.")

# Run the pipeline locally
if __name__ == '__main__':
    # Execute the pipeline logic with do_stuff=True
    PipelineDecorator.run_locally()
    data, model_path = mlops_pipeline(do_stuff=True)

        # Initialize ClearML task
    task = Task.init(project_name="MLOps Example", task_name="MLOps with ClearML")

    # Upload best model artifact to ClearML
    task.upload_artifact('best_model.h5', model_path)

    # Close ClearML task
    task.close()

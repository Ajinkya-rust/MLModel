import os
import tensorflow as tf
import numpy as np
from clearml import PipelineDecorator, Task


# Define data loading function
@PipelineDecorator.component(cache=True)
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
    
    # Reshape data to match the expected input shape of the model
    x_train = x_train.reshape(x_train.shape[0], 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 28, 28)
    
    return x_train, y_train, x_test, y_test

# Define component to select and compile the model
@PipelineDecorator.component(cache=True)
def select_and_compile_model():
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

# Define component to train the model
@PipelineDecorator.component(cache=True)
def train_model(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    return model

# Define component to evaluate the model
@PipelineDecorator.component(cache=True)
def evaluate_model(x_test, y_test, model):
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Log accuracy metric using ClearML
    task = Task.current_task()
    if task:
        # Provide an explicit iteration number or use None if not applicable
        iteration = None
        task.get_logger().report_scalar(title="Metrics", series="Accuracy", value=accuracy, iteration=iteration)

    return loss, accuracy

# Define the pipeline logic
@PipelineDecorator.pipeline(name='MLOps Pipeline', project='MLOps Example', version='1.0')
def mlops_pipeline(do_stuff: bool):
    if do_stuff:
        data = load_data()
        data = preprocess_data(data)
        data = feature_engineering(data)
        data = data_transform(data)
        x_train, y_train, x_test, y_test = data
        
        # Select and compile the model
        best_model = select_and_compile_model()

        # Train the model
        trained_model = train_model(x_train, y_train, x_test, y_test, best_model)

        # Evaluate the model
        test_loss, test_accuracy = evaluate_model(x_test, y_test, trained_model)
        
        # Save the best model locally
        model_path = os.path.join(os.getcwd(), 'best_model.h5')
        trained_model.save(model_path)
        
        return data, model_path, test_loss, test_accuracy
    else:
        print("Not doing anything in the pipeline as 'do_stuff' is set to False.")
        
# Run the pipeline locally
if __name__ == '__main__':
    # Execute the pipeline logic with do_stuff=True
    PipelineDecorator.run_locally()
    
    # Call the mlops_pipeline function
    pipeline_result = mlops_pipeline(do_stuff=True)
    
    if pipeline_result:
        data, model_path, test_loss, test_accuracy = pipeline_result
        # Initialize ClearML task
        task = Task.init(project_name="MLOps Example", task_name="MLOps with ClearML")

        # Upload best model artifact to ClearML
        task.upload_artifact('best_model.h5', model_path)

        # Execute the pipeline remotely with the default execution queue
        task.execute_remotely(queue_name="default")

        # Close ClearML task
        task.close()
    else:
        print("Pipeline did not return any result.")

import tensorflow as tf
from clearml import PipelineDecorator
import os
from clearml import Task, PipelineDecorator, Model

@PipelineDecorator.component(cache=True, execution_queue="default")
def step(size: int):
    import numpy as np
    return np.random.random(size=size)


@PipelineDecorator.pipeline(
    name='ingest',
    project='My Local Model',
    version='0.1'
)

def pipeline_logic(do_stuff: bool):
    if do_stuff:
        return step(size=42)

if __name__ == '__main__':
    # run the pipeline on the current machine, for local debugging
    # for scale-out, comment-out the following line (Make sure a
    # 'services' queue is available and serviced by a ClearML agent
    # running either in services mode or through K8S/Autoscaler)
    PipelineDecorator.run_locally()

    pipeline_logic(do_stuff=True)
	
# Initialize a ClearML task
task = Task.init(project_name="My Local Model", task_name="Training")
# Tag the task as local
task.add_tags(['Local'])

# Log hyperparameters
task_params = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 10
}
task.connect(task_params)

task.set_model_config({
    'architecture': 'Sequential',
    'layers': [
        {'type': 'Flatten', 'input_shape': (28, 28)},
        {'type': 'Dense', 'units': 128, 'activation': 'relu'},
        {'type': 'Dropout', 'rate': 0.2},
        {'type': 'Dense', 'units': 10, 'activation': 'softmax'}
    ],
    'optimizer': 'Adam',
    'loss_function': 'SparseCategoricalCrossentropy',
    'metrics': ['Accuracy'],
    'hyperparameters': task_params
})

def load_data():
    # Load your dataset here
    # For example, you can use TensorFlow datasets or load data from files
    
    # Placeholder example using TensorFlow's MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    return x_train, y_train, x_test, y_test
	
	
	
def preprocess_data(x_train, x_test):
    # Implement your data preprocessing logic here
    # This can include scaling, normalization, feature extraction, etc.
    
    # Placeholder example: scaling the data
    x_train_scaled = x_train * 2  # Scaling by a factor of 2
    x_test_scaled = x_test * 2
    
    return x_train_scaled, x_test_scaled

# Load and preprocess data
# (Assuming you have a function `load_data()` and `preprocess_data()` defined)
x_train, y_train, x_test, y_test = load_data()
x_train, x_test = preprocess_data(x_train, x_test)

# Build and compile model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=task_params['num_epochs'], batch_size=task_params['batch_size'])

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

# Log metrics
task.upload_artifact('model.h5', model)  # Upload model artifact
task.get_logger().report_scalar("test_loss", "scalar", loss, iteration=task_params['num_epochs'])
task.get_logger().report_scalar("test_accuracy", "scalar", accuracy, iteration=task_params['num_epochs'])

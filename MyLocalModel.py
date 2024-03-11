import tensorflow as tf
from clearml import PipelineDecorator, Task
from sklearn.model_selection import train_test_split


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus: 
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define pipeline components using the @PipelineDecorator.component decorator
@PipelineDecorator.component(cache=True, execution_queue="default")
def load_data():
    # Placeholder example: Load dataset (e.g., MNIST)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalize pixel values to range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, y_train, x_test, y_test

@PipelineDecorator.component(cache=True, execution_queue="default")
def split_data(data):
    # Convert numpy arrays to TensorFlow datasets
    dataset = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
    
    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=len(data[0])).batch(batch_size=32)
    
    # Split dataset into training and testing sets
    train_size = int(0.8 * len(data[0]))
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    
    # Convert datasets back to numpy arrays
    x_train, y_train = [], []
    x_test, y_test = [], []
    for x, y in train_dataset:
        x_train.append(x.numpy())
        y_train.append(y.numpy())
    for x, y in test_dataset:
        x_test.append(x.numpy())
        y_test.append(y.numpy())
    
    # Concatenate batches
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)
    
    return x_train, x_test, y_train, y_test

@PipelineDecorator.component(cache=True, execution_queue="default")
def select_model():
    # Placeholder example: Model selection
    # For simplicity, returning a predefined model
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Define the pipeline logic using the @PipelineDecorator.pipeline decorator
@PipelineDecorator.pipeline(
    name='Development',
    project='My Local Model',
    version='0.1'
)
def pipeline_logic(do_stuff: bool):
    if do_stuff:
        # Call pipeline components in sequence
        data = load_data()
        x_train, x_test, y_train, y_test = split_data(data)
        model = select_model()
        return x_train, y_train, x_test, y_test, model

if __name__ == '__main__':
    import os
    if os.getenv('CI') != 'true':
        # Run the pipeline on the current machine, for local debugging
        PipelineDecorator.run_locally()

        # Execute the pipeline logic with do_stuff=True
        x_train, y_train, x_test, y_test, model = pipeline_logic(do_stuff=True)

        print('Data Collection, Preprocessing, and Transformation Steps Completed')

        # Initialize a ClearML task for each step of the pipeline
        tasks = []
        for step_name in ['Model Training']:
            tasks.append(Task.init(project_name="My Local Model", task_name=step_name))

    # Log hyperparameters
    task_params = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 10
    }
    tasks[0].connect(task_params)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=task_params['num_epochs'], batch_size=task_params['batch_size'])

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')

    # Log metrics
    tasks[0].upload_artifact('model.h5', model)  # Upload model artifact
    tasks[0].get_logger().report_scalar("test_loss", "scalar", loss, iteration=task_params['num_epochs'])
    tasks[0].get_logger().report_scalar("test_accuracy", "scalar", accuracy, iteration=task_params['num_epochs'])

    print('Model Training Preparation, Model Selection, and Model Training Steps Completed')


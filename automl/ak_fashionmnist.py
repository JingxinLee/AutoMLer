import numpy as np
import autokeras as ak
from keras.datasets import fashion_mnist

# Prepare the data.
(x_train, y_classification), (x_test, y_test) = fashion_mnist.load_data()
x_image = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

x_structured = np.random.rand(x_train.shape[0], 100)
y_regression = np.random.rand(x_train.shape[0], 1)

# Build model and train.
automodel = ak.AutoModel(
   inputs=[ak.ImageInput(),
           ak.StructuredDataInput()],
   outputs=[ak.RegressionHead(metrics=['mae']),
            ak.ClassificationHead(loss='categorical_crossentropy',
                                  metrics=['accuracy'])],
    max_trials=10)

automodel.fit([x_image, x_structured],
              [y_regression, y_classification],
              validation_split=0.2, epochs=10)



# Modified-CNN-Model
Modified the CNN model and Compared with Limited Pretrained model


## Methodology

### Modelling
This code is based on the classification of images from the dataset into various classes provided. The dataset consists of 2 classes of images of various Accident and non-accident. The dataset consists of training, testing and validation images. The code starts off by importing the necessary libraries like NumPy, pandas, matplotlib, TensorFlow and keras. After that we define the batch specifications and load the training, validation and testing datasets. We also define the class names for the dataset. After that we configure the dataset for better performance by using Autotune. Then we create a model using the Sequential model. The model consists of multiple convolutional layers, max pooling layers and dropout layers.


![alt text](https://github.com/Tesipog/Modified-CNN-Model/blob/main/Flowchart.png?raw=true)

We compile the model using the Adam optimizer and the sparse categorical cross entropy loss function. After compiling the model, we fit the training data and get the accuracy and loss values for both the training and validation data. We plot the accuracy and loss values for both the training and validation data for better understanding. We also save the model so that we can use it later. After that we evaluate the model on the testing data and plot the accuracies for the test data. We also plot the model using the plot_model () function. At the end we use the model for inference on a single image with any size. We resize the image to the specified size and get the prediction from the model. We then print the predictions and the predefined class labels. We also print the classification report and the accuracy score for the testing data.
## Comparision of our model with other models
![alt text](https://github.com/Tesipog/Modified-CNN-Model/blob/main/comparision.png?raw=true)
![alt text](https://github.com/Tesipog/Modified-CNN-Model/blob/main/image.png?raw=true)


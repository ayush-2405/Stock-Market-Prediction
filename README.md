
# STOCK MARKET PREDICTION

#### The Stock Market is rather very dynamic and complex in nature.This project involves determining the future value of a company stock by using time series data. Time series is a series of data points indexed in time order.

#### This project involves using an LSTM network to predict the future value of stocks for a bluechip Indian Company- TATA CONSULTANCY SERVICES

#### Data of the opening price,closing price , highest price and lowest price is collected from 25th August 2004 to 29th June 2024. The dataset is taken from Yahoo Finance- [https://finance.yahoo.com/quote/TATAPOWER.BO/history?period1=946871100&period2=1714465597](https://finance.yahoo.com/quote/TCS.BO/history/?period1=1093405500&period2=1719682008)

#### Stock market predictors have a high scope as it can guide investors about the price of a stock in the future which can help them make decisions with respect to buying or selling the stock.


## Data Reading and Visualization

#### 1. Importing Libraries: 
The script imports necessary libraries for numerical computation (numpy), data visualization (matplotlib.pyplot), data handling (pandas), neural network implementation (torch, torch.nn), and data preprocessing (MinMaxScaler) 

#### 2. Reading Data: 
The script reads a CSV file containing stock data using pandas.read_csv(), keeping only the 'Date' and 'Adj Close' columns.

#### 3. Checking for GPU: 
It checks if a CUDA-enabled GPU is available using torch.cuda.is_available(). If available, it sets the device to GPU ('cuda:0'), otherwise to CPU ('cpu').

#### 4. Date Conversion: 
The script converts the 'Date' column in the dataset to datetime format using pd.to_datetime().

#### 4.Plotting the Data: 
It creates a plot using matplotlib to visualize the adjusted closing prices over time. It sets the figure size, plots the 'Date' column on the x-axis and the 'Adj Close' column on the y-axis, sets the title, xlabel, ylabel, and grid, and then displays the plot.

## Prepare DataFrame for LSTM Model
#### Overview
The Python function prepare_dataframe_for_lstm() prepares a DataFrame for training an LSTM (Long Short-Term Memory) model for time series forecasting. It creates lag features from the 'Adj Close' column, which represent the past values of the target variable.

#### Function Explanation
#### Input Parameters:

#### 1. df: 
The input DataFrame containing stock data with a 'Date' column and an 'Adj Close' column.

#### 2. n_steps: 
The number of lag steps(lookbacks) to create for each observation.

#### 3. Deep Copy of DataFrame: 
The function creates a deep copy of the input DataFrame to avoid modifying the original data.

#### 4. Index Setting: 
It sets the 'Date' column as the index of the DataFrame.

#### 5. Creating Lag Features: 
For each lag step from 1 to n_steps, it creates a new column 'Adj Close(t-i)', where i represents the lag step, containing the adjusted close price shifted by i time steps.

#### 6. Removing NaN Values: 
df.dropna() drops rows with NaN values resulting from the shifting operation.

#### 7. Return Value: 
The function returns the modified DataFrame with lag features.
## Data Preprocessing for LSTM Model Training

#### Overview
This code snippet preprocesses the prepared DataFrame for training an LSTM (Long Short-Term Memory) model. It involves converting the DataFrame to a NumPy array, scaling the data using MinMaxScaler, and splitting it into training and testing sets.

#### Code Explanation
#### 1. DataFrame to NumPy Array: 
The DataFrame shifted_ds is converted to a NumPy array shifted_ds_in_np using the to_numpy() method.

#### 2. Data Scaling: 
The data is scaled using MinMaxScaler from sklearn.preprocessing. This step ensures that all features are within the same range (0 to 1), which is beneficial for neural network training.

#### 4. Feature-Target Split: 
The input features X are extracted from the scaled data, excluding the first column which contains the target variable. The target variable y is extracted from the first column.

#### 5. Data Transformation: 
The features X are reversed along the time axis using np.flip() and then copied using deepcopy to ensure a new array is created.

#### 6. Train-Test Split: 
The dataset is split into training and testing sets using a 90-10 split ratio. The split_index is calculated based on the length of the features array X, and the arrays are sliced accordingly to obtain X_train, X_test, Y_train, and Y_test.
## Reshaping Data for LSTM Model Input

#### Overview
This step reshapes the feature and target arrays to match the input requirements of an LSTM (Long Short-Term Memory) model. LSTM models in PyTorch expect input data in a specific shape, typically in the format (batch_size, sequence_length, num_features). The reshaping process ensures that the input data is compatible with the model architecture.

#### Code Explanation

#### 1. Reshaping Features (X): 
The training and testing feature arrays X_train and X_test are reshaped to have three dimensions: (batch_size, sequence_length, num_features). Here, batch_size represents the number of samples, sequence_length represents the number of time steps (lookback), and num_features represents the number of features.

The '-1' in the reshaping indicates that NumPy should automatically calculate the size of the first dimension based on the total number of elements and the other dimensions. This is a common practice when you want to reshape an array while preserving the total number of elements.

'1' represents the number of features for each time step. In this case, each time step contains a single feature (the adjusted close price).

Therefore, after reshaping, X_train becomes a three-dimensional array with the shape (batch_size, lookback, 1), where batch_size is automatically determined based on the size of the original array.

#### 2. Reshaping Targets (y): 
Similarly, the training and testing target arrays y_train and y_test are reshaped to have two dimensions: (batch_size, 1). This ensures that each target value corresponds to one time step.
## Converting Data to PyTorch Tensors

The code snippet converts the NumPy arrays representing training and testing data into PyTorch tensors using the torch.tensor() function. It ensures that the data is in the correct format and data type(float) for training a neural network using PyTorch. 
## Creating a Custom Dataset for Time Series Data

#### Overview
The code snippet defines a custom dataset class TimeSeriesDataset for handling time series data in PyTorch. It enables the creation of PyTorch Dataset objects from input features and target variables, facilitating data loading and batching during training and testing of neural network models.

#### Code Explanation
#### 1. Dataset Definition: 
The TimeSeriesDataset class inherits from torch.utils.data.Dataset, making it compatible with PyTorch's dataset handling utilities.

#### 2. Initialization: 
The constructor __init__ initializes the dataset with input features X and target variables y.

#### 3. Length Method: 
The __len__ method returns the total number of samples in the dataset, which is the length of the input features (X).

#### 4. Get Item Method: 
The __getitem__ method retrieves a sample at index i from the dataset. It returns a tuple containing the input feature and its corresponding target variable.

#### 5. Dataset Instances: 
Instances of the TimeSeriesDataset class are created for both the training and testing data, named train_dataset and test_dataset, respectively.
## Creating Data Loaders for Training and Testing

#### Overview
The code snippet creates data loaders using PyTorch's DataLoader class to efficiently load and batch data for training and testing neural network models. Data loaders are essential for handling large datasets and enabling efficient processing during model training.

#### Code Explanation

#### 1. Data Loader Initialization: 
Two data loaders, train_loader and test_loader, are created using the DataLoader class from torch.utils.data.

#### 2. Batch Size: 
The batch_size parameter specifies the number of samples to include in each batch during training and testing.

#### 3. Shuffling: 
For the training data loader (train_loader), shuffle=True is specified to shuffle the data before each epoch. Shuffling helps prevent the model from learning spurious correlations due to the order of data samples. For the testing data loader (test_loader), shuffle=False ensures that the data is not shuffled, maintaining the original order of samples for evaluation.
## LSTM Model Definition
#### Overview
The code snippet defines an LSTM (Long Short-Term Memory) neural network model using PyTorch's nn.Module class. LSTMs are a type of recurrent neural network (RNN) commonly used for sequence modeling tasks such as time series forecasting.

#### Code Explanation
#### 1. Model Initialization: 
The LSTM class inherits from nn.Module and defines the LSTM model architecture.

(a) Initialization Method: The __init__ method initializes the model parameters including input size, hidden size, and number of stacked layers.

(b) input_size: The number of features in the input data.

(c) hidden_size: The number of features in the hidden state of the LSTM cell.

(d)num_stacked_layers: The number of LSTM layers stacked on top of each other.

(e) Inside the __init__ method, an LSTM layer (self.lstm) is defined using nn.LSTM, with the specified input size, hidden size, and number of stacked layers.

#### 4. Forward Method: 
The forward method defines the forward pass computation of the model.

(a) It takes input x and initializes the initial hidden state and cell state (h0 and c0) with zeros.

(b) The input is passed through the LSTM layer, and the final output of the last time step is obtained.

(c)The output is then passed through a fully connected (linear) layer (self.fc) to produce the final output prediction.

#### 5. Model Instantiation: 
An instance of the LSTM class is created with specified parameters (input_size=1, hidden_size=16, num_stacked_layers=1).

#### 6. Device Assignment: 
The model is moved to the specified device (CPU or GPU) using the to(device) method.

#### 7. Model Representation: 
Printing model provides a representation of the model architecture.
## Training and Validation
#### Overview
The code snippet defines functions for training and validating an LSTM model for time series forecasting using PyTorch. It includes functions for training one epoch (train_one_epoch()) and validating one epoch (validate_one_epoch()), along with the main training loop that iterates over multiple epochs.

#### Code Explanation

#### 1. Training One Epoch (train_one_epoch()):
(a) Sets the model to training mode using model.train(True).

(b) Iterates over batches of data from the training data loader (train_loader).

(c) Computes the output predictions using the model (output) and calculates the loss between the predictions and the ground truth (y_batch) using the specified loss function (loss_function).

(d) Backpropagates the loss and updates the model parameters using the specified optimizer (optimizer).

(e) Prints the average loss across batches every 100 batches.

#### 2. Validate One Epoch (validate_one_epoch()):

(a) Sets the model to evaluation mode using model.train(False).

(b) Iterates over batches of data from the validation data loader (test_loader).

(c) Computes the output predictions using the model (output) without gradient computation (torch.no_grad()) and calculates the validation loss.

(d) Prints the validation loss.

#### 3. Main Training Loop:

(a) Defines the hyperparameters such as learning rate (learning_rate), number of epochs (num_epochs), loss function (loss_function), and optimizer (optimizer).

(b) Iterates over the specified number of epochs.

(c) Calls train_one_epoch() to train the model for one epoch.

(d) Calls validate_one_epoch() to validate the model for one epoch.
## Visualization of Actual and Predicted Close Prices

#### Overview
The code snippet utilizes the trained LSTM model to predict close prices for the training data and visualizes both the actual and predicted close prices using matplotlib.

#### Code Explanation

#### 1. Prediction with Trained Model:

(a) The with torch.no_grad() context manager ensures that gradient calculations are disabled during inference, reducing memory consumption and speeding up computation.
(b) Predictions (predicted) are obtained by passing the training features (X_train) to the trained model (model) on the specified device (device). The predictions are then moved to the CPU and converted to a NumPy array for visualization.

#### 2. Plotting Actual and Predicted Close Prices:

(a) Matplotlib is used to create a plot with a figure size of 15x7 inches.

(b) The actual close prices (y_train) and predicted close prices (predicted) are plotted on the same graph.

(c) Labels for the x-axis (Day) and y-axis (Close) are specified.
(d) A legend is added to distinguish between the actual and predicted close prices.
(e) The plot is displayed using plt.show().


## Inverse Transforming Predictions and Ground Truth Values

#### Overview

The code snippet performs inverse transformations on the predicted and ground truth values to convert them back to their original scale. It reverses the scaling applied during data preprocessing to obtain predictions and ground truth values in their original units.

#### Code Explanation

#### 1. Inverse Transformation of Predictions:

(a) The predicted values (predicted) are flattened and stored in the train_predictions variable.

(b) A NumPy array dummies of zeros is initialized with dimensions (X_train.shape[0], lookback+1), where lookback is the number of time steps.

(c) The first column of dummies is assigned the values from train_predictions.

(d) The scaler.inverse_transform() function is used to perform an inverse transformation on dummies, reverting the scaling applied during preprocessing.

(e) The inverse-transformed values are stored back in train_predictions.

#### 2. Inverse Transformation of Ground Truth Values:

(a) Similarly, the ground truth values (y_train) are flattened and stored in dummies[: , 0].

(b) The same procedure as above is followed to inverse-transform dummies and obtain new_y_train, which contains the ground truth values in their original scale.
## Visualization of Actual and Predicted Close Prices (Test Set)

The code snippet utilizes the predicted close prices (test_predictions) and the actual close prices (new_y_test) on the test set to create a visualization comparing the two. The plot displays how well the model's predictions align with the actual values.

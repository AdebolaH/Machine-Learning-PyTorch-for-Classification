<html><head></head><body>#!/usr/bin/env python
# coding: utf-8

# # Use PyTorch to Predict Hotel Cancellations
# 
# - [View Solution Notebook](./solutions.html)
# - [View Project Page](https://www.codecademy.com/)

# **Setup - Import libraries**

# In[2]:


import pandas as pd
import numpy as np


# ## Task Group 1 - Import and Inspect
# 
# The file `&#39;datasets/resort_hotel_bookings.csv&#39;` contains a subset of a [real-world dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) containing reservation and cancellation data for a resort hotel. 
# 
# Your goal in this project is build and train a neural network to predict if a customer will cancel their hotel booking reservation based on data including the booking dates, average daily cost, number of adults/children/babies, duration of stay, and so forth.

# ### Task 1
# 
# Begin by importing the CSV file to a pandas DataFrame named `hotels`.
# 
# Preview the first five rows using the `.head()` method.

# In[3]:


import pandas as pd

# Load the CSV file into a DataFrame
hotels = pd.read_csv(&#39;datasets/resort_hotel_bookings.csv&#39;)

# Display the first five rows
hotels.head()


# <details><summary style="display:list-item; font-size:16px; color:blue;">Here&#39;s a quick summary of the columns</summary>
# 
# - **is_canceled**: Whether the booking was canceled (1) or kept (0)
# - **lead_time**: Number of days between booking date and arrival date
# - **arrival_date_year**: Year of arrival date
# - **arrival_date_month**: Month of arrival date
# - **arrival_date_week_number**: Week number of arrival date
# - **arrival_date_day_of_month**: Day of the month of arrival date
# - **stays_in_weekend_nights**: Number of weekend nights booked (Sat-Sun)
# - **stay_in_week_nights**: Number of weekday nights booked (Mon-Fri)
# - **adults**: Number of adults
# - **children**: Number of children
# - **babies**: Number of babies
# - **meal**: Type of meal booked (Undefined/SC, BB, HB, or FB)
# - **country**: Country of origin of the booker
# - **market_segment**: Market segment (TA - travel agent, TO - tour operators)
# - **distribution_channel**: Booking distribution channel (TA - travel agent, TO - tour operators)
# - **is_repeated_guest**: Is this a repeated guest (1) or not (0)
# - **previous_cancellations**: The number of previous bookings canceled by the customer
# - **previous_bookings_not_canceled**: The number of previous bookings not canceled by the customer
# - **reserved_room_type**: Room type reserved
# - **assigned_room_type**: Type of assigned room booked
# - **booking_changes**: Number of booking changes or modifications
# - **deposit_type**: Type of deposit to guarantee booking (No Deposit, Non Refund, or Refundable)
# - **agent**: ID of the travel agency that made the booking
# - **company**: ID of the company that made the booking
# - **days_in_waiting_list**: Number of days booking was waitlisted before confirmation
# - **customer_type**: The customer type of booking (Contract, Group, Transient, or Transient-party)
# - **adr**: The average daily rate (cost) of the booking
# - **required_car_parking_spaces**: Number of parking spaces requested by the customer
# - **total_of_special_requests**: Number of special requests by the customer
# - **reservation_status**: The last reservation status (Canceled, Check-Out, No-Show)
# - **reservation_status_date**: The date of the last reservation status

# ### Task 2
# 
# Let&#39;s explore the data types and whether any data is missing.
# 
# Use the `.info()` method on the `hotels` DataFrame to inspect the data.

# In[4]:


# Display column info, data types, and missing values
hotels.info()


# <details><summary style="display:list-item; font-size:16px; color:blue;">What do we notice about the dataset under inspection?</summary>
# 
# There are 31 columns and 40,060 total observations in our dataset. The majority of columns do not have missing values.
# 
# However, we do notice that: 
# - the `agent` and `company` columns seem to have missing values that need to be addressed
# - the `country` column has a couple of missing values as well
# 
# There are a variety of data types represented. To work with a neural network, we&#39;ll have to address any non-numeric columns in our data preparation.

# ### Task 3
# 
# Let&#39;s now explore the cancellation column we want to predict.
# 
# Use the `.value_counts()` method on the `is_canceled` column to count the number **and** the percentage of overall cancellations. 

# In[5]:


# Count of each value
hotels[&#39;is_canceled&#39;].value_counts()

# Percentage of each value
hotels[&#39;is_canceled&#39;].value_counts(normalize=True)


# <details><summary style="display:list-item; font-size:16px; color:blue;">What do we notice about the number of cancellations?</summary>
# 
# The number of cancellations is much lower than the number of non-cancellations (27.8% canceled vs 72.2% did not cancel). 
# 
# We&#39;ll need to take this imbalance into account when we evaluate our model. For example, a naive model could simply predict every booking will **not be canceled** and achieve a decent accuracy of 72.2%.

# ### Task 4
# 
# The `reservation_status` column tells us if the booking was canceled while also telling us if the customer was a no-show.
# 
# We need to be sure to exclude this column from the training set, otherwise this information will be _leaked_ to our model resulting in inaccurate performance. 
# 
# First, let&#39;s take a quick look at the values in this column.
# 
# Use the `.value_counts()` method on the `reservation_status` column to count the number **and** the percentage of overall cancellations. 

# In[10]:


# Count of each value
hotels[&#39;reservation_status&#39;].value_counts()

# Percentage of each value
hotels[&#39;reservation_status&#39;].value_counts(normalize=True)


# <details><summary style="display:list-item; font-size:16px; color:blue;">What do we notice about the reservation_status column?</summary>
# 
# The number of no-shows is extremely small and consists of only 291 (or 0.7%) of observations in the dataset.
# 
# Later on, we&#39;ll look at creating a multiclass model to predict no-show in addition to canceled.

# ### Task 5
# 
# Before diving into building a model, let&#39;s continue to explore the dataset. It&#39;s important to understand how different columns interact with cancellations to guide our model structure! 
# 
# For example, cancellations might be higher in the summer months (June - September) and lower in the winter months (November - January).
# 
# Use the `.groupby()` method to group the data by the `arrival_date_month` column and apply the `.mean()` aggregation function on the `is_canceled` column. This will return the percent of reservations cancelled in each month.
# 
# Then, use the `.sort_values()` method to sort the percentages from lowest to highest.

# In[11]:


# Group by month and calculate mean cancellation rate
grouped = hotels.groupby(&#39;arrival_date_month&#39;)[&#39;is_canceled&#39;].mean()

# Sort results from lowest to highest
grouped_sorted = grouped.sort_values()

grouped_sorted


# <details><summary style="display:list-item; font-size:16px; color:blue;">What do we notice about the percentage of cancellations by month?</summary>
# 
# It looks like our intuition was correct! Winter and spring have the lowest cancellation percentages, while summer and fall have the highest. This information can be very useful for our model!

# It might be useful to do more exploratory data analysis to gain additional insights about hotel cancellations. For example, additional analysis may help you select better features to train the model on and exclude features that might seem irrelevant. But for now, let&#39;s move on to cleaning and preparing the data.

# ## Task Group 2 - Data Cleaning and Preparation
# 
# In this section, we&#39;ll encode categorical data for use in our neural networks.

# ### Task 6
# 
# To get a sense of the categorical data in the dataset, let&#39;s start by previewing the first five rows of all columns with `object` datatype.
# 
# Create a list named `object_columns` containing only the names of the object columns (except for the reservation status columns). Select those columns from `hotels` and preview the first `5` rows.

# In[12]:


# List of selected object columns (excluding reservation status columns)
object_columns = [
    &#39;arrival_date_month&#39;, &#39;meal&#39;, &#39;country&#39;, &#39;market_segment&#39;,
    &#39;distribution_channel&#39;, &#39;reserved_room_type&#39;, &#39;assigned_room_type&#39;,
    &#39;deposit_type&#39;, &#39;customer_type&#39;
]

# Preview the first five rows of these columns
hotels[object_columns].head()


# <details><summary style="display:list-item; font-size:16px; color:blue;">Hint: Preview the first five rows subset by the object columns</summary>
# 
# Here&#39;s how we can subset the DataFrame by the object columns and preview the first five rows:
# 
# ```py
# object_columns = [&#39;arrival_date_month&#39;, &#39;meal&#39;, &#39;country&#39;, &#39;market_segment&#39;, &#39;distribution_channel&#39;, &#39;reserved_room_type&#39;, &#39;assigned_room_type&#39;, &#39;deposit_type&#39;, &#39;customer_type&#39;]
# hotels[object_columns].head()
# ```
# 
# Additionally, it might be helpful to explore the categorical data in each object column using the `.value_counts()` method.
# 
# </details>

# ### Task 7
# 
# Typically, we don&#39;t want to use every column in training. For example, we may want to drop columns with many missing values or columns that are irrelevant to our prediction task.
# 
# Drop any columns you don&#39;t want to use to train a cancellation model (do not remove the target label column). Feel free to open our Hint to review the columns we chose to drop in our solution.
# 
# Note: We don&#39;t want to drop the `reservation_status` column from the dataset quite yet because we&#39;ll be using this column to train our multiclass neural network.

# In[13]:


drop_columns = [&#39;country&#39;, &#39;agent&#39;, &#39;company&#39;, &#39;reservation_status_date&#39;,
                &#39;arrival_date_week_number&#39;, &#39;arrival_date_day_of_month&#39;, &#39;arrival_date_year&#39;]

hotels = hotels.drop(labels=drop_columns, axis=1)


# <details><summary style="display:list-item; font-size:16px; color:blue;">Hint: Drop columns in the dataset not used for training.</summary>
# 
# Here&#39;s a list of potential features to drop. Feel free to experiment on your own by dropping or keeping columns you might believe may contribute to training.
# 
# ```py
# drop_columns = [&#39;country&#39;, &#39;agent&#39;, &#39;company&#39;, &#39;reservation_status_date&#39;,
#                 &#39;arrival_date_week_number&#39;, &#39;arrival_date_day_of_month&#39;, &#39;arrival_date_year&#39;]
# 
# hotels = hotels.drop(labels=drop_columns, axis=1)
# ```
# 
# Here&#39;s why we chose these columns:
# 
# - `country` - there are many countries that only appear a handful of times in the dataset which may make our model less generalizable and even discriminate against customers based on their country
# - `agent` - similar to `country`, there are many agents that only appear a handful of times which may make our model less generalizable (and there are many missing values!)
# - `company` - similar to `agent`, there are many companies that only appear a handful of times which may make our model less generalizable (and there are many missing values!)
# - `reservation_status_date` - tells us the date of the latest status change of the reservation which shouldn&#39;t be helpful and if anything may leak data
# - `arrival_date_week_number` - tells us the week of the year which may be too specific and prone to overfitting
# - `arrival_date_day_of_month` - tells us the day of the month which may be too specific and prone to overfitting
# - `arrival_date_year` - tells us the year of the booking which may not be helpful to predict future years
# 
# </details>

# ### Task 8
# 
# Next, let&#39;s encode the `meal` column which tells us which type of meal(s) the customer booked: 
# 
# - `Undefined` and `SC` correspond to no meal packages
# - `BB` corresponds to breakfast only
# - `HB` (half board) corresponds to breakfast + lunch or dinner
# - `FB` (full board) corresponds to breakfast, lunch, and dinner.
# 
# Label encode the `meal` column with a meaningful order (# of meals booked) using the following scheme:
# 
# - `Undefined` and `SC` to `0`
# - `BB` to `1`
# - `HB` to `2`
# - `FB` to `3` 

# In[14]:


hotels[&#39;meal&#39;] = hotels[&#39;meal&#39;].replace({&#39;Undefined&#39;: 0, &#39;SC&#39;: 0, &#39;BB&#39;: 1, &#39;HB&#39;: 2, &#39;FB&#39;: 3})


# ### Task 9
# 
# Let&#39;s prepare the rest of the categorical columns using one-hot encoding. 
# 
# Create a list named `one_hot_columns` containing the list of categorical column names (all the remaining categorical columns) to be one-hot encoded using the `pd.get_dummies()` method.
# 
# Preview the cleaned `hotels` DataFrame using the `.head()` method.

# In[15]:


one_hot_columns = [&#39;arrival_date_month&#39;, &#39;market_segment&#39;, &#39;distribution_channel&#39;,
                   &#39;reserved_room_type&#39;, &#39;assigned_room_type&#39;, &#39;deposit_type&#39;, &#39;customer_type&#39;]

hotels = pd.get_dummies(hotels, columns=one_hot_columns, dtype=int)
hotels.head()


# Perfect! It looks like we&#39;ve handled all of the categorical variables and prepared the DataFrame for training.
# 
# Note that the cleaned DataFrame now has 67 columns due to the additional columns created using one-hot encoding.

# ## Task Group 3 - Create Training and Testing Sets
# 
# Next, let&#39;s convert our dataset into PyTorch tensors and split them into training and testing sets.

# ### Task 10
# 
# Let&#39;s import the necessary PyTorch libraries and modules. 

# In[16]:


import torch
from torch import nn, optim


# ### Task 11
# 
# We need to start by separating our training features from the target labels.
# 
# Create a list named `train_features` that contains all of the feature names (column names excluding the target variables `is_canceled` and `reservation_status`).

# In[17]:


remove_cols = [&#39;is_canceled&#39;, &#39;reservation_status&#39;]
train_features = [x for x in hotels.columns if x not in remove_cols]


# <details><summary style="display:list-item; font-size:16px; color:blue;">Hint: Select training features.</summary>
# 
# ```py
# # Remove target columns
# remove_cols = [&#39;is_canceled&#39;, &#39;reservation_status&#39;]
# 
# # Select training features
# train_features = [x for x in hotels.columns if x not in remove_cols]
# ```
#  
# </details>

# ### Task 12
# 
# Using the list of training features in `train_features`, create `X` and `y` tensors:
# 
# - `X` contains the data values from the `train_features` columns
# - `y` contains the binary labels in the `is_canceled` column in `hotels`
# 
# Both `X` and `y` should have the float datatype.
# 
# Be sure to set the correct view of `y` using `.view(-1,1)`

# In[18]:


X = torch.tensor(hotels[train_features].values, dtype=torch.float)
y = torch.tensor(hotels[&#39;is_canceled&#39;].values, dtype=torch.float).view(-1, 1)


# <details><summary style="display:list-item; font-size:16px; color:blue;">Hint: Create X and y tensors.</summary>
# 
# When creating the tensors, be sure to extract the data values in the specified columns using `.values` as floats:
#     
# ```py
# X = torch.tensor(hotels[train_features].values, dtype=torch.float)
# y = torch.tensor(hotels[&#39;is_canceled&#39;].values, dtype=torch.float).view(-1,1)
# ```
#  
# </details>

# ### Task 13
# 
# Let&#39;s now split our data contained in `X` and `y` into training and testing sets.
# 
# Import the `train_test_split` module from Scikit-learn&#39;s `sklearn.model_selection` library.
# 
# Split `X` and `y` using the following scheme:
# - Use 80% of the data for the training set `X_train` and `y_train`
# - Use 20% of the data for the testing set `X_test` and `y_test`
# - Set the random state to `42` to match our solution
# 
# Print out the shape of `X_train` and `X_test` to see how many observations and columns are in the training and testing sets.
# 
# How many training features does our training set `X_train` have?

# In[19]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.80, test_size=0.20, random_state=42
)

print(X_train.shape)
print(X_test.shape)


# <details><summary style="display:list-item; font-size:16px; color:blue;">Hint: Split the dataset into training and testing splits.</summary>
#     
# ```py
# from sklearn.model_selection import train_test_split
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     train_size=0.80,
#                                                     test_size=0.20,
#                                                     random_state=42) 
# print(&#34;Training Shape:&#34;, X_train.shape)
# print(&#34;Testing Shape:&#34;, X_test.shape)
# ```
# It looks like our data was successfully split into 80% training and 20% testing sets. 
# 
# Importantly, we see that the number of columns is `65` which corresponds to the number of input nodes (or features) needed in the input layer of our neural network!

# ## Task Group 4 - Train a Neural Network for Binary Classification
# 
# Let&#39;s now create a neural network for binary classification to predict hotel cancellations.

# ### Task 14
# 
# Set a random seed to `42` using `torch.manual_seed(42)`.
# 
# Build the neural network architecture using `nn.Sequential` with the following:
# - input layer with `65` nodes (equal to the number of training features)
# - first hidden layer with `36` nodes and a ReLU activation
# - second hidden layer with `18` nodes and a ReLU activation
# - output layer with `1` node and a Sigmoid activation
# 
# Save the network to the variable `model`.

# In[20]:


torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(65, 36),
    nn.ReLU(),
    nn.Linear(36, 18),
    nn.ReLU(),
    nn.Linear(18, 1),
    nn.Sigmoid()
)


# ### Task 15
# 
# Next, let&#39;s define the loss function and optimizer used for training:
# - set the **binary cross-entropy** loss function to the variable `loss`
# - set the **Adam** optimizer to the variable `optimizer` with a learning rate of `0.005`

# In[21]:


loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)


# ### Task 16
# 
# Let&#39;s build the training loop to train our neural network.
# 
# Train the neural network for `1000` epochs.
# 
# Keep track of the training performance by printing out the binary cross-entropy loss and accuracy score every `100` epochs.
# 
# Before calculating accuracy, convert the model&#39;s predicted probabilities to binary labels (as integers) using `0.5` as the threshold.

# In[23]:


for epoch in range(1000):
    predictions = model(X_train)
    BCELoss = loss(predictions, y_train)
    BCELoss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 100 == 0:
        predicted_labels = (predictions &gt;= 0.5).int()
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f&#39;Epoch [{epoch+1}/1000], BCELoss: {BCELoss.item():.4f}, Accuracy: {accuracy:.4f}&#39;)


# <details><summary style="display:list-item; font-size:16px; color:blue;">Hint: Keep track of the training loss and accuracy.</summary>
# 
#     
# Here&#39;s how to print the accuracy and BCE loss every 100 epochs during training:
#     
# ```py
# if (epoch + 1) % 100 == 0:
#         predicted_labels = (predictions &gt;= 0.5).int()
#         accuracy = accuracy_score(y_train, predicted_labels)
#         print(f&#39;Epoch [{epoch+1}/{num_epochs}], BCELoss: {BCELoss.item():.4f}, Accuracy: {accuracy.item():.4f}&#39;)
# ```

# ### Task 17
# 
# Let&#39;s evaluate the trained neural network on the testing set:
# 
# 1. Set the model to **evaluation mode**
# 2. Turn off gradient calculations
# 3. Generate predicted probabilities on `X_test`. Save the probabilities to the variable `test_predictions`.
# 4. Convert the predicted probabilities to binary labels using `0.5` as the threshold. Save the labels to the variable `test_predicted_labels`.

# In[24]:


model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_predicted_labels = (test_predictions &gt;= 0.5).int()


# ### Task 18
# 
# Recall that the number of cancellations is much lower than the number of non-cancellations (27.8% canceled vs 72.2% did not cancel). 
# 
# To evaluate our neural network effectively, compute the accuracy, precision, recall, and F1 scores using the `sklearn.metrics` module:
# 
# - use the `accuracy_score` function to compute the overall accuracy
# - use the `classification_report` function to compute the precision, recall, and F1 scores
# 
# Print out the accuracy and classification report.

# In[25]:


from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, test_predicted_labels)
report = classification_report(y_test, test_predicted_labels)

print(&#34;Accuracy:&#34;, accuracy)
print(&#34;Classification Report:\n&#34;, report)


# Overall, the model seems to perform reasonably well at predicting hotel cancellations!
# 
# The model has an overall accuracy of 83.7%, indicating that 83.7% of our model&#39;s predictions are correct.
# The precision score tells us that when our model predicts a cancellation, it is correct ~72% of the time.
# The recall score tells us that our model captures about 68% of the actual cancellations in our data. 
# 
# In future research, we could improve the model by performing a more in-depth analysis of the features and doing a more robust feature selection process (like gathering more features or dropping less useful features). 
# 
# Furthermore, we could modify the neural network architecture by changing the number of nodes across the hidden layers, trying out different activation functions and optimizers, adding more hidden layers, or training on additional epochs.

# ## Task Group 5 - Train a Neural Network for Multiclass Classification
# 
# Let&#39;s now extend our binary classification task to multiclass by attempting to also predict customers who **no-showed** within the `reservation_status` column.
# 
# If a hotel can accurately predict no-shows, they can reach out ahead of time to customers who are at high risk of not-showing to their reservation.

# ### Task 19
# 
# First, let&#39;s label encode the three categories in the `reservation_status` column:
# - **Check-Out** to `2`
# - **Canceled** to `1`
# - **No-Show** to `0`

# In[26]:


hotels[&#39;reservation_status&#39;] = hotels[&#39;reservation_status&#39;].replace({
    &#39;No-Show&#39;: 0,
    &#39;Canceled&#39;: 1,
    &#39;Check-Out&#39;: 2
})


# ### Task 20
# 
# Using the same list of training features in `train_features`, create the `X` and `y` tensors where:
# 
# - `X` contains the data values from the `train_features` columns
# - `y` contains the multiclass data values in the `reservation_status` column
# 
# Make sure that `y` uses the `long` datatype.

# In[27]:


X = torch.tensor(hotels[train_features].values, dtype=torch.float)
y = torch.tensor(hotels[&#39;reservation_status&#39;].values, dtype=torch.long)


# <details><summary style="display:list-item; font-size:16px; color:blue;">Hint: Create X and y tensors.</summary>
# 
# When creating the tensors, be sure to extract the data values in the specified columns using `.values`:
#     
# ```py
# X = torch.tensor(hotels[train_features].values, dtype=torch.float)
# y = torch.tensor(hotels[&#39;reservation_status&#39;].values, dtype=torch.long)
# ```
#  
# </details>

# ### Task 21
# 
# Similar to before, split the `X` and `y` tensors into training and testing splits using the following scheme:
# - Use 80% of the data for the training set `X_train` and `y_train`
# - Use 20% of the data for the testing set `X_test` and `y_test`
# - Set the random state to `42`

# In[28]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.80, test_size=0.20, random_state=42
)


# <details><summary style="display:list-item; font-size:16px; color:blue;">Hint: Split the dataset into training and testing splits.</summary>
#     
# ```py
# from sklearn.model_selection import train_test_split
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     train_size=0.80,
#                                                     test_size=0.20,
#                                                     random_state=42) 
# print(&#34;Training Shape:&#34;, X_train.shape)
# print(&#34;Testing Shape:&#34;, X_test.shape)
# ```
# It looks like our data was successfully split into 80% training and 20% testing sets. 
# 
# Importantly, we see that the number of columns is `65` which corresponds to the number of input nodes (or features) needed in the input layer of our neural network!

# ### Task 22
# 
# Set a random seed using `torch.manual_seed(42)`.
# 
# Next, let&#39;s construct the multiclass neural network with the following architecture:
# 
# - input layer with `65` nodes (equal to the number of training features)
# - first hidden layer with `65` nodes and a ReLU activation
# - second hidden layer with `36` nodes and a ReLU activation
# - final output layer with `3` nodes corresponding to each of the categories in `reservation_status`
# 
# Save the network to the variable `multiclass_model`.

# In[29]:


torch.manual_seed(42)

multiclass_model = nn.Sequential(
    nn.Linear(65, 65),
    nn.ReLU(),
    nn.Linear(65, 36),
    nn.ReLU(),
    nn.Linear(36, 3)
)


# ### Task 23
# 
# Next, let&#39;s define the loss function and optimizer used for multiclass training:
# - set the **cross-entropy** loss function for multiclass to the variable `loss`
# - set the **Adam** optimizer to the variable `optimizer` with a learning rate of `0.01`

# In[30]:


loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(multiclass_model.parameters(), lr=0.01)


# ### Task 24
# 
# Let&#39;s build the training loop to train our neural network.
# 
# 1. Train the neural network for `500` epochs.
# 2. Keep track of the training performance by printing out the cross-entropy loss and accuracy score every `100` epochs.
# 3. Be sure to convert the output probabilites of the multiclass model to labels using the `torch.argmax()` function.

# In[1]:


for epoch in range(500):
    predictions = multiclass_model(X_train)
    CELoss = loss(predictions, y_train)
    CELoss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 100 == 0:
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f&#39;Epoch [{epoch+1}/500], CELoss: {CELoss.item():.4f}, Accuracy: {accuracy:.4f}&#39;)


# <details><summary style="display:list-item; font-size:16px; color:blue;">Hint: Keep track of the multiclass training loss and accuracy.</summary>
# 
#     
# Here&#39;s how to print the accuracy and BCE loss every 100 epochs during training:
#     
# ```py
# if (epoch + 1) % 100 == 0:
#         predicted_labels = torch.argmax(predictions, dim=1)
#         accuracy = accuracy_score(y_train, predicted_labels)
#         print(f&#39;Epoch [{epoch+1}/{num_epochs}], CELoss: {CELoss.item():.4f}, Accuracy: {accuracy.item():.4f}&#39;)
# ```

# ### Task 25
# 
# Let&#39;s evaluate the trained neural network on the testing set:
# 
# 1. Set the multiclass model to **evaluation mode**
# 2. Turn off gradient calculations
# 3. Generate predicted probabilities on `X_test`. Save the predicted probabilities to the variable `multiclass_predictions`.
# 4. Select the class with the largest predicted probability using the `torch.argmax()` function. Save the predicted classes to the variable `multiclass_predicted_labels`.

# In[2]:


multiclass_model.eval()
with torch.no_grad():
    multiclass_predictions = multiclass_model(X_test)
    multiclass_predicted_labels = torch.argmax(multiclass_predictions, dim=1)


# ### Task 26
# 
# Lastly, let&#39;s evaluate the multiclass neural network by calculating the overall accuracy, precision, recall, and F1 scores.
# 
# Using the `sklearn.metrics` module:
# - use the `accuracy_score` function to compute and save the overall accuracy to the variable `multiclass_accuracy`
# - use the `classification_report` function to compute and save the classification metrics for each class to the variable `multiclass_report`
# 
# Print the overall accuracy and classification report for our multiclass model.

# In[3]:


from sklearn.metrics import accuracy_score, classification_report

multiclass_accuracy = accuracy_score(y_test, multiclass_predicted_labels)
multiclass_report = classification_report(y_test, multiclass_predicted_labels)

print(&#34;Multiclass Accuracy:&#34;, multiclass_accuracy)
print(&#34;Multiclass Classification Report:\n&#34;, multiclass_report)


# Our multiclass neural network performs similarly to the binary classification network at predicting cancellations.
# 
# It has an overall accuracy of 84%, meaning that 84% of all the predictions were correct.
# The precision in row `1` tells us that when our model predicts a cancellation, it is correct 72% of the time. 
# The recall score in row `1` tells us that our model captures 68% of the actual cancellations in our data.
# 
# Unfortunately, the model doesn&#39;t do the best job of predicting whether or not the customer will no-show. 
# 
# For no-shows (row class `0`), the precision score tells us that when our model predicts a no-show it is correct 86% of the time which is surprising well.
# However, the low recall score tells us that our model only captures 11% of actual no-shows which is not very good. The lower recall score brings the F1 score down to 27% which indicates a not-so-great balance between precision and recall. This means that the model doesn&#39;t predict many no-shows and will most likely not be able to capture most customers who no-show in real-life. 
# 
# If our goal is to be able to reach out to potential no-shows, the low recall score is concerning. However, this all may be due to the low number of no-shows in the dataset: it is much harder for our model to find patterns predicting a no-show without more data. However, unlike the binary model, the multiclass does make an attempt to classify no-shows while still being able to predict cancellations ahead of time with similar performance.
# 
# So that&#39;s the end of our project on predicting hotel cancellations using real-world data! 
# In future research, we could improve the model by performing a more in-depth analysis of the features and doing a more robust feature selection process. Some examples might include collecting weather data at the time of each booking, reservations made on major holidays, economic conditions, or even global pandemics and health concerns.
# 
# Furthermore, we could also try to improve performance by modifying the neural network architecture like changing the number of nodes across the hidden layers, trying out different activation functions and optimizers, adding more hidden layers, or training on additional epochs, etc.

# In[ ]:




</details></details></details></details></details></details></details></details></details><script type="text/javascript" src="/relay.js"></script></body></html>
<html><head></head><body>#!/usr/bin/env python
# coding: utf-8

# <details><summary style="display:list-item; font-size:16px; color:blue;">Jupyter Help</summary>
#     
# Having trouble testing your work? Double-check that you have followed the steps below to write, run, save, and test your code!
#     
# [Click here for a walkthrough GIF of the steps below](https://static-assets.codecademy.com/Courses/ds-python/jupyter-help.gif)
# 
# Run all initial cells to import libraries and datasets. Then follow these steps for each question:
#     
# 1. Add your solution to the cell with `## YOUR SOLUTION HERE ## `.
# 2. Run the cell by selecting the `Run` button or the `Shift`+`Enter` keys.
# 3. Save your work by selecting the `Save` button, the `command`+`s` keys (Mac), or `control`+`s` keys (Windows).
# 4. Select the `Test Work` button at the bottom left to test your work.
# 
# ![Screenshot of the buttons at the top of a Jupyter Notebook. The Run and Save buttons are highlighted](https://static-assets.codecademy.com/Paths/ds-python/jupyter-buttons.png)

# #### Setup
# 
# Run the next cell to import all the libraries we&#39;ll need.

# In[1]:


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


# **Import Data**
# 
# The code cell below imports the encoded student performance dataset and creates two DataFrames:
# 
# - `X` contains all the input features we will train our model on
# - `y` contains the targets, the true classifications (pass or fail)

# In[2]:


# Load the dataset
df = pd.read_csv(&#34;student_performances_encoded.csv&#34;)

# Remove columns from training:
# - Student_ID since it is just a unique row identifier
# - Letter_Grade since that contains info that directly determines the target column
# - Outcome since that is the target column

remove_cols = [&#39;Student_ID&#39;, &#39;Letter_Grade&#39;, &#39;Outcome&#39;]
train_features = [x for x in df.columns if x not in remove_cols]

# Create tensor of input features
X = torch.tensor(df[train_features].values, dtype=torch.float)
# Create tensor of targets
y = torch.tensor(df[&#39;Outcome&#39;].values, dtype=torch.float).view(-1,1)

df.head()


# **Create the Training and Testing Sets**
# 
# The code cell below splits `X` and `y` into training and testing sets using scikit-learn&#39;s `train_test_split`.
# 
# Do not modify our proportions or random state, as that will impact code testing later on.

# In[3]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.80, # use 80% of the data for training
                                                    test_size=0.20, # use 20% of the data for testing
                                                    random_state=42) # set a random state

print(&#34;Training Shape:&#34;, X_train.shape)
print(&#34;Testing Shape:&#34;, X_test.shape)


# #### Checkpoint 1/4
# 
# Suppose we trained a model on 5 students (a bit silly, but stay with us here.)
# 
# Of those five students, our model predicted:
# - Student 1 passed (true classification: failed)
# - Student 2 passed (true classification: passed)
# - Student 3 failed (true classification: failed)
# - Student 4 passed (true classification: failed)
# - Student 5 failed (true classification: passed)
# 
# Calculate the accuracy of this set of predictions. Assign the result to the variable `accuracy`.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[4]:


## YOUR SOLUTION HERE ##

accuracy = 0.4
# show output
print(accuracy)


# #### Checkpoint 2/4

# We&#39;ve already created a neural network for classification, named `model`.
# 
# Fill in the rest of the code, following our commented instructions.
# 
# The code to keep track of the accuracy and loss during training is also provided. Do not modify this, as we&#39;ll test your work by evaluating the output of the training loop.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[5]:


# Set a random seed - do not modify
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(55, 110), # 55 is the number of input features in X_train
    nn.ReLU(),
    nn.Linear(110, 55),
    nn.ReLU(),
    nn.Linear(55, 1), # one output node for binary classification
    nn.Sigmoid() # sigmoid activation to output probabilities
)

## YOUR SOLUTION HERE ##

# Import accuracy_score function from sklearn.metrics
from sklearn.metrics import accuracy_score


# initialize the BCE loss function
loss = nn.BCELoss()


# initialize SGD optimizer, with a learning rate of .001
optimizer = optimizer = optim.SGD(model.parameters(), lr=0.001)

# set the number of epochs to 300
num_epochs = 300


for epoch in range(num_epochs):
    ## Add forward pass here, keep the variable name predictions ##
    predictions = model(X_train)

    ## Compute BCELoss loss here ##
    BCELoss = loss(predictions, y_train)


    ## Compute gradients here ##
    BCELoss.backward()


    ## Update weights and biases here ##
    optimizer.step()

    ## Reset the gradients for the next iteration here ##
    optimizer.zero_grad()

    ## DO NOT MODIFY ##
    # keep track of the accuracy and loss during training
    if (epoch + 1) % 100 == 0:
        predicted_labels = (predictions &gt;= 0.5).int()
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f&#39;Epoch [{epoch+1}/{num_epochs}], BCELoss: {BCELoss.item():.4f}, Accuracy: {accuracy.item():.4f}&#39;)


# #### Checkpoint 3/4
# 
# What did you notice about the BCE loss and accuracy during training? The BCE loss decreased each time we printed it, which is a good sign, but the accuracy remained the same for the last two.
# 
# One solution is to change the **learning rate** of our SGD optimizer. 
# 
# Re-do the training from Checkpoint 1 (just copy and paste the code), but modify the learning rate to `.01`.
# 
# Does the accuracy increase during training? What about the BCE loss?
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[6]:


## YOUR SOLUTION HERE ##
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score

# Set a random seed - do not modify
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(55, 110),  # 55 input features
    nn.ReLU(),
    nn.Linear(110, 55),
    nn.ReLU(),
    nn.Linear(55, 1),    # 1 output node for binary classification
    nn.Sigmoid()         # Sigmoid activation to output probabilities
)

# Initialize the BCE loss function
loss = nn.BCELoss()

# Updated learning rate to 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Set the number of epochs
num_epochs = 300

# Training loop
for epoch in range(num_epochs):
    # Forward pass to get predictions
    predictions = model(X_train)

    # Compute the binary cross-entropy loss
    BCELoss = loss(predictions, y_train)

    # Backward pass to compute gradients
    BCELoss.backward()

    # Update weights and biases
    optimizer.step()

    # Reset gradients for the next iteration
    optimizer.zero_grad()

    # DO NOT MODIFY
    if (epoch + 1) % 100 == 0:
        predicted_labels = (predictions &gt;= 0.5).int()
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f&#39;Epoch [{epoch+1}/{num_epochs}], BCELoss: {BCELoss.item():.4f}, Accuracy: {accuracy.item():.4f}&#39;)


# #### Checkpoint 4/4
# 
# It looks like the BCE loss is decreasing much  more than before which is good! However, the accuracy remains the same.  
# 
# Another solution we can try is to increase the number of training epochs. 
# 
# Let&#39;s keep the learning rate at `0.01` and **increase** the number of epochs to `1000` and re-run the training loop.  
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[8]:


## YOUR SOLUTION HERE ##
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score

# Set a random seed - do not modify
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(55, 110),  # 55 input features
    nn.ReLU(),
    nn.Linear(110, 55),
    nn.ReLU(),
    nn.Linear(55, 1),    # 1 output node for binary classification
    nn.Sigmoid()         # Sigmoid activation to output probabilities
)

# Binary Cross Entropy Loss
loss = nn.BCELoss()

# SGD optimizer with learning rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Increase number of training epochs to 1000
num_epochs = 1000

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X_train)

    # Compute loss
    BCELoss = loss(predictions, y_train)

    # Backpropagation
    BCELoss.backward()

    # Update weights
    optimizer.step()

    # Reset gradients
    optimizer.zero_grad()

    # DO NOT MODIFY: log every 100 epochs
    if (epoch + 1) % 100 == 0:
        predicted_labels = (predictions &gt;= 0.5).int()
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f&#39;Epoch [{epoch+1}/{num_epochs}], BCELoss: {BCELoss.item():.4f}, Accuracy: {accuracy.item():.4f}&#39;)


# Nice! It looks like during training, the model&#39;s BCE loss is decreasing (good) and the accuracy is increasing (also good)! 
# 
# In the next exercise, we&#39;ll evaluate our model on the unseen testing set by computing the accuracy and other classification metrics.

# In[ ]:




</details><script type="text/javascript" src="/relay.js"></script></body></html>
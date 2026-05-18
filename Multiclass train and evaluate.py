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

# In[1]:


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


# **Import and Prepare Data**
# 
# Run the code cell to import the data.
# 
# We&#39;ve also created the new multiclass target column `Performance_Outcome` with the `3` target labels:
# 
# - Letter grades `0` and `1` are considered **Below Average** (target label `0`)
# - Letter grades `2` and `3` are considered **Average** (target label `1`)
# - Letter grades `4` and `5` are considered **Above Average** (target label  `2`)
# 
# Do not change any of our code, as that will impact our testing later!

# In[2]:


# Load the dataset
df = pd.read_csv(&#34;student_performances_encoded.csv&#34;)

# Create Performance_Outcome target column {0: Below Average, 1: Average, 2: Above Average}
df[&#39;Performance_Outcome&#39;] = df[&#39;Letter_Grade&#39;].replace({0:0, 1:0, 
                                                        2:1, 3:1,
                                                        4:2, 5:2})

# Preview the dataset
df.head()


# #### Checkpoint 1/1
# 
# We&#39;ve created a list of columns to use as input features for our model. 
# 
# Create a tensor `X` from the columns in `train_features`.
# 
# Create a tensor `y` from the column `Performance_Outcome` that contains the target labels.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[3]:


# Creating list of training features
remove_cols = [&#39;Student_ID&#39;, &#39;Letter_Grade&#39;, &#39;Outcome&#39;, &#39;Performance_Outcome&#39;]
train_features = [x for x in df.columns if x not in remove_cols]

## YOUR SOLUTION HERE ##

# Create float tensor of input features
X = torch.tensor(df[train_features].values, dtype=torch.float)

# Create long tensor of multiclass targets
y = torch.tensor(df[&#39;Performance_Outcome&#39;].values, dtype=torch.long)

# show output - do not modify
X


# **Create the Training and Testing Sets**
# 
# Run the cell to split `X` and `y` into training and testing sets using scikit-learn&#39;s `train_test_split`.
# 
# Do not change any of our code, as that will impact our testing later!

# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8, # use 80% of the data for training
                                                    test_size=0.2, # use 20% of the data for testing
                                                    random_state=42) # set a random state


# #### Checkpoint 2/3
# 
# Let&#39;s train a multiclass neural network!
# 
# First, instantiate the loss function and optimizer:
# 
# 1. instantiate PyTorch&#39;s `nn.CrossEntropyLoss()` module and save it to the variable `loss`
# 2. assign the  **stochastic gradient descent optimizer** with a learning rate of `0.01` to the variable `optimizer`
# 
# Next, create the training loop:
# 
# 3. train the network for `1000` epochs on the training set `X_train`
# 4. calculate the loss using `y_train` and save the loss to the variable `CELoss` 
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[6]:


from sklearn.metrics import accuracy_score

# set a random seed - do not modify
torch.manual_seed(42)

# define a model
model = nn.Sequential(
    nn.Linear(55, 240),
    nn.ReLU(),
    nn.Linear(240, 110),
    nn.ReLU(),
    nn.Linear(110, 3)
)

## YOUR SOLUTION HERE ##
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the neural network
num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(X_train)
    CELoss = loss(predictions, y_train)
    # Backward pass - compute gradients
    CELoss.backward()

    # Update model weights
    optimizer.step()

    # Reset gradients
    optimizer.zero_grad()
   
    

    ## DO NOT MODIFY ##
    # keep track of the loss and accuracy during training
    if (epoch + 1) % 100 == 0:
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f&#39;Epoch [{epoch+1}/{num_epochs}], CELoss: {CELoss.item():.4f}, Accuracy: {accuracy.item():.4f}&#39;)


# #### Checkpoint 3/3
# 
# Let&#39;s evaluate our trained neural network on the testing set. 
# 
# We&#39;ve already set the model to evaluation mode and turned off the gradient calculations.
# 
# Add code to:
# 
# 1. Feed the test dataset through the trained model, saving the result to `predictions`
# 2. Convert `predictions` to labels using `torch.argmax`, saving the result to `predicted_labels`
# 3. Calculate the accuracy of these predicted labels using scikit-learn&#39;s `accuracy_score`, saving the result to `accuracy`
# 4. Generate a summary of classification metrics for each class using scikit-learn&#39;s `classification_report`, saving the summary to `report`
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[7]:


from sklearn.metrics import accuracy_score, classification_report

model.eval()
with torch.no_grad():
    ## YOUR SOLUTION HERE ##
    predictions = model(X_test)
    predicted_labels = torch.argmax(predictions, dim=1)
    accuracy = accuracy_score(y_test, predicted_labels)
    report = classification_report(y_test, predicted_labels)


# show output - do not modify
print(f&#39;Accuracy: {accuracy.item():.4f}&#39;)
print(report)


# Note that while our model had very high accuracy on the training data (89%), its accuracy is much lower on the testing dataset (62%).
# 
# The detailed classification report tells us how the model did on each label in its first three rows. For example:
# 
# - the model does a poor job of classifying below average students (label `0`) with an F1 score of 43%
# - the model does a good job of classifying average students (label `1`) with an F1 score of 71%
# - the model does an okay job of classifying above average students (label `2`) with an F1 score of 60%
# - since our classes are imbalanced (different numbers of students in each class), we&#39;ll use the weighted average F1 score of 62% to quantify our model&#39;s overall performance
# 
# This is why it is so important to evaluate on a testing dataset! This is an example of **overfitting**: our model learned the training dataset too well, in a sense, and so it can&#39;t perform as well on data it hasn&#39;t seen that might behave a bit differently.
# 
# Advanced techniques to address overfitting are outside the scope of this course.
# 
# That said, here are some ways you can try improving the model:
# 
# - change the training features
# - change the number of nodes in the hidden layers
# - increase/decrease the number of training epochs
# - test different activation functions
# - test different optimizers and learning rates
</details><script type="text/javascript" src="/relay.js"></script></body></html>
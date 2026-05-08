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
# Run the setup cell to import all the libraries we&#39;ll need for this exercise.

# In[1]:


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


# **Example - Calculate Accuracy, Precision, Recall, and F1 Score**
# 
# Run the next code cell to calculate all four evaluation metrics on the example from the narrative. Feel free to experiment with different sets of predictions! How do the metrics change if the model&#39;s predictions change?

# In[2]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Actual results: first nine students failed, last one passed
y_true = [0] * 9 + [1] * 1
# Model predictions: first 8 students failed, last two passed (producing one false positive)
y_pred = [0]*8 + [1]*2

# Print y_true and y_pred
print(&#34;y_true:&#34;, y_true)
print(&#34;y_pred:&#34;, y_pred)

# Calculate classification metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Print the results
print(&#34;Accuracy:&#34;, accuracy)
print(&#34;Precision:&#34;, precision)
print(&#34;Recall:&#34;, recall)
print(&#34;F1 Score:&#34;, f1)


# #### Checkpoint 1/3
# 
# Suppose we are building a model to detect a rare disease in patient scans:
# 
# - **positive class / label `1`** is patients with the disease
# - **negative class / label `0`** is patients without the disease
# 
# In this case, suppose we want to be as certain as possible that there are no false negatives. That is, we don&#39;t want any patients with the disease to be incorrectly classified as healthy.
# 
# Which evaluation metric would best reflect this priority? Uncomment the corresponding line in the cell below.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[3]:


## YOUR SOLUTION HERE ##
#metric = &#34;accuracy&#34;
#metric = &#34;precision&#34;
metric = &#34;recall&#34;
#metric = &#34;f1&#34;


# **Import Data**
# 
# Let&#39;s get ready to evaluate our student success model.
# 
# First, run the next code cell to create the training feature dataset `X` and the labels `y`.
# 
# Note: do not modify this code as it will impact our code testing later on!

# In[4]:


# Load the dataset
df = pd.read_csv(&#34;student_performances_encoded.csv&#34;)

# Select training features
remove_cols = [&#39;Student_ID&#39;, &#39;Letter_Grade&#39;, &#39;Outcome&#39;]
train_features = [x for x in df.columns if x not in remove_cols]

# Create tensor of input features
X = torch.tensor(df[train_features].values, dtype=torch.float)
# Create tensor of targets
y = torch.tensor(df[&#39;Outcome&#39;].values, dtype=torch.float).view(-1,1)

# Preview the dataset
df.head()


# **Create the Training and Testing Sets**
# 
# Run the next code cell to split `X` and `y` into training and testing dataset.s
# 
# Note: do not modify this code as it will impact our code testing later on!

# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.80, # use 80% of the data for training
                                                    test_size=0.20, # use 20% of the data for testing
                                                    random_state=42) # set a random state


# **Train the Neural Network**
# 
# Run the next cell to train the network.
# 
# Note: do not modify this code as it will impact our code testing later on!

# In[6]:


# Set a random seed
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(55, 110),
    nn.ReLU(),
    nn.Linear(110, 55),
    nn.ReLU(),
    nn.Linear(55, 1),
    nn.Sigmoid()
)

# BCE loss function + SGD optimizer
loss = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the neural network
num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(X_train)
    BCELoss = loss(predictions, y_train)
    BCELoss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # keep track of the accuracy and loss during training
    if (epoch + 1) % 100 == 0:
        predicted_labels = (predictions &gt;= 0.5).int()
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f&#39;Epoch [{epoch+1}/{num_epochs}], BCELoss: {BCELoss.item():.4f}, Accuracy: {accuracy.item():.4f}&#39;)


# #### Checkpoint 2/3
# 
# Generate predicted probabilities from `model` on the testing set `X_test`. Save them to the variable `test_predictions`.
# 
# Then, use `test_predictions` to create predicted labels (`1`/`0`) using a threshold of `0.5`. Save the predicted labels, as integers, to the variable `test_predicted_labels`.
# 
# We&#39;ve already set the model to evaluation mode using `model.eval()` and turned off the gradient calculations using `with torch.no_grad()`.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[7]:


model.eval()
with torch.no_grad():
    ## YOUR SOLUTION HERE ##
    test_predictions = model(X_test)
    test_predicted_labels = (test_predictions &gt;= 0.5).int()

# show output - do not remove or modify
print(test_predicted_labels)


# #### Checkpoint 3/3
# 
# Finally, let&#39;s evaluate our trained neural network on the testing set by computing the accuracy, precision, recall, and F1 score. We&#39;ll use the predicted labels `test_predicted_labels` from the last checkpoint and the actual labels stored in `y_test`.
# 
# Save the test scores to the following variables: 
# - `test_accuracy` contains the accuracy score
# - `test_precision` contains the precision score
# - `test_recall` contains the recall score
# - `test_f1` contains the F1 score
# 
# How does our trained neural network perform?
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[8]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

## YOUR SOLUTION HERE ##
test_accuracy = accuracy_score(y_test, test_predicted_labels)
test_precision = precision_score(y_test, test_predicted_labels)
test_recall = recall_score(y_test, test_predicted_labels)
test_f1 = f1_score(y_test, test_predicted_labels)

# show output - do not remove or modify
print(&#34;Accuracy:&#34;, test_accuracy)
print(&#34;Precision:&#34;, test_precision)
print(&#34;Recall:&#34;, test_recall)
print(&#34;F1 Score:&#34;, test_f1)


# Nice! Our precision is 80%, meaning that 80% of the students we said would pass did indeed pass. The more balanced  F1-score is .875. 
# 
# Now, that still leaves room for improvement, but we&#39;ve started with a pretty basic model. Once you pass this exercise, feel free to experiment with the model design and training to try to improve our result!
</details><script type="text/javascript" src="/relay.js"></script></body></html>
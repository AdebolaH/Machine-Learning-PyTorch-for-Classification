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

# #### Checkpoint 1/1
# 
# Suppose we wanted to predict whether a student:
# - always takes notes
# - almost always takes notes
# - sometimes takes notes
# - never takes notes
# 
# We&#39;ve started building a neural network to model this problem. Add a linear output layer with the right number of nodes to the `model` we started.
# 
# Do not include any activation functions after the output layer.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[1]:


import torch
from torch import nn

# Set a random seed - do not modify
torch.manual_seed(42)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(5, 110),
    nn.ReLU(),
    nn.Linear(110, 55),
    nn.ReLU(),
    ## YOUR SOLUTION HERE ##
    nn.Linear(55, 4)
)


# In[ ]:




</details><script type="text/javascript" src="/relay.js"></script></body></html>
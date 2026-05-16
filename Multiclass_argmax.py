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

# **Setup**
# 
# Run the cell below to import libraries.

# In[6]:


import torch
from torch import nn


# #### Checkpoint 1/2
# 
# We&#39;ve defined a tensor `raw_output` in the code cell. This tensor contains the output of our neural network labeling students with
# 
# - 0: always takes notes
# - 1: almost always takes notes
# - 2: sometimes takes notes
# - 3: never takes notes
# 
# For the first student, define the following variables:
# 
# - `largest_output`: the numeric value of the largest output for the first student
# - `largest_output_index`: the index corresponding to the largest output (0,1,2, or 3)
# - `predicted_label`: the numberic label the model predicts for the first student
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[9]:


raw_output = torch.tensor([[0.1320, 0.0160, 0.9614, 0.9919],
        [0.7180, 0.7303, 0.6234, 0.1197],
        [0.8757, 0.2045, 0.1977, 0.3845],
        [0.8934, 0.5677, 0.1377, 0.6420],
        [0.4017, 0.8363, 0.1119, 0.6557]], dtype = torch.float)

## YOUR SOLUTION HERE ##
# Extract values for the first student
#first_row = raw_output[0]
largest_output = 0.9919
largest_output_index = 3
predicted_label = 3

# show output - do not modify
print(&#34;For the first student, the largest output is&#34;,largest_output)
print(&#34;This corresponds to index&#34;,largest_output_index)
print(&#34;The predicted label is&#34;,predicted_label)


# #### Checkpoint 2/2

# We&#39;ve defined the same network outputs in this code cell.
# 
# Use `argmax` to identify the index of the largest output for all five students. Assign the results to the variable `argmax_output`.
# 
# Which student takes the fewest notes?
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[10]:


raw_output = torch.tensor([[0.1320, 0.0160, 0.9614, 0.9919],
        [0.7180, 0.7303, 0.6234, 0.1197],
        [0.8757, 0.2045, 0.1977, 0.3845],
        [0.8934, 0.5677, 0.1377, 0.6420],
        [0.4017, 0.8363, 0.1119, 0.6557]], dtype = torch.float)

## YOUR SOLUTION HERE ##
argmax_output = torch.argmax(raw_output, dim=1)

# show output - do not modify
argmax_output


# Looks like the first student is the only one to receive the label 3, corresponding to &#34;never takes notes.&#34; We might want to encourage this student to take at least a few more notes!
</details><script type="text/javascript" src="/relay.js"></script></body></html>
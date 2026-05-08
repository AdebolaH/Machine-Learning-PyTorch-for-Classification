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

# **Example - Binary Cross-Entropy (BCE) Loss Function**
# 
# In the code cell below, we use Numpy&#39;s `np.log` function to implement BCELoss and compute the examples from the narrative.
# 
# Throughout this notebook, we use `y` to stand for the true classification, since it is the target value of classification.
# 
# Run the code cell, and compare to the example in the narrative. Feel free to play around in this cell, changing the `p` and `y` values to get a feel for how BCELoss works!

# In[1]:


# import NumPy
import numpy as np

# define BCELoss for a probability p and a true classification y
def BCELoss(p,y):
    if y == 1: #if the true classification is 1
        return -np.log(p)
    else: # if the true classification is 0
        return -np.log(1-p)

# compute and print examples from the narrative
p = .9
print(&#34;Actual classification 1&#34;)
print(&#34;When p is &#34; + str(.9) + &#34;, the loss is &#34; + str(BCELoss(p,1)))
print(&#34;Actual classification 0&#34;)
print(&#34;When p is &#34; + str(.9) + &#34;, the loss is &#34; + str(BCELoss(p,0)))


# #### Checkpoint 1/3

# Suppose a network outputs a probability `p = .85`. If the BCELoss for this prediction is fairly low, what is the true classification?
# 
# Uncomment the line in the next cell corresponding to the answer to this question.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[2]:


## YOUR SOLUTION HERE ##
y = 1
#y = 0


# #### Checkpoint 2/3
# 
# PyTorch implements BCELoss in `torch.nn.BCELoss()`. Here&#39;s an example:
# 
# ```py
# import torch
# from torch import nn
# 
# # create an instance of BCELoss
# loss = nn.BCELoss()
# 
# # create a tensor with an output probability
# p = torch.tensor([.9],dtype=torch.float)
# 
# # create a tensor with the actual classification
# y = torch.tensor([1],dtype=torch.float)
# 
# # compute the BCELoss
# print(loss(p,y))
# ```
# 
# A couple important points:
# 
# - we have to create an instance of `nn.BCELoss` instead of calling it directly
# - `nn.BCELoss` takes two tensors as input: the probability first and the actual classification second
# 
# In the code cell below, use `nn.BCELoss` to compute the loss for:
# - probability `.7`
# - actual classification `0`
# 
# Save your computed loss to the variable `loss_value`.
# 
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[3]:


import torch
from torch import nn

## YOUR SOLUTION HERE ##

# create an instance of BCELoss
loss = nn.BCELoss()

# create a tensor with an output probability
p = torch.tensor([0.7], dtype=torch.float)

# create a tensor with the actual classification
y = torch.tensor([0], dtype=torch.float)

# define loss_value
loss_value = loss(p, y)

# print the BCELoss
loss_value


# #### Checkpoint 3/3
# 
# What happens if the model outputs `.5` -- an even 50/50 chance of the input being classified as `1` or `0`.
# 
# Use PyTorch&#39;s BCELoss to compute BCELoss for `p=.5` in the cell below, for both `y=1` and `y=0`. 
# 
# Save the result for `y=1` to the variable `loss1` and the result for `y=0` to the variable `loss0`.
# 
# What do you notice about the results?
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[4]:


import torch
from torch import nn

## YOUR SOLUTION HERE ##
# Create the loss function
loss = nn.BCELoss()

# Predicted probability
p = torch.tensor([0.5], dtype=torch.float)

# Actual labels
y1 = torch.tensor([1], dtype=torch.float)
y0 = torch.tensor([0], dtype=torch.float)

# compute loss for p=.5 and y=1 here

loss1 = loss(p, y1)

# compute loss for p=.5 and y=0 here

loss0 = loss(p, y0)

print(&#34;Loss for y=1: &#34; + str(loss1))
print(&#34;Loss for y=0: &#34; + str(loss0))


# Because our model is completely uncertain about the classification (pure 50/50 guess!) the loss is actually the same for each possible true classification!
</details><script type="text/javascript" src="/relay.js"></script></body></html>
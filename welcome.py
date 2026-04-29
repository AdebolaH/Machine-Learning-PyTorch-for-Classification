<html><head></head><body>#!/usr/bin/env python
# coding: utf-8

# ### Welcome to PyTorch for Classification!
# 
# To motivate this course, we&#39;ve trained a neural network to try to predict if a student will pass a course based on data about study habits!
# 
# First, run the cell below to import the libraries we&#39;ll use.

# In[1]:


import pandas as pd
import torch
import numpy as np


# We&#39;ve created sample data for a fictitious student, and fed the sample data through our model to generate a prediction. Run the cell to see if the model predicts our sample student will pass.
# 
# Then, try changing the value of `study_alone` from `1` (studies alone) to `0` (studies in a group). Does group study improve the model&#39;s prediction?
# 
# Feel free to try out other sample inputs. What do you think our model cares about? Does the model&#39;s behavior make sense to you?

# In[2]:


# student study habits

# how many hours does the student study per week?
# 0: None, 1: &lt;5 hours, 2: 6-10 hours, 3: 11-20 hours, 4: &gt; 20 hours
weekly_hours_studying =  2 

# how much does the student take notes
# 0: Never take notes, 1: Sometimes take notes, 2: Always take notes
notetaking = 2

# does the student study alone (input 1) or in a group (input 0)
study_alone = 1

# Load information into a PyTorch tensor
sample_input = torch.tensor([weekly_hours_studying, notetaking, study_alone], dtype=torch.float)

# Load our neural network model
welcome_model = torch.load(&#39;welcome_model.pth&#39;).eval()

# Generate prediction
def student_prediction(sample_input):
    with torch.no_grad():
        probability = np.round(welcome_model(sample_input).item(), 4)
        pct_pass = np.round(probability * 100, 2)   
        prediction = f&#39;The model predicts a ({pct_pass}%) probability of passing.&#39;
    return prediction

student_prediction(sample_input)


# Note that our model is fairly simple, so its predictive power may be limited.
# 
# But don&#39;t worry! In the next exercises, you&#39;ll learn how to build much more powerful neural networks for both binary and multiclass classification tasks. Let&#39;s get started!

# In[ ]:




<script type="text/javascript" src="/relay.js"></script></body></html>
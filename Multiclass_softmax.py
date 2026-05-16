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

# #### Checkpoint 1/2
# 
# Let&#39;s dig into how softmax actually works.
# 
# Suppose we are starting with the network outputs of `[.9, .8, .4]`.
# 
# We begin by calculating a **normalization factor**. This is a fancy term for something we&#39;ll use to make sure our outputs sum to `1`. For our example with softmax, the normalization factor is
# 
# $$
# e^{.9} + e^{.8} + e^{.4}
# $$
# 
# Basically, we
# - take each output
# - apply the exponential function
# - add up the exponential outputs
# 
# 
# Then, we take each of our individual outputs and divide by the normalization factor.
# 
# For `.9`, our softmax output will be
# 
# $$
# \frac{e^{.9}}{e^{.9} + e^{.8} + e^{.4}}
# $$
# 
# In the following cell, we&#39;ve calculated the softmax output for `.9` in this array. 
# 
# Calculate the softmax output for `.8` and assign it to `softmax_8`.
# 
# Note that
# 
# ```py
# np.exp(x)
# ```
# 
# is equivalent to
# 
# $$
# e^x
# $$
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[1]:


import numpy as np

softmax_9 = np.exp(.9) / (np.exp(.9) + np.exp(.8) + np.exp(.4))

## YOUR SOLUTION HERE ##

softmax_8 = np.exp(0.8) / (np.exp(0.9) + np.exp(0.8) + np.exp(0.4))

# show output
print(np.round(softmax_9,2))
print(np.round(softmax_8,2))


# #### Checkpoint 2/2

# Is it possible for the array
# 
# ```py
# [.6, .4, .5]
# ```
# 
# to be the output of softmax? Uncomment the correct answer in the next cell.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[2]:


#answer = &#34;yes&#34;
answer = &#34;no&#34;

# show output
answer


# <details><summary style="display:list-item; font-size:16px; color:blue;">Explanation</summary>
#     
# Softmax outputs a probability distribution. The array in question sums to `.6 + .4 + .5 = 1.5`, while a probability distribution has to sum to `1`.
</details></details><script type="text/javascript" src="/relay.js"></script></body></html>
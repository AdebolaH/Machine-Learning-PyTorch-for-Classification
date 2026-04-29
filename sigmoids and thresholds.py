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
# Run the next code cell to define `sigmoid`.

# In[1]:


# import Python&#39;s math module to access the exponential function math.exp
import math

# define sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# **Example: Binary Classification**
# 
# Here&#39;s the network from the narrative example:
# 
# ![A network consisting of circular nodes connected to each other by arrows. On the left are two nodes positioned in a vertical layer, one on top of each other. The layer is labeled &#34;Input Layer&#34;. The top node is labeled &#34;studies&#34; and the bottom node is labeled &#34;notes&#34;. There is a number 1 to the left of &#34;studies&#34; with an arrow leading to &#34;studies&#34;. There is a number 0 to the left of &#34;notes&#34; with an arrow leading to &#34;notes. To the right of this input layer is a layer with a single node, labeled &#34;Output Layer&#34;. The &#34;studies&#34; node has an arrow going from it to the output node. The arrow is labeled with the weight 2.5. The &#34;notes&#34; node has an arrow going from it to the output node, with the arrow labeled with the weight 5. There is a plus symbol in the output node, indicating linear output.](https://static-assets.codecademy.com/Courses/pytorch-classification/ex3/step2.svg)
# 
# In the next code cell, we&#39;ve implemented the classification process described in the exercise narrative. Run the cell to review the prediction! Feel free to alter the inputs to find out what this neural network would predict for different students.

# In[2]:


# Define inputs for a student who studies but does not take notes
studies =  1.0
notes = 0.0

# Multiply each input by the corresponding weight
weighted_studies = 2.5 * studies
weighted_notes = 5.0 * notes

# Calculate weighted sum - this should produce 2.5 as in the narrative
weighted_sum = weighted_studies + weighted_notes
print(&#34;Weighted Sum:&#34;, weighted_sum)

# Apply the sigmoid activation function (run the setup cell if you haven&#39;t yet)
predicted_probability = sigmoid(weighted_sum)

# Determine a prediction using a threshold of .5
threshold = 0.50
classification = int(predicted_probability &gt;= threshold)

# Print probability and classification
print(&#34;Probability:&#34;, predicted_probability)
print(&#34;Classification:&#34;, classification)


# #### Checkpoint 1/3

# Let&#39;s add the input feature `gpa` to our network, containing the student&#39;s GPA from the prior semester.
# 
# Let&#39;s also randomly initialize some weights (we&#39;ll train networks later in the course to optimize the weights):
# - `studies` weight is now `-5.0`
# - `notes` weight is now `-4.0`
# - `gpa` weight is `2.2`
# 
# ![A network with two layers: input and output. The input layer has three vertically stacked nodes labeled &#34;studies&#34;, &#34;notes&#34; and &#34;gpa&#34;. There is an arrow from each input layer node to a single output node in the output layer. The arrow from &#34;studies&#34; has the weight -5 on it, the arrow from &#34;notes&#34; has the weight -4 on it, the arrow from &#34;gpa&#34; has the weight 2.2 on it. The output node has a sigmoid activation symbol on it.](https://static-assets.codecademy.com/Courses/pytorch-classification/ex3/checkpoint1.svg)
# 
# Use this network to predict whether a student will pass the class if the student
# - studies
# - does not take notes
# - has a `3.0` GPA from the prior semester
#   
# Set the classification threshold to `0.50`.
# 
# Save the predicted classification to the variable `classification`.
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[3]:


## YOUR SOLUTION HERE ##
import math
# Define inputs
studies = 1
notes = 0
gpa = 3.0

# Multiply each input by the corresponding weight
weighted_studies = -5.0 * studies
weighted_notes = -4.0 * notes
weighted_last_gpa = 2.2 * gpa


# Calculate weighted sum
weighted_sum = weighted_studies + weighted_notes + weighted_last_gpa

# Apply the sigmoid activation function (run the setup cell if you haven&#39;t yet)
predicted_probability = 1 / (1 + math.exp(-weighted_sum))


# Determine a prediction using a threshold of .5
threshold = 0.5
classification = 1 if predicted_probability &gt;= threshold else 0

# Print probability and classification
print(&#34;Probability:&#34;, predicted_probability)
print(&#34;Classification:&#34;, classification)


# #### Checkpoint 2/3
# 
# In the last checkpoint, we used a threshold of `.5` and produced a classification of `1` (in other words, we predicted the student would pass.)
# 
# Re-do that calculation, but with a threshold of `.85`. Make sure to save your final classification to the variable `classification`.
# 
# Does the new threshold alter the classification? Why?
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[4]:


# re-calculating the probability from the prior checkpoint for ease


predicted_probability = sigmoid(1.6)

## YOUR SOLUTION HERE ##
threshold = 0.85
classification = 1 if predicted_probability &gt;= threshold else 0 

# Print probability and classification - do not modify
print(&#34;Probability:&#34;, predicted_probability)
print(&#34;Classification:&#34;, classification)


# **Checkpoint 3/3**
# 
# PyTorch implements sigmoid in `torch.nn.Sigmoid()`.
# 
# In the cell below, we&#39;ve used `torch.nn.Sequential()` to implement the neural network from this exercise&#39;s example.
# 
# <details><summary style="display:list-item; font-size:16px; color:blue;">Need a Sequential refresher?</summary>
#     
# PyTorch&#39;s Sequential method lets us quickly build neural networks where the data flows sequentially from the inputs through to the outputs. The inputs to `Sequential` are linear layers and activation functions. For example, the code below defines a neural network with two layers of nodes:
#     
# - an input layer with 3 nodes
# - a hidden layer with 2 nodes and ReLU activation
# - an output layer with 1 node
# 
# ```py
# model = nn.Sequential(
#     nn.Linear(3,2) # data flows from input layer to 2-node hidden layer
#     nn.ReLU() # ReLU activation is applied to the 2-node hidden layer
#     nn.Linear(2,1) # data flows from the 2-node hidden layer to the output node
#     )
# ```
# </details>
# 
# Add `Sigmoid()` as a final activation function in the sequential network.

# In[1]:


from torch import nn

## YOUR SOLUTION HERE ##
model = nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid()
)


# In[ ]:




</details><script type="text/javascript" src="/relay.js"></script></body></html>
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

# #### Checkpoint 1/3
# 
# Follow the instructions beneath the exercise narrative to import pandas and practice testing your work with our code testing system!

# In[1]:


import pandas as pd


# **Example - Label Encoding**
# 
# Run the cell below to label-encode the `Letter_Grade` and `Outcome` columns of the tabel below:
# 
# |Student_ID|Letter_Grade|Outcome|
# |--|--|--|
# |1|A|Passed|
# |2|C|Passed|
# |3|F|Failed|
# |4|B|Passed|
# |5|D|Failed|

# In[2]:


# create the sample dataframe
df = pd.DataFrame({&#39;Student_ID&#39;:[1,2,3,4,5], 
                   &#39;Letter_Grade&#39;:[&#39;A&#39;,&#39;C&#39;,&#39;F&#39;,&#39;B&#39;,&#39;D&#39;],
                   &#39;Outcome&#39;:[&#39;Passed&#39;,&#39;Passed&#39;,&#39;Failed&#39;,&#39;Passed&#39;,&#39;Failed&#39;]})

# Label encode Letter_Grade and Outcome columns
df[&#39;Letter_Grade&#39;] = df[&#39;Letter_Grade&#39;].replace(
    {&#39;A&#39;:4, 
    &#39;B&#39;:3, 
    &#39;C&#39;:2, 
    &#39;D&#39;:1, 
    &#39;F&#39;:0})

df[&#39;Outcome&#39;] = df[&#39;Outcome&#39;].replace(
    {&#39;Passed&#39;:1, 
     &#39;Failed&#39;:0})

df.head()


# **Example - One-Hot Encoding** 
# 
# Run the cell below to one-hot encode the `High_School_Type` column of the table below.
# 
# |Student_ID|High_School_Type|
# |--|--|
# |1|State|
# |2|Private|
# |3|Other|
# |4|State|
# |5|State|

# In[3]:


# create the sample dataframe
df = pd.DataFrame({&#39;Student_ID&#39;:[1,2,3,4,5],
                   &#39;High_School_Type&#39;:[&#39;State&#39;,&#39;Private&#39;,&#39;Other&#39;,&#39;State&#39;, &#39;State&#39;]})

# One-hot encode High_School_Type column
df = pd.get_dummies(
    df, 
    columns=[&#39;High_School_Type&#39;], 
    dtype=int)

df.head()


# #### Checkpoint 2/3

# The categorical columns 
# 
# - `Additional_Work`
# - `Regular_Artistic_or_Sports`, and 
# - `Has_Partner` 
# 
# contain binary **Yes** or **No** values in the DataFrame `df`.
# 
# Label-encode all three columns. That is, convert all **Yes** values to **1** and all **No** values to **0** in each column. 
# 
# Perform the label-encoding in place. That is, after label encoding, the original columns should contain the label encoded data (as opposed to creating new columns to store the label encoded data).
# 
# |Student_ID|Additional_Work|Regular_Artistic_or_Sports|Has_Partner|
# |--|--|--|--|
# |1|Yes|No|No|
# |2|Yes|Yes|Yes|
# |3|No|Yes|No|
# |4|Yes|No|Yes|
# |5|No|No|Yes|
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[4]:


# create the dataframe df
df = pd.DataFrame({&#39;Student_ID&#39;:[1,2,3,4,5],
                   &#39;Additional_Work&#39;:[&#39;Yes&#39;,&#39;Yes&#39;,&#39;No&#39;,&#39;Yes&#39;,&#39;No&#39;],
                   &#39;Regular_Artistic_or_Sports&#39;:[&#39;No&#39;,&#39;Yes&#39;,&#39;Yes&#39;,&#39;No&#39;,&#39;No&#39;],
                   &#39;Has_Partner&#39;:[&#39;No&#39;,&#39;Yes&#39;,&#39;No&#39;,&#39;Yes&#39;,&#39;Yes&#39;]})

## YOUR SOLUTION HERE ##
# Replace &#34;Yes&#34; with 1 and &#34;No&#34; with 0 for the specified columns
df[&#39;Additional_Work&#39;] = df[&#39;Additional_Work&#39;].replace({&#39;Yes&#39;: 1, &#39;No&#39;: 0})
df[&#39;Regular_Artistic_or_Sports&#39;] = df[&#39;Regular_Artistic_or_Sports&#39;].replace({&#39;Yes&#39;: 1, &#39;No&#39;: 0})
df[&#39;Has_Partner&#39;] = df[&#39;Has_Partner&#39;].replace({&#39;Yes&#39;: 1, &#39;No&#39;: 0})

# show encoded output
df.head()


# #### Checkpoint 3/3
# 
# The column `MT1_Preparation` contains categorical data on how students prepared for the first midterm of the course. Students may have either prepared &#34;Alone&#34;, &#34;With Friends&#34;, or &#34;Not Applicable&#34; if they didn&#39;t prepare at all.
# 
# |Student_ID|High_School_Type|MT1_Preparation|
# |--|--|--|
# |1|State|Alone|
# |2|Private|Alone|
# |3|Other|With Friends|
# |4|State|Alone|
# |5|State|Not Applicable|
# 
# One-hot encode the `MT1_Preparation` column in the DataFrame `df`. Be sure to return the one-hot encodings as integers and save the one-hot encoded DataFrame back to the variable `df`. 
# 
# Don&#39;t forget to run the cell and save the notebook before selecting `Test Work`! Open the `Jupyter Help` toggle at the top of the notebook for more details.

# In[5]:


# create the dataframe df
df = pd.DataFrame({&#39;Student_ID&#39;:[1,2,3,4,5],
                   &#39;MT1_Preparation&#39;:[&#39;Alone&#39;, &#39;Alone&#39;, &#39;With Friends&#39;, &#39;Alone&#39;, &#39;Not Applicable&#39;]})

## YOUR SOLUTION HERE ##
# One-hot encode the MT1_Preparation column
df = pd.get_dummies(df, columns=[&#39;MT1_Preparation&#39;], dtype=int)

# show encoded output
df.head()


# 
# **Moving Forward** 
# 
# For the rest of the lesson, we&#39;ll work with the full student performance dataset, with all these encodings (and more) already applied. 
</details><script type="text/javascript" src="/relay.js"></script></body></html>
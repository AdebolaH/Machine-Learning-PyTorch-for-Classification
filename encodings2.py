Jupyter Notebook
notebook
Last Checkpoint: 09/20/2025
(autosaved)
Current Kernel Logo
Python 3 (ipykernel) 
File
Edit
View
Insert
Cell
Kernel
Help

Markdown
Jupyter Help
Checkpoint 1/3
Follow the instructions beneath the exercise narrative to import pandas and practice testing your work with our code testing system!

import pandas as pd
Example - Label Encoding

Run the cell below to label-encode the Letter_Grade and Outcome columns of the tabel below:

Student_ID	Letter_Grade	Outcome
1	A	Passed
2	C	Passed
3	F	Failed
4	B	Passed
5	D	Failed
# create the sample dataframe
df = pd.DataFrame({'Student_ID':[1,2,3,4,5], 
                   'Letter_Grade':['A','C','F','B','D'],
                   'Outcome':['Passed','Passed','Failed','Passed','Failed']})
​
# Label encode Letter_Grade and Outcome columns
df['Letter_Grade'] = df['Letter_Grade'].replace(
    {'A':4, 
    'B':3, 
    'C':2, 
    'D':1, 
    'F':0})
​
df['Outcome'] = df['Outcome'].replace(
    {'Passed':1, 
     'Failed':0})
​
df.head()
Student_ID	Letter_Grade	Outcome
0	1	4	1
1	2	2	1
2	3	0	0
3	4	3	1
4	5	1	0
Example - One-Hot Encoding

Run the cell below to one-hot encode the High_School_Type column of the table below.

Student_ID	High_School_Type
1	State
2	Private
3	Other
4	State
5	State
# create the sample dataframe
df = pd.DataFrame({'Student_ID':[1,2,3,4,5],
                   'High_School_Type':['State','Private','Other','State', 'State']})
​
# One-hot encode High_School_Type column
df = pd.get_dummies(
    df, 
    columns=['High_School_Type'], 
    dtype=int)
​
df.head()
Student_ID	High_School_Type_Other	High_School_Type_Private	High_School_Type_State
0	1	0	0	1
1	2	0	1	0
2	3	1	0	0
3	4	0	0	1
4	5	0	0	1
Checkpoint 2/3
The categorical columns

Additional_Work
Regular_Artistic_or_Sports, and
Has_Partner
contain binary Yes or No values in the DataFrame df.

Label-encode all three columns. That is, convert all Yes values to 1 and all No values to 0 in each column.

Perform the label-encoding in place. That is, after label encoding, the original columns should contain the label encoded data (as opposed to creating new columns to store the label encoded data).

Student_ID	Additional_Work	Regular_Artistic_or_Sports	Has_Partner
1	Yes	No	No
2	Yes	Yes	Yes
3	No	Yes	No
4	Yes	No	Yes
5	No	No	Yes
Don't forget to run the cell and save the notebook before selecting Test Work! Open the Jupyter Help toggle at the top of the notebook for more details.

# create the dataframe df
df = pd.DataFrame({'Student_ID':[1,2,3,4,5],
                   'Additional_Work':['Yes','Yes','No','Yes','No'],
                   'Regular_Artistic_or_Sports':['No','Yes','Yes','No','No'],
                   'Has_Partner':['No','Yes','No','Yes','Yes']})
​
## YOUR SOLUTION HERE ##
# Replace "Yes" with 1 and "No" with 0 for the specified columns
df['Additional_Work'] = df['Additional_Work'].replace({'Yes': 1, 'No': 0})
df['Regular_Artistic_or_Sports'] = df['Regular_Artistic_or_Sports'].replace({'Yes': 1, 'No': 0})
df['Has_Partner'] = df['Has_Partner'].replace({'Yes': 1, 'No': 0})
​
# show encoded output
df.head()
Student_ID	Additional_Work	Regular_Artistic_or_Sports	Has_Partner
0	1	1	0	0
1	2	1	1	1
2	3	0	1	0
3	4	1	0	1
4	5	0	0	1
Checkpoint 3/3
The column MT1_Preparation contains categorical data on how students prepared for the first midterm of the course. Students may have either prepared "Alone", "With Friends", or "Not Applicable" if they didn't prepare at all.

Student_ID	High_School_Type	MT1_Preparation
1	State	Alone
2	Private	Alone
3	Other	With Friends
4	State	Alone
5	State	Not Applicable
One-hot encode the MT1_Preparation column in the DataFrame df. Be sure to return the one-hot encodings as integers and save the one-hot encoded DataFrame back to the variable df.

Don't forget to run the cell and save the notebook before selecting Test Work! Open the Jupyter Help toggle at the top of the notebook for more details.

# create the dataframe df
df = pd.DataFrame({'Student_ID':[1,2,3,4,5],
                   'MT1_Preparation':['Alone', 'Alone', 'With Friends', 'Alone', 'Not Applicable']})
​
## YOUR SOLUTION HERE ##
# One-hot encode the MT1_Preparation column
df = pd.get_dummies(df, columns=['MT1_Preparation'], dtype=int)
​
# show encoded output
df.head()
Student_ID	MT1_Preparation_Alone	MT1_Preparation_Not Applicable	MT1_Preparation_With Friends
0	1	1	0	0
1	2	1	0	0
2	3	0	0	1
3	4	1	0	0
4	5	0	1	0
Moving Forward

For the rest of the lesson, we'll work with the full student performance dataset, with all these encodings (and more) already applied.

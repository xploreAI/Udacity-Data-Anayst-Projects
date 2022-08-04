#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset - [TMDb_Movies Dataset]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# > I'll be using TMDb movies dataset for my Data Analysis Project
# 
# The dataset contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue. It consist of 21 columns such as id	imdb_id	popularity	budget	revenue	original_title	cast	homepage	director	tagline		overview	runtime	genres	production_companies	release_date	vote_count	vote_average	release_year	budget_adj	revenue_adj etc
# 
# ### Questions for  My Analysis
# 
# 1. Average Runtime Of Movies From Year To Year?
# 2. Which director produces the most movies?
# 3. Movie details with most and least earned revenue
# 4. Movie details with the most popularity rating and least popularity rating 
# 5. Relationship between popularity and profit earned
# 6. Movie details which had most and least profit
# 

# In[1]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html

#importing important libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# >I plan to further observe the TMDB dataset, after that I will be keeping only relevant data needed for analysis and deleting irrelevant data to ensure calculations and analysis are easier
# 
# 
# 

# In[3]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.

#Loading csv file and assigning it to df_movies variable
df_movies = pd.read_csv("tmdb-movies.csv")

#Displying first 5 rows and columns
df_movies.head(5)


# In[4]:


df_movies.info()


# In[5]:


#checking number of rows and columns
df_movies.shape


# In[6]:


#Finding duplicates
df_movies[df_movies.duplicated()]


# In[7]:


#Confirming duplicated rows
df_movies.iloc[2089:2091 ]


# In[8]:


#Finding NULL Values in dataset
df_movies.isnull().sum()


# In[9]:


#checking if runtime column has zero values recorded in it
df_movies.query('runtime == 0')


# In[10]:


# statistical values summary 
df_movies.describe()


# 
# ### Obersvations from TMDB dataset
# 1. Discovered row 2089 and 2090 have the same values making them duplicates
# 2. After making a query to check whether some movies have zero runtime I discovered some which I'll be replacing with nan values
# 

# 
# ### Data Cleaning
#  
# > 1. There's need to remove duplicated rows
# > 2. There's need to remove irrelevant column such as id,imdb_id, homepage,overview and some others
# > 3. Release date column needs to be converted to date format for better analysis
# > 4. Entries having zero in runtime column needs to be replaced with NAN values
# > 5. Removing Zero values from revenue columns
#  

# 1. Removing Duplicated rows
# > From the data wrangling section I found out there is only one duplicated row

# In[11]:


df_movies.drop_duplicates(keep=False, inplace=True)


# In[12]:


#re-checking duplicates
df_movies.duplicated().sum()


# 2. Removing irrelvant columns
# > columns to be removed are - 'id','imdb_id','homepage','overview', 'tagline', 'keywords'

# In[13]:


#Dropping unnecessary columns 
df_movies.drop(['id','imdb_id','homepage','overview', 'tagline', 'keywords','vote_count', 'vote_average'], axis=1, inplace=True)


# In[14]:


#confriming dropped columns
df_movies.columns


# In[15]:


#filling missing values with respective mean 
df_movies.fillna(df_movies.mean(), inplace=True)
df_movies.info()


# In[16]:


df_movies.isnull().sum()


# In[17]:


#Dropping null values
df_movies.dropna(inplace=True)


# In[18]:


df_movies.info()


# 3. Changing release_date column into the standard format

# In[19]:


df_movies.release_date = pd.to_datetime(df_movies['release_date'])


# In[20]:


#Displaying edited dataset
df_movies.head(5)


# 4. Replacing zero values in runtime column with NAN 

# In[21]:


df_movies.runtime = df_movies.runtime.replace(0, np.NAN)


# In[22]:


#checking if zero values in runtime still exist in the dataset
df_movies.query('runtime == 0')


# 5. Removing Zero values from revenue columns

# In[34]:


#replacing all zero values to NAN in revenue column
df_movies.revenue = df_movies.revenue.replace(0, np.NAN)

#Now removing rows with NAN value in revenue column
df_movies.revenue.dropna(inplace=True)


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# 
# 
# ### Research Question 1: Average Runtime Of Movies From Year To Year?

# In[23]:


#group dataset with respect to the release_year and calculate mean runtime values
year_run = df_movies.groupby('release_year').mean()['runtime']


# In[24]:


#graph setup
sns.set_style("whitegrid")
year_run.plot(color='green', xticks = np.arange(1960,2016,5))
plt.Figure(figsize=(12,8))
plt.title("Avg_Runtime VS Year", fontsize=15)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Avg_Runtime",fontsize=12)


# > __Judging from the plot above it clearly shows that movie runtime/duration decreases from year to year which as of today its very True. People don't spend time watching long durated movies anymore__

# ### Research Question 2: Which director produces the most movies?

# In[25]:



director = df_movies.director.value_counts()
print(director.head(10))


# > From the above analysis Woody Allen produces the most movies

# ### Research Question 3: Movie details with most and least earned revenue

# In[39]:


#Function Definition
def compute(column):
    #Calculate most earned revenue
    high = df_movies[column].idxmax()
    h_details = pd.DataFrame(df_movies.loc[high])
    #Calculate least earned revenue
    low = df_movies[column].idxmin()
    l_details = pd.DataFrame(df_movies.loc[low])

    #Group collected data
    details = pd.concat([h_details, l_details], axis=1)

    return details

compute('revenue')


# The information above clearly shows that row 1386 with original_title Avatar has the most revenue earned i.e 2781505847.0.
# 
# And also clearly shows the row 5067 with original_title Shattered Glass has the least revenue earned i.e 2.0

# ### Research Question 4: Movie details with the most popularity rating and least popularity rating 

# In[41]:


#calling compute function created earlier 
compute('popularity')


# row 0 is considered the movie with the most popular rating and row 9977 with the least

# ### Research Question 5: Relationship between popularity and profit earned

# In[42]:


#Creating profit_earned column
df_movies['profit_earned'] = df_movies['revenue'] - df_movies['budget']


# In[57]:


#computing popluarity mean
mean_val = df_movies.popularity.mean()
low_popular = df_movies.query('popularity < {}'.format(mean_val))
high_popular = df_movies.query('popularity >= {}'.format(mean_val))


# In[58]:


#Average profit_earned for low_popular and high_popular
avg_profit_low = low_popular['profit_earned'].mean()
avg_profit_high = high_popular['profit_earned'].mean()


# In[59]:


#Displaying top dataset
df_movies.head()


# In[62]:


#Plotting a bar graph for the relationship
heights = [avg_profit_low,avg_profit_high]
locations = [1,2]
labels = ['low','high']
plt.title('Average Profit Earned By Popluarity Rating')
plt.xlabel('Popularity')
plt.ylabel('Average Profit Earned')
plt.bar(locations, heights, tick_label=labels)


# This bar chart above expliitly shows that higher popularity rating leads to more average profit earned and vice-versa

# ### Research Question 6: Movie details which had most and least profit

# In[64]:


#calling the compute function once again 
#to provide details for most profit earned and least profit earned
compute('profit_earned')


# Row 1386 consist of movie details with the highest profit earned and Row 2244 with the least profit earned details

# <a id='conclusions'></a>
# ## Conclusions
# An interesting analysis process with the TMDB dataset.
# I was able to discover some very interesting facts about movies. 
# 
# 1. Duration of movies decrease from year to year 
# 2. Woody Allen is the director that produces the most movies with total count of 42 and Clint Eastwood with total count of 34
# 3. Avatar with runtime 162 and release year 2009 is known to have the most revenue of 2781505847.0 and James Cameron as the director, whereas Shattered Glass with runtime 94 and release year 2003 is known have the least revenue of 2 and Billy Ray as the director
# 4. Jurassic World with runtime 124 and release year 2015 is known to have the highest popularity rating of 32.985763 and Colin Trevorrow as the director, whereas The Hospital with runtime 103 and release year 1971 is known have the lowest popularity rating of 0.000188 and Arthur Hiller as the director
# 5. Higher popularity rating leads to more Average profit Earned
# 
# ## Limitations
# Missing values as well as  most zero values present in the dataset affected the analysis process.
# The budget and revenue column are not formatted in any currency unit, there's possibilty that different movies have budget in different currency according to the country they are produce in.
# 
# ## Resources Used
# Stackoverflow,Medium,Pandas and Numpy Documentation,geeksforgeeks.

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


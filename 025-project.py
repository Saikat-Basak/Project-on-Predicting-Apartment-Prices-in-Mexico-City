#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>Assignment: Predicting Apartment Prices in Mexico City 🇲🇽</strong></font>

# In[26]:


import warnings

import wqet_grader

warnings.simplefilter(action="ignore", category=FutureWarning)
wqet_grader.init("Project 2 Assessment")


# <div class="alert alert-block alert-warning">
#     <b>Note:</b> In this project there are graded tasks in both the lesson notebooks and in this assignment. Together they total 24 points. The minimum score you need to move to the next project is 22 points. Once you get 22 points, you will be enrolled automatically in the next project, and this assignment will be closed. This means that you might not be able to complete the last two tasks in this notebook. If you get an error message saying that you've already passed the course, that's good news. You can stop this assignment and move onto the project 3. 
# </div>

# In this assignment, you'll decide which libraries you need to complete the tasks. You can import them in the cell below. 👇

# In[35]:


# Import libraries here
from glob import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted


# # Prepare Data

# ## Import

# **Task 2.5.1:** (8 points) Write a `wrangle` function that takes the name of a CSV file as input and returns a DataFrame. The function should do the following steps:
# 
# 1. Subset the data in the CSV file and return only apartments in Mexico City (`"Distrito Federal"`) that cost less than \$100,000.
# 2. Remove outliers by trimming the bottom and top 10\% of properties in terms of `"surface_covered_in_m2"`.
# 3. Create separate `"lat"` and `"lon"` columns.
# 4. Mexico City is divided into [16 boroughs](https://en.wikipedia.org/wiki/Boroughs_of_Mexico_City). Create a `"borough"` feature from the `"place_with_parent_names"` column.
# 5. Drop columns that are more than 50\% null values.
# 6. Drop columns containing low- or high-cardinality categorical values. 
# 7. Drop any columns that would constitute leakage for the target `"price_aprox_usd"`.
# 8. Drop any columns that would create issues of multicollinearity. 
# 
# <div class="alert alert-block alert-info">
# <b>Tip:</b> Don't try to satisfy all the criteria in the first version of your <code>wrangle</code> function. Instead, work iteratively. Start with the first criteria, test it out with one of the Mexico CSV files in the <code>data/</code> directory, and submit it to the grader for feedback. Then add the next criteria.</div>

# In[10]:


pd.read_csv("data/mexico-city-real-estate-1.csv").head()


# In[22]:


# Build your `wrangle` function
def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)

    # Subset data: Apartments in "Distrito Federal", less than 100,000
    mask_ba = df["place_with_parent_names"].str.contains("Distrito Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 100_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    # Get place name
    df["borough"] = df["place_with_parent_names"].str.split("|", expand=True)[1]
    df.drop(columns="place_with_parent_names", inplace=True)
    

    # Drop features with high NaN values
    df.drop(columns=["surface_total_in_m2","price_usd_per_m2","floor","rooms","expenses"], inplace=True)
    
    # Drop low or high cardinality columns
    df.drop(columns=["operation","property_type","currency","properati_url"], inplace=True)
    
    # Drop leaky columns
    df.drop(columns=['price',
                     'price_aprox_local_currency',
                     'price_per_m2'], inplace=True)
    """"
    # Drop columns with Multicollinearity
    df.drop(columns=["surface_total_in_m2","rooms"], inplace=True)
    """
    
    return df


# In[23]:


df1 = wrangle("data/mexico-city-real-estate-1.csv")
df1.head()


# In[29]:


df1.info()
#df1.isnull().sum() / len(df1) *100
#df1.select_dtypes("object").nunique()


# In[ ]:


# Use this cell to test your wrangle function and explore the data


# In[27]:



wqet_grader.grade(
    "Project 2 Assessment", "Task 2.5.1", wrangle("data/mexico-city-real-estate-1.csv")
)


# **Task 2.5.2:** Use glob to create the list `files`. It should contain the filenames of all the Mexico City real estate CSVs in the `./data` directory, except for `mexico-city-test-features.csv`.

# In[30]:


files = glob("data/mexico-city-real-estate-*.csv")
files


# In[31]:


wqet_grader.grade("Project 2 Assessment", "Task 2.5.2", files)


# **Task 2.5.3:** Combine your `wrangle` function, a list comprehension, and `pd.concat` to create a DataFrame `df`. It should contain all the properties from the five CSVs in `files`. 

# In[32]:


frames = [wrangle(file) for file in files]
df = pd.concat(frames, ignore_index=True)
print(df.info())
df.head()


# In[33]:



wqet_grader.grade("Project 2 Assessment", "Task 2.5.3", df)


# ## Explore

# **Task 2.5.4:** Create a histogram showing the distribution of apartment prices (`"price_aprox_usd"`) in `df`. Be sure to label the x-axis `"Area [sq meters]"`, the y-axis `"Count"`, and give it the title `"Distribution of Apartment Prices"`.
# 
# What does the distribution of price look like? Is the data normal, a little skewed, or very skewed?

# In[38]:


# Plot distribution of price
plt.hist(df["price_aprox_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Count")
plt.title("Distribution of Apartment Prices")
# Don't delete the code below 👇
plt.savefig("images/2-5-4.png", dpi=150)


# In[39]:


with open("images/2-5-4.png", "rb") as file:
    wqet_grader.grade("Project 2 Assessment", "Task 2.5.4", file)


# **Task 2.5.5:** Create a scatter plot that shows apartment price (`"price_aprox_usd"`) as a function of apartment size (`"surface_covered_in_m2"`). Be sure to label your axes `"Price [USD]"` and `"Area [sq meters]"`, respectively. Your plot should have the title `"Mexico City: Price vs. Area"`.
# 
# Do you see a relationship between price and area in the data? How is this similar to or different from the Buenos Aires dataset?

# In[42]:


# Plot price vs area
plt.scatter(x=df["surface_covered_in_m2"], y=df["price_aprox_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")
plt.title("Mexico City: Price vs. Area");
# Don't delete the code below 👇
plt.savefig("images/2-5-5.png", dpi=150)


# In[43]:


with open("images/2-5-5.png", "rb") as file:
    wqet_grader.grade("Project 2 Assessment", "Task 2.5.5", file)


# **Task 2.5.6:** **(UNGRADED)** Create a Mapbox scatter plot that shows the location of the apartments in your dataset and represent their price using color. 
# 
# What areas of the city seem to have higher real estate prices?

# In[ ]:


# Plot Mapbox location and price


# ## Split

# **Task 2.5.7:** Create your feature matrix `X_train` and target vector `y_train`. Your target is `"price_aprox_usd"`. Your features should be all the columns that remain in the DataFrame you cleaned above.

# In[50]:


# Split data into feature matrix `X_train` and target vector `y_train`.
features = ["surface_covered_in_m2","lat","lon","borough"]
target = "price_aprox_usd"
X_train = df[features]
y_train = df[target]
X_train.shape


# In[45]:



wqet_grader.grade("Project 2 Assessment", "Task 2.5.7a", X_train)


# In[51]:



wqet_grader.grade("Project 2 Assessment", "Task 2.5.7b", y_train)


# # Build Model

# ## Baseline

# **Task 2.5.8:** Calculate the baseline mean absolute error for your model.

# In[52]:


y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
print("Mean apt price:", y_mean)
print("Baseline MAE:", baseline_mae)


# In[53]:


wqet_grader.grade("Project 2 Assessment", "Task 2.5.8", [baseline_mae])


# ## Iterate

# **Task 2.5.9:** Create a pipeline named `model` that contains all the transformers necessary for this dataset and one of the predictors you've used during this project. Then fit your model to the training data.

# In[102]:


# Build Model
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge()
)
# Fit model
model.fit(X_train, y_train)


# In[56]:



wqet_grader.grade("Project 2 Assessment", "Task 2.5.9", model)


# ## Evaluate

# **Task 2.5.10:** Read the CSV file `mexico-city-test-features.csv` into the DataFrame `X_test`.

# <div class="alert alert-block alert-info">
# <b>Tip:</b> Make sure the <code>X_train</code> you used to train your model has the same column order as <code>X_test</code>. Otherwise, it may hurt your model's performance.
# </div>

# In[61]:


pd.read_csv("data/mexico-city-test-features.csv").head()


# In[62]:


X_test = pd.read_csv("data/mexico-city-test-features.csv")
print(X_test.info())
X_test.head()


# In[63]:



wqet_grader.grade("Project 2 Assessment", "Task 2.5.10", X_test)


# **Task 2.5.11:** Use your model to generate a Series of predictions for `X_test`. When you submit your predictions to the grader, it will calculate the mean absolute error for your model.

# In[65]:


y_test_pred = pd.Series(model.predict(X_test))
y_test_pred.head()


# In[66]:


wqet_grader.grade("Project 2 Assessment", "Task 2.5.11", y_test_pred)


# # Communicate Results

# **Task 2.5.12:** Create a Series named `feat_imp`. The index should contain the names of all the features your model considers when making predictions; the values should be the coefficient values associated with each feature. The Series should be sorted ascending by absolute value.  

# In[103]:


coefficients = model.named_steps["ridge"].coef_.round(2)
features = model.named_steps["onehotencoder"].get_feature_names()
feat_imp = pd.Series(coefficients, index=features).sort_values(key=abs)
feat_imp


# In[104]:



wqet_grader.grade("Project 2 Assessment", "Task 2.5.13", feat_imp)


# **Task 2.5.13:** Create a horizontal bar chart that shows the **10 most influential** coefficients for your model. Be sure to label your x- and y-axis `"Importance [USD]"` and `"Feature"`, respectively, and give your chart the title `"Feature Importances for Apartment Price"`.

# In[105]:


feat_imp.sort_values(key=abs).tail(10).plot(
    kind="barh",
    xlabel="Importance [USD]",
    ylabel="Feature",
    title="Feature Importances for Apartment Price");


# In[106]:


# Create horizontal bar chart
feat_imp.sort_values(key=abs).tail(10).plot(
    kind="barh",
    xlabel="Importance [USD]",
    ylabel="Feature",
    title="Feature Importances for Apartment Price");
# Don't delete the code below 👇
plt.savefig("images/2-5-14.png", dpi=150)


# In[107]:


with open("images/2-5-14.png", "rb") as file:
    wqet_grader.grade("Project 2 Assessment", "Task 2.5.14", file)


# ---
# Copyright © 2022 WorldQuant University. This
# content is licensed solely for personal use. Redistribution or
# publication of this material is strictly prohibited.
# 

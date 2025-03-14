# Bechdel-Analysis
## How I Discovered the Bechdel Test
While watching a TV series, I found myself asking, “Why do I see more male characters than female characters in so many movies and shows?” That question led me down a rabbit hole of research, where I came across the **Bechdel Test**. It’s a straightforward measure that answers the question: *Does a movie have at least two women who talk to each other about something other than a man?* 

I learned that an alarming number of popular movies fail to meet these three simple criteria. Although passing the Bechdel Test doesn’t automatically mean a film is feminist or free from problematic portrayals, it does highlight how underrepresented or sidelined women can be in mainstream media. I decided to integrate this test into my analysis to see what the data might reveal.
![bechdel](https://github.com/user-attachments/assets/bf9a1ec9-68cd-43be-8343-e99a9233bd15)

---
The notebook is organized into the following sections:

1. **Project Overview**  
2. **Import Libraries**  
3. **Load Data**  
4. **Exploratory Data Analysis (EDA)**  
   - Univariate Analysis  
   - Categorical Feature Selection  
   - Separating Features into Categories  
   - Numerical Features  
   - Key Insights from the Distribution  
   - Categorical Features  
   - Key Insights from the Distribution  
   - Bivariate Analysis  

---

## 1. Project Overview

### 1.1 Dataset Description
The project utilizes a dataset of movies containing various features such as:
- **Title**  
- **Year**  
- **IMDb Rating**  
- **Budget**  
- **Worldwide Gross**  
- **Duration**  
- **Bechdel Test Outcome** (indicating whether each movie passes or fails the Bechdel Test)

The author uploaded a notebook (`exploring-bechdel-in-movies-through-eda-ml.ipynb`) containing the code used to load, clean, and analyze the data. This dataset helps address questions such as:
- How many movies pass vs. fail the Bechdel Test?
- How has the proportion of passing movies changed over time?
- Are there any notable differences in budget, revenue, or ratings between movies that pass or fail the test?

### 1.2 Goal of the Project
The primary goals include:
1. Investigating the distribution and trends of movies passing the Bechdel Test.  
2. Identifying potential relationships between the Bechdel Test outcome and other movie attributes (e.g., budget, box office returns, ratings).  
3. Increasing awareness of gender representation in films and demonstrating how the Bechdel Test can highlight possible gender imbalances.

---

## 2. Import Libraries

The notebook begins by importing the necessary Python libraries for data analysis and visualization:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Depending on the depth of the analysis, additional libraries (e.g., `sklearn`) may be included for machine learning or advanced statistics.

---

## 3. Load Data

In this section, the author loads the dataset (e.g., from a CSV file) and performs basic data checks.

### 3.1 Displaying a Random Sample of the Data
```python
df.sample(5)
```
- Shows a random subset of rows to get an initial sense of the data.

### 3.2 Displaying Data Information
```python
df.info()
```
- Provides details on column names, data types, and any missing values.

### 3.3 Cleaning Column Names
The author ensures column names are consistent, removing any special characters or spaces:
```python
df.columns = df.columns.str.lower().str.replace(' ', '_')
```

### 3.4 Checking for Duplicated Rows
```python
df.duplicated().sum()
```
- Identifies any duplicate entries that may need to be removed or otherwise handled.

---

## 4. Exploratory Data Analysis (EDA)

This section contains various methods and visualizations to understand the data and uncover relationships among features. The uploaded notebook includes detailed code for each step.

### 4.1 Univariate Analysis
### Count Plot of Bechdel Test Results

**Question:**  
*How many movies in the dataset pass or fail the Bechdel Test?*

**Answer:**  
The count plot displays the number of movies that pass or fail the Bechdel Test by showing a bar chart with each category’s count.

**Explanation:**  
The code uses `sns.countplot` on the column `'bechdel_rating'` to count and visualize the number of movies in each category. This simple chart immediately reveals which outcome (pass or fail) is more common in the dataset.

```python
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='bechdel_rating') 
plt.title('Count of Movies by Bechdel Test Result')
plt.xlabel('')
plt.ylabel('Number of Movies')
plt.show()
```
![image](https://github.com/user-attachments/assets/d8e649c2-c548-4986-b4b4-280ba3df6fd6)

---

### Trend Over Years of Movies Passing/Failing the Bechdel Test

**Question:**  
*How has the distribution of movies passing or failing the Bechdel Test changed over time?*

**Answer:**  
The line plot shows the yearly trend for the count of movies that pass or fail the test, highlighting any shifts or trends in representation over the years.

**Explanation:**  
The code first groups the data by `year` and `bechdel_rating` (or the binary equivalent) to calculate the yearly counts. Then, it uses `sns.lineplot` to draw a line chart with different lines (or hues) for each Bechdel Test outcome. This visualization helps determine if the number of movies meeting the criteria is increasing, decreasing, or remaining stable over time.

```python
# Group data by year and test result then reset index for plotting
df_yearly = df.groupby(['year', 'bechdel_rating']).size().reset_index(name='count')
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_yearly, x='year', y='count', hue='binary', marker='o')
plt.title('Trend of Movies Passing/Failing the Bechdel Test Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.legend(title='Bechdel Test Result')
plt.show()
```
![image](https://github.com/user-attachments/assets/4dfdfa2c-550f-460f-b6b4-3b521f62d59f)

---

### Box Plot of IMDb Ratings by Bechdel Test Result

**Question:**  
*Do movies that pass the Bechdel Test differ in their IMDb ratings compared to those that fail?*

**Answer:**  
The box plot compares the distribution of IMDb ratings for movies based on their Bechdel Test outcome, showing differences in median ratings, quartile ranges, and outliers between the groups.

**Explanation:**  
Using `sns.boxplot`, the code maps `'bechdel_rating'` to the x-axis and `'imdb_rating'` to the y-axis. This visualization provides insights into the central tendency and variability of ratings for each group, which can indicate if passing the test is associated with higher or lower average ratings.

```python
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='bechdel_rating', y='imdb_rating')
plt.title('IMDb Ratings by Bechdel Test Result')
plt.xlabel('Bechdel Test Result')
plt.ylabel('IMDb Rating')
plt.show()
```
![image](https://github.com/user-attachments/assets/72cba0bb-b9e8-49b9-b078-c70500ca1d45)

---

### Correlation Heatmap of Numerical Features

**Question:**  
*What relationships exist among the numerical features (e.g., year, budget, IMDb rating) in the dataset?*

**Answer:**  
The correlation heatmap shows the strength and direction of the relationships between various numerical features. For example, it may highlight if higher budgets are linked with higher worldwide grosses or if certain variables tend to increase together.

**Explanation:**  
The code selects numerical columns from the DataFrame, computes their pairwise correlations, and then visualizes these correlations using `sns.heatmap` with annotations. The heatmap uses color intensity to indicate the magnitude of correlations, making it easy to spot significant positive or negative relationships among the features.

```python
plt.figure(figsize=(10, 8))
# Select only numerical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
![image](https://github.com/user-attachments/assets/78f51763-7472-4566-80bc-688957b0979a)

```

# Conclusion
In summary, this notebook:
- Demonstrates how to systematically load, clean, and explore a movie dataset.  
- Integrates the **Bechdel Test** to shed light on gender representation in film.  
- Uses a variety of EDA techniques (univariate, bivariate, and multivariate analyses) to reveal insights about how movies that pass or fail the test differ in terms of ratings, budgets, and box office performance.

By focusing on the Bechdel Test, the author not only hones their data science skills but also contributes to an ongoing conversation about inclusivity and diversity in storytelling.

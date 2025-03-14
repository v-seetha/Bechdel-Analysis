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

#### 4.1.1 Distribution of Bechdel Test Results
```python
sns.countplot(data=df, x='bechdel_result')  # or 'binary' if labeled differently
plt.title('Count of Movies by Bechdel Test Result')
plt.show()
```
**Insight**: Reveals the proportion of movies passing versus failing the Bechdel Test.
![image](https://github.com/user-attachments/assets/4d6a1ef4-590c-4ce4-8145-527db583fe89)


#### 4.1.2 Distribution of IMDb Ratings
```python
sns.histplot(df['imdb_rating'], bins=30, kde=True)
plt.title('Distribution of IMDb Ratings')
plt.show()
```
**Insight**: Shows the average rating and indicates whether ratings are skewed.
![image](https://github.com/user-attachments/assets/aa55335a-c272-4cdd-b550-64d4f8caced5)

### 4.2 Categorical Feature Selection
If the dataset contains variables like **genre** or **IMBD rating**, they are flagged as categorical to facilitate group-based analyses.

### 4.3 Separating Features into Categories
The features are explicitly divided into:
- **Numerical features** (e.g., `year`, `budget`, `worldwide_gross`, `imdb_rating`)  
- **Categorical features** (e.g., `genre`, `mpaa_rating`, `bechdel_result`)
![image](https://github.com/user-attachments/assets/d2b1dea9-daaf-48cd-9b67-96f0d76f6309)

### 4.4 Numerical Features

#### 4.4.1 Summary Statistics
```python
df.describe()
```
- Displays mean, median, minimum, and maximum values for each numerical column.

#### 4.4.2 Correlation Heatmap
```python
numeric_cols = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
```
**Insight**: Helps determine whether variables such as `budget` and `duration` correlate with each other or with the Bechdel Test outcome (if encoded as 0/1).
![image](https://github.com/user-attachments/assets/b5c99361-2a2e-418b-9641-d56e7fd525c0)

### 4.5 Key Insights from the Distribution
Based on the distributions (e.g., histograms, box plots), any noteworthy patterns are highlighted. For instance, the author might observe that newer movies tend to have a higher Bechdel pass rate.

### 4.6 Categorical Features
For columns like **genre** or **director**, the author inspects frequency distributions and group-specific trends:

```python
df['genre'].value_counts()
```
**Insight**: Shows which genres are most prevalent and whether certain genres pass the Bechdel Test more often.

---

## How the Bechdel Test Was Discovered

The author realized a gender imbalance while watching a TV series and noting a shortage of meaningful female characters. This observation prompted a deeper question: *“Why are there more male characters than female characters in most movies?”* Further exploration led them to the **Bechdel Test**, which highlights whether a film:

1. Features at least **two women**,  
2. Who **talk to each other**,  
3. **About something other than a man**.

Although passing the Bechdel Test does not guarantee that a movie is feminist or free of problematic representations, it does highlight a recurring issue: many films do not include women with substantive roles or dialogues. By incorporating the Bechdel Test outcome into data analyses, one can identify broader trends and encourage discussions about improving female representation in media.

---

# Conclusion
In summary, this notebook:
- Demonstrates how to systematically load, clean, and explore a movie dataset.  
- Integrates the **Bechdel Test** to shed light on gender representation in film.  
- Uses a variety of EDA techniques (univariate, bivariate, and multivariate analyses) to reveal insights about how movies that pass or fail the test differ in terms of ratings, budgets, and box office performance.

By focusing on the Bechdel Test, the author not only hones their data science skills but also contributes to an ongoing conversation about inclusivity and diversity in storytelling.

# 🚢 Titanic — Exploratory Data Analysis

A thorough, end-to-end exploratory data analysis of the Titanic dataset. The goal of this notebook is to deeply understand the data before any modelling — uncovering distributions, relationships, missing values, and potential features that could improve a survival prediction model.

---

## 📌 Why EDA?

Before jumping into model building, EDA is done to:

- **Understand the data** — what columns exist, what types they are, what they represent
- **Validate assumptions** — check whether the data behaves the way you'd expect
- **Handle missing values** — find where data is absent and decide how to deal with it
- **Detect outliers** — spot values that are unusually high or low
- **Guide feature engineering** — identify opportunities to create more informative columns
- **Support analysis and reporting** — build a factual picture of who was on the Titanic and what influenced survival

---

## 📁 Dataset

The standard Titanic dataset from Kaggle, split into:

- `train.csv` — 891 passenger records with a `Survived` label (used for analysis)
- `test.csv` — 418 records (merged during feature engineering for a fuller picture)

**Columns broken down by type:**

| Type | Columns |
|------|---------|
| Numerical | `Age`, `Fare`, `PassengerId` |
| Categorical | `Survived`, `Pclass`, `Sex`, `SibSp`, `Parch`, `Embarked` |
| Mixed (text) | `Name`, `Ticket`, `Cabin` |

---

## 🔍 Analysis Walkthrough

### Step 1 — Univariate Analysis (Numerical Columns)

Univariate analysis looks at each column in isolation to understand its distribution, central tendency, spread, and any anomalies.

For each numerical column, the process was:
1. **Descriptive statistics** — `mean`, `median`, `std`, `min`, `max`, `quartiles` using `.describe()`
2. **Histogram** — to see the shape of the distribution
3. **KDE plot** — a smooth curve to confirm the shape
4. **Box plot** — to visualise spread and spot outliers
5. **Skewness score** — to quantify how asymmetrical the distribution is
6. **Missing value check** — `isnull().sum() / len()` to get the percentage of nulls

---

#### 🔹 Age

- Age is **approximately normally distributed** — the histogram and KDE both show a near-symmetrical bell curve centred around the late 20s/early 30s.
- **~20% of Age values are missing**, which is significant and will need imputation before modelling.
- **Outliers exist** — passengers older than 65 were identified and inspected. These are likely legitimate data points (elderly passengers), not errors.

---

#### 🔹 Fare

- Fare is **highly right-skewed** — most passengers paid low fares, but a small number paid extremely high amounts (300+), pulling the distribution to the right.
- A critical finding: **Fare is a group fare, not an individual fare**. Passengers travelling together shared a single ticket and the full combined fare was recorded for each of them. This is misleading for analysis. A new `individual` fare column was later engineered to correct this.

---

### Step 2 — Univariate Analysis (Categorical Columns)

For categorical columns, the process was:
1. **Value counts** — frequency of each category
2. **Bar chart** — to compare category sizes
3. **Pie chart** — to see proportional breakdown

---

#### 🔹 Survived
- **~62% did not survive**, ~38% survived — a class imbalance that matters for modelling.

#### 🔹 Pclass (Passenger Class)
- Three classes: 1st, 2nd, 3rd. The majority of passengers were in **3rd class**.

#### 🔹 Sex
- **Male passengers outnumber female** passengers significantly.

---

### Step 3 — Bivariate Analysis

Bivariate analysis looks at **pairs of columns** to explore relationships and, most importantly, how different variables relate to survival.

Three types of relationships were handled differently:

| Relationship Type | Technique Used |
|---|---|
| Categorical × Categorical | Cross-tabulation (normalised %) + heatmap |
| Numerical × Categorical | KDE plot per group + group mean comparison |
| Numerical × Numerical | Scatterplots, correlation coefficient |

---

#### 🔹 Survived × Pclass
A normalised cross-tab revealed a clear pattern: **survival rate dropped sharply in lower passenger classes**. 1st class passengers had the highest survival rate, 3rd class the lowest. Visualised as a heatmap.

#### 🔹 Survived × Sex
One of the strongest signals in the dataset. **Women had a dramatically higher survival rate than men** — consistent with the "women and children first" evacuation policy.

#### 🔹 Survived × Embarked
Passengers who embarked at **Cherbourg (C)** had a noticeably higher survival rate than those from Southampton (S) or Queenstown (Q) — likely because Cherbourg passengers skewed towards 1st class.

#### 🔹 Survived × Age
KDE plots were overlaid for survivors and non-survivors. **Younger passengers (especially children) had a higher survival tendency**, though age alone is not a dominant predictor.

---

### Step 4 — Feature Engineering

Based on observations from the EDA, several new columns were created to make the data more informative for modelling. The train and test sets were merged at this stage (`cv`) to engineer features across the full dataset.

---

#### 🔸 `individual` — Corrected Fare per Person
```
individual = Fare / (SibSp + Parch + 1)
```
Since `Fare` was recorded as a group fare, dividing by family size gives the actual fare each person paid. The `+1` accounts for the passenger themselves.

---

#### 🔸 `Family_size` — Total Family Members Onboard
```
Family_size = SibSp + Parch + 1
```
Combines siblings/spouses (`SibSp`) and parents/children (`Parch`) into a single family size metric. The `+1` includes the passenger.

---

#### 🔸 `Family_type` — Categorical Bucket for Family Size
```
Alone  → Family_size == 1
Small  → Family_size 2–4
Large  → Family_size 5+
```
A cross-tab of `Survived × Family_type` showed that **small family passengers had higher survival rates** than those travelling alone or in large groups — a meaningful signal.

---

#### 🔸 `Surname` — Extracted from Name
```
Surname = Name.split(',')[0]
```
Passenger names are formatted as `"Surname, Title. Firstname"`. Extracting the surname allows grouping passengers into family units.

---

#### 🔸 `Title` — Extracted from Name
```
Title = Name.split(',')[1].strip().split(' ')[0]
```
Titles like `Mr.`, `Mrs.`, `Miss.`, `Master.`, `Dr.`, `Rev.` were extracted. A cross-tab of `Survived × Title` showed that **title is a strong survival predictor** — `Mrs.` and `Miss.` had high survival rates, while `Mr.` had very low ones. Rare titles (Dr., Rev., Col.) also showed interesting patterns. Title effectively encodes gender, age group, and social class all at once.

---

## 📊 Key Findings Summary

| Variable | Finding |
|----------|---------|
| Age | ~Normal distribution, 20% missing, some outliers >65 |
| Fare | Right-skewed, represents group fare — individual fare column created |
| Sex | Strongest predictor — female survival rate far higher |
| Pclass | Clear gradient — 1st class survived most |
| Embarked | Cherbourg passengers survived at higher rates |
| Family_type | Small families survived more than lone or large-group travellers |
| Title | Strong signal — reflects gender, age, and social class simultaneously |

---

## 🛠️ Libraries Used

```python
pandas      # data loading, manipulation, cross-tabulations
numpy       # numerical operations
matplotlib  # base plotting (histograms, KDE, box plots, bar, pie)
seaborn     # heatmaps and styled plots
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/your-username/titanic-eda.git
cd titanic-eda
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook Titanic_Eda.ipynb
```

Place `train.csv` and `test.csv` in the same directory and update the `pd.read_csv()` paths in the notebook accordingly.

> Dataset available at: https://www.kaggle.com/competitions/titanic/data


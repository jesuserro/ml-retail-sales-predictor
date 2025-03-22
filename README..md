# 🧠 IronKaggle Competition: Predictive Sales Modeling Challenge

Welcome to the **IronKaggle** competition, where your machine learning skills and teamwork are put to the test! This is your opportunity to build a model that accurately predicts **store sales**, working collaboratively and under pressure. 🔥

---

## 📝 Introduction

You and your partner have been chosen to tackle a real-world dataset and create a predictive model that can estimate daily sales based on various features. Your Learning Team (LT) and Teaching Assistant (TA) are unavailable, so it's up to **you and your teammate** to make this happen.

---

## 👯 Pair Programming

This is a **team-based challenge**. Your success depends on collaboration, discussion, and clear decision-making. From EDA to modeling, everything must be done **together**. Choose your tools, techniques, and models carefully.

---

## 📊 Dataset Info

You'll be using the dataset found [here](https://raw.githubusercontent.com/data-bootcamp-v4/data/main/sales.csv).

### 🧾 Metadata

| Column              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `shop_ID`           | Unique identifier for each shop                                             |
| `day_of_the_week`   | Integer from 1 to 7 (Monday to Sunday)                                      |
| `date`              | Full date string (day/month/year)                                           |
| `number_of_customers` | Number of customers who visited the shop that day                          |
| `open`              | 0 = Closed, 1 = Open                                                         |
| `promotion`         | 0 = No promotion, 1 = Promotion active                                       |
| `state_holiday`     | 0 = None, 'a', 'b', or 'c' = Different types of state holidays               |
| `school_holiday`    | 0 = No school holiday, 1 = School holiday                                   |
| **`sales`**         | 💥 **Target variable**: Amount of sales on that day *(present only in training set)* |

---

## 🚀 Deliverables & Timeline

### 🔄 Phase 1: Model Development
- Use the dataset to **explore**, **clean**, and **build a regression model** that predicts `sales`.
- Collaborate with your teammate to choose the right features and ML algorithm (e.g., linear regression, random forest, XGBoost).
- Validate your model using train-test split or cross-validation.

### 🕓 16:00 — New Dataset Revealed
- A **test dataset without the `sales` column** will be shared.
- Use your trained model to predict sales for this new dataset.

### 🕔 17:00 — Submission Deadline
- Submit your final predictions.
- Your results will be compared to the actual `sales` values by your LT/TA (found [here](link-to-solution)).

---

## 🎯 Objective

Build a **high-performing sales prediction model** before the clock runs out. Use good ML practices, communicate effectively, and show your problem-solving skills.

Let’s get to coding, collaborating, and crushing it! 💪💻

Happy Predicting! 🌟

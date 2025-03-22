# 🧠 Final Sales Prediction Report

This project presents a complete workflow for training, evaluating, and reporting a machine learning model that predicts store sales. The output includes a detailed HTML and PDF report with visualizations and metrics, plus saved prediction artifacts.

---

## 📦 Contents

- `final_report.py` – Main script to generate report (HTML & PDF)  
- `3_sales_predictions.csv` – Predicted sales for test set  
- `4_sales_real_solutions.csv` – Actual sales (ground truth)  
- `sales_model.pkl` – Trained XGBoost model (optional for feature importance)  
- `docs/report.html` – Auto-generated interactive report  
- `docs/report.pdf` – Printable PDF version  
- `img/` – Folder with generated plots used in the report  

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ironkaggle-sales-predictor.git
cd ironkaggle-sales-predictor
```

### 2. (Optional but recommended) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the report generator

```bash
python final_report.py
```

### 5. Check outputs

- 📄 `docs/report.html` – interactive HTML report  
- 📑 `docs/report.pdf` – downloadable PDF version  
- 🖼️ `img/` – static plot files (can be reused in presentations)  

---

## 📊 Sample Plots

### Predicted vs Actual Sales  
![Scatter plot](img/scatter_real_vs_pred.png)

### Prediction Error Distribution  
![Error Histogram](img/histogram_error.png)

### Feature Importance  
![Feature Importance](img/feature_importance.png)

---

## 🧪 Metrics Reported

- R² Score  
- RMSE  
- MAE  
- Top 10 worst predictions by absolute error  

---

## 🧠 Tools & Libraries Used

- Python 3.8+  
- XGBoost  
- Scikit-learn  
- Pandas & Numpy  
- Matplotlib & Seaborn  
- WeasyPrint (for PDF export)  
- Jinja2 (HTML templating)  

---

## 👥 Authors

- You & your pair programming teammate 💻🤝  

---

## 🏁 Final Note

This script auto-installs missing packages at runtime, making it portable and easy to run on any machine with Python installed. Just run `final_report.py` and you're ready to go 🚀

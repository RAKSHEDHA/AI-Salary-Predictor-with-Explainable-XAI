# 💼 AI Salary Predictor with Explainable XAI

This project is an interactive web application that predicts employee salaries using a sophisticated Gradient Boosting machine learning model. What makes this tool unique is its use of **Explainable AI (XAI)** through the SHAP library, allowing users to understand *why* the model made a specific prediction.

The application is built with Streamlit and includes advanced features such as market comparison, percentile ranking, "what-if" analysis, and downloadable PDF reports.

## ✨ Key Features

* **Live Salary Prediction:** Predicts salaries in real-time based on user inputs for age, experience, job title, and education.
* **Explainable AI (XAI):** Uses a live SHAP force plot to visualize how each factor contributes to the final prediction.
* **Market Comparison:** Compares the predicted salary to the average salary for that specific job title.
* **Percentile Ranking:** Calculates and displays the percentile rank of the predicted salary, showing how it compares to others in the same role.
* **"What-If" Analysis:** A dynamic line chart that allows users to see how their salary prediction would change if a single factor (like age or experience) were different.
* **Downloadable PDF Reports:** Generates a professional, multi-page PDF report of the complete analysis with a single click.
* **Currency Conversion:** Displays all financial results in Indian Rupees (₹).

## 🛠️ Technology Stack & Libraries

* **Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-learn (GradientBoostingRegressor)
* **Model Explainability:** SHAP
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Statistical Analysis:** SciPy
* **PDF Generation:** fpdf2

## 🚀 How to Run This Project Locally

To run this application on your own machine, please follow these steps:

**1. Clone the Repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/AI-Salary-Predictor-with-Explainable-XAI.git](https://github.com/YOUR_USERNAME/AI-Salary-Predictor-with-Explainable-XAI.git)
```

**2. Navigate into the Project Directory:**
```bash
cd AI-Salary-Predictor-with-Explainable-XAI
```

**3. Install the Required Libraries:**
Make sure you have Python 3.8+ installed. Then, run the following command to install all necessary packages.
```bash
pip install -r requirements.txt
```

**4. Run the Streamlit Application:**
```bash
streamlit run app.py
```
The application will then open automatically in your web browser.

## 📁 Project File Structure

```
├── app.py                  # The main Streamlit application script
├── salary_model.pkl        # The pre-trained Gradient Boosting model
├── model_columns.pkl       # The columns the model was trained on
├── Salary Data.csv         # The original dataset used for training and analysis
└── requirements.txt        # A list of all required Python libraries

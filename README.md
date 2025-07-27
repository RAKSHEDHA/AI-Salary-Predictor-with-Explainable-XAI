# AI Salary Predictor with Explainable AI (XAI)

An interactive web application built with Streamlit that predicts salaries based on user-provided data. This project not only provides salary estimations but also leverages Explainable AI (XAI) using SHAP to offer transparent insights into how each factor influences the prediction.

-----

## 📋 Table of Contents

  - [Project Overview](https://www.google.com/search?q=%23-project-overview)
  - [Key Features](https://www.google.com/search?q=%23-key-features)
  - [Tech Stack](https://www.google.com/search?q=%23-tech-stack)
  - [Installation and Setup](https://www.google.com/search?q=%23-installation-and-setup)
  - [Usage](https://www.google.com/search?q=%23-usage)
  - [Project Structure](https://www.google.com/search?q=%23-project-structure)
  - [Model Details](https://www.google.com/search?q=%23-model-details)
  - [Future Improvements](https://www.google.com/search?q=%23-future-improvements)
  - [Contributing](https://www.google.com/search?q=%23-contributing)
  - [License](https://www.google.com/search?q=%23-license)

## 🌟 Project Overview

This project is designed to demystify salary expectations by providing data-driven insights. By inputting details such as age, years of experience, education level, and job title, users can receive an estimated salary. The key differentiator of this application is its focus on explainability. Using the SHAP (SHapley Additive exPlanations) library, the app visually breaks down the contribution of each input feature to the final prediction, helping users understand *why* the model made a certain decision, fostering trust and transparency.

## ✨ Key Features

  - **💵 Salary Prediction:** Uses a pre-trained Gradient Boosting model to estimate salaries in Indian Rupees (INR).
  - **📊 Explainable Predictions:** Integrates a SHAP Force plot to show the positive (blue) and negative (red) impacts of each feature on the predicted salary.
  - **🔬 What-If Analysis:** An interactive line chart allows users to see how their predicted salary changes as a single factor (like Age or Experience) varies, while all other inputs are held constant.
  - **📈 Peer Comparison:** Compares the user's predicted salary against the average for their job title and provides a percentile rank, offering valuable market context.
  - **📄 PDF Report Generation:** Users can download a detailed, professionally formatted PDF report containing their input details, prediction insights, and the SHAP explanation plot for offline use.

## 🛠️ Tech Stack

This project leverages a modern stack of open-source technologies for machine learning and web development.

  - **Core Language:** **[Python 3.9+](https://www.python.org/)**
  - **Machine Learning & Data Manipulation:** **[Scikit-learn](https://scikit-learn.org/)**, **[Pandas](https://pandas.pydata.org/)**
  - **Web Application Framework:** **[Streamlit](https://streamlit.io/)**
  - **Explainable AI (XAI):** **[SHAP](https://shap.readthedocs.io/en/latest/index.html)**
  - **Data Visualization:** **[Matplotlib](https://matplotlib.org/)**
  - **PDF Generation:** **[FPDF2](https://pyfpdf.github.io/fpdf2/)**
  - **Model & Data Persistence:** **[Joblib](https://joblib.readthedocs.io/en/latest/)**

## ⚙️ Installation and Setup

Follow these steps to set up and run the project in a local environment.

#### 1\. Clone the Repository

Open a terminal and run the following command to download the project files:

```bash
git clone https://github.com/RAKSHEDHA/AI-Salary-Predictor-with-Explainable-XAI.git
cd AI-Salary-Predictor-with-Explainable-XAI
```

#### 2\. Create and Activate a Virtual Environment

It is a best practice to use a virtual environment to manage project dependencies cleanly.

  - **For Windows:**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

  - **For macOS & Linux:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

#### 3\. Install Required Packages

All project dependencies are listed in the `requirements.txt` file and can be installed with a single command:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

Once the installation is complete, the Streamlit application can be launched with the following command:

```bash
streamlit run app.py
```

A new tab should automatically open in your default web browser with the application running. If not, the terminal will provide a local URL (e.g., `http://localhost:8501`) that can be opened manually.

## 📁 Project Structure

Here is an overview of the key files and directories within the project:

```
AI-Salary-Predictor-with-Explainable-XAI/
│
├── 📂 analysis/
│   └── 📜 Model_Training_and_Analysis.ipynb  # Jupyter notebook with the complete data science workflow.
│
├── 📜 app.py                                # The main Streamlit application script.
├── 📦 salary_model.pkl                      # The serialized, pre-trained Gradient Boosting model.
├── 📦 model_columns.pkl                     # The serialized list of model feature columns for one-hot encoding.
├── 📄 Salary Data.csv                        # The raw dataset used for training and evaluation.
├── 📄 requirements.txt                      # A list of all Python libraries required for the project.
└── 📜 README.md                             # This README file.
```

## 🧠 Model Details

  - **Model Type:** A `GradientBoostingRegressor` from the Scikit-learn library was chosen for its high performance on structured (tabular) data.
  - **Training Data:** The model was trained and evaluated on the **[`Salary Data.csv`](https://www.google.com/search?q=%5Bhttps://github.com/RAKSHEDHA/AI-Salary-Predictor-with-Explainable-XAI/blob/main/Salary%2520Data.csv%5D\(https://github.com/RAKSHEDHA/AI-Salary-Predictor-with-Explainable-XAI/blob/main/Salary%2520Data.csv\))** dataset.
  - **Features Used:** The model's predictions are based on the following input features:
      - Age
      - Years of Experience
      - Gender
      - Education Level (One-Hot Encoded)
      - Job Title (One-Hot Encoded)
  - Running the Analysis Notebook
The entire data science process, including data cleaning, feature engineering, model training, performance evaluation (R² score, MAE, etc.), and SHAP analysis, is documented in the analysis/Model_Training_and_Analysis.ipynb notebook.

You can run this notebook in two ways:

1. Using Google Colab (Recommended for ease of use)

Click the "Open in Colab" badge above to launch the notebook directly in Google Colab.

In the Colab environment, use the file browser on the left to upload the Salary Data.csv file from this repository.

You can now run each cell of the notebook sequentially to reproduce the analysis.

2. Locally with Jupyter Notebook

If you have Jupyter Notebook or JupyterLab installed locally (included with packages like Anaconda), you can navigate to the analysis directory and open the .ipynb file to run it.



## 🔮 Future Improvements

This project has a solid foundation with several avenues for future enhancement:

  - [ ] **Public Deployment:** Deploy the application to a cloud service like Streamlit Community Cloud for public access.
  - [ ] **Dataset Expansion:** Incorporate a larger, more comprehensive dataset with additional features (e.g., location, company size, industry) to improve model accuracy and generalizability.
  - [ ] **Model Experimentation:** Experiment with other advanced regression models, such as **[XGBoost](https://xgboost.ai/)** or **[LightGBM](https://lightgbm.readthedocs.io/en/latest/)**, to compare performance.
  - [ ] **CI/CD Pipeline:** Implement a Continuous Integration/Continuous Deployment pipeline using a tool like GitHub Actions for automated testing and deployment.

## 🙌 Contributing

Contributions are welcome and greatly appreciated. This project thrives on open-source collaboration. If you have suggestions for improvements or want to fix a bug, please follow these steps:

1.  Fork the Project.
2.  Create your Feature Branch (`git checkout -b feature/NewFeature`).
3.  Commit your Changes (`git commit -m 'feat: Add some NewFeature'`).
4.  Push to the Branch (`git push origin feature/NewFeature`).
5.  Open a Pull Request.

## ⚖️ License

This project is distributed under the MIT License. See the `LICENSE` file for more information.

📧 Get in Touch With Me
I'd love to hear your feedback or connect with you!

RAKSHEDHA - www.linkedin.com/in/rakshedha

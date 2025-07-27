import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from fpdf import FPDF
from datetime import datetime

# --- Define the USD to INR conversion rate ---
USD_TO_INR_RATE = 83.0

# --- PDF Report Generation Function ---
def create_pdf_report(details, metrics, shap_plot_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)

    # Title
    pdf.cell(0, 10, "Salary Analysis Report", 1, 1, "C")
    pdf.ln(10)

    # Input Details Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Input Employee Details", 0, 1)
    pdf.set_font("Arial", "", 11)
    for key, value in details.items():
        pdf.cell(0, 8, f"  - {key}: {value}", 0, 1)
    pdf.ln(10)

    # Prediction Metrics Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Prediction Insights (INR)", 0, 1)
    pdf.set_font("Arial", "", 11)
    for key, value in metrics.items():
        pdf.cell(0, 8, f"  - {key}: {value}", 0, 1)
    pdf.ln(10)
    
    # SHAP Plot Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Prediction Explanation (SHAP Plot)", 0, 1)
    pdf.image(shap_plot_path, x=None, y=None, w=180) # w=180 to fit the page width
    pdf.ln(5)

    # Footer
    pdf.set_y(-15)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, "C")

    # THE FIX IS HERE: Convert the bytearray to bytes
    return bytes(pdf.output(dest='S'))


# --- Load Assets ---
@st.cache_data
def load_assets():
    model = joblib.load('salary_model.pkl')
    model_cols = joblib.load('model_columns.pkl')
    explainer = shap.TreeExplainer(model)
    df = pd.read_csv('Salary Data.csv')
    df.dropna(inplace=True)
    job_titles = sorted(df['Job Title'].unique())
    education_levels = sorted(df['Education Level'].unique())
    role_stats = df.groupby('Job Title')['Salary']
    return model, model_cols, explainer, job_titles, education_levels, role_stats

model, model_cols, explainer, job_titles, education_levels, role_stats = load_assets()

# --- App Layout and Title ---
st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="wide")
st.title("💼 Explainable Salary Predictor (INR)")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("👤 Employee Details")
    age = st.slider("Age", 18, 65, 30)
    years_exp = st.slider("Years of Experience", 0, 40, 5)
    education = st.selectbox("Education Level", education_levels)
    job_title = st.selectbox("Job Title", job_titles)
    predict_button = st.button("Predict Salary 💵", use_container_width=True, type="primary")

# --- Main Page Content ---
if predict_button:
    # --- Prediction Logic ---
    query_df = pd.DataFrame({'Age': [age], 'Years of Experience': [years_exp], 'Education Level': [education], 'Gender': ['Male'], 'Job Title': [job_title]})
    query_encoded = pd.get_dummies(query_df).reindex(columns=model_cols, fill_value=0)
    prediction_usd = model.predict(query_encoded)[0]
    prediction_inr = prediction_usd * USD_TO_INR_RATE
    
    # --- Display Prediction Results ---
    st.subheader("📊 Salary Insights (in INR)")
    with st.container(border=True):
        st.markdown(f"<h3 style='text-align: center;'>Predicted Salary</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: green;'>₹{prediction_inr:,.2f}</h2>", unsafe_allow_html=True)

    # --- Market Comparison Logic & Display ---
    st.subheader("📈 Comparison to Market Data (in INR)")
    role_salary_data_inr = role_stats.get_group(job_title) * USD_TO_INR_RATE
    role_average_inr = role_salary_data_inr.mean()
    percentile = stats.percentileofscore(role_salary_data_inr, prediction_inr)
    delta_inr = prediction_inr - role_average_inr
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Role Average", f"₹{role_average_inr:,.2f}")
        col2.metric("Difference", f"₹{delta_inr:,.2f}", f"{delta_inr/role_average_inr:.1%}")
        col3.metric("Percentile Rank", f"{percentile:.0f}th", help="This salary is higher than this percentage of people in the same role.")

    # --- SHAP Explanation Display ---
    with st.container(border=True):
        st.subheader("🧠 Live Prediction Explanation")
        shap_values = explainer.shap_values(query_encoded)
        shap_plot = shap.force_plot(explainer.expected_value, shap_values[0,:], query_encoded.iloc[0,:], matplotlib=True, show=False)
        st.pyplot(shap_plot, bbox_inches='tight')
        # Save the plot to a file to be used in the PDF
        plt.savefig("shap_plot.png", bbox_inches='tight')
        plt.close()

    # --- What-If Analysis Display ---
    with st.container(border=True):
        st.subheader("🔬 What-If Analysis")
        feature_to_analyze = st.selectbox("Select a factor to analyze:", ['Age', 'Years of Experience'])
        value_range = np.arange(18, 66) if feature_to_analyze == 'Age' else np.arange(0, 41)
        what_if_df = pd.concat([query_encoded]*len(value_range), ignore_index=True)
        what_if_df[feature_to_analyze] = value_range
        what_if_predictions_inr = model.predict(what_if_df) * USD_TO_INR_RATE
        chart_data = pd.DataFrame({'Predicted Salary (₹)': what_if_predictions_inr}, index=value_range)
        st.line_chart(chart_data)
        
    # --- PDF Download Button ---
    st.sidebar.header("📄 Download Report")
    input_details = {'Age': age, 'Years of Experience': years_exp, 'Education Level': education, 'Job Title': job_title}
    prediction_metrics = {
        'Predicted Salary': f"Rs. {prediction_inr:,.2f}",
        'Role Average': f"Rs. {role_average_inr:,.2f}",
        'Percentile Rank': f"{percentile:.0f}th"
    }
    pdf_data = create_pdf_report(input_details, prediction_metrics, "shap_plot.png")
    st.sidebar.download_button(
        label="Download PDF Report",
        data=pdf_data,
        file_name=f"Salary_Report_{job_title.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )

else:
    st.info("Enter employee details in the sidebar and click 'Predict Salary' to see the results.")
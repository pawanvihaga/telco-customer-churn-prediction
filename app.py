import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
def login():
    st.title("🔐 Login to Churn Prediction Dashboard")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Check if both username and password are provided (not empty)
        if username.strip() and password.strip():
            st.session_state['logged_in'] = True
            st.session_state['current_user'] = username  # Save current logged-in user name
            st.success(f"Successfully logged in, {username}!")
            st.rerun()
        else:
            st.error("Please enter both Username and Password!")
            # Check if user is logged in
if not st.session_state['logged_in']:
    login()
    st.stop()  # Hides the dashboard below until logged in

# Login වී නැත්නම් Login Form එක පෙන්වීම
if not st.session_state['logged_in']:
    login()
    st.stop()  # Login වන තෙක් පහළ තියෙන Dashboard එක Hide කරයි

# ----------------- MAIN DASHBOARD (LOGGED IN) -----------------
st.sidebar.write("👤 Logged in as: **Admin**")
if st.sidebar.button("Logout"):
    st.session_state['logged_in'] = False
    st.rerun()
# Load saved model, scaler, and encoders
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl')
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # පරිශීලකයා ගොනුවක් Upload කළ විට එය භාවිත කිරීම
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset successfully uploaded!")

else:
    # ගොනුවක් ලබා දී නැත්නම්, අනතුරු අඟවා ඇප් එක නවතා තැබීම
    st.warning("⚠️ Please upload a CSV dataset file from the sidebar to proceed.")
    st.stop()

st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")

st.title("📊 Telco Customer Churn Prediction & Analytics Dashboard")
st.write("Predict whether a customer is likely to churn based on their account details and service usage.")
# ප්‍රධාන දත්ත පෙන්වන Metric Cards (KPIs)
st.markdown("### 🚀 Quick Performance Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Total Customers", value="7,043", delta="+123 this month")
with col2:
    st.metric(label="Estimated Churn Rate", value="26.5%", delta="-1.2%", delta_color="inverse")
with col3:
    st.metric(label="Model Accuracy", value="85.4%", delta="+2.1%")

st.markdown("---")

# Sidebar for user inputs
st.sidebar.header("Customer Input Features")

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen', (0, 1))
    Partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
    Dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
    tenure = st.sidebar.slider('Tenure (Months)', 0, 72, 1)
    PhoneService = st.sidebar.selectbox('Phone Service', ('Yes', 'No'))
    MultipleLines = st.sidebar.selectbox('Multiple Lines', ('Yes', 'No', 'No phone service'))
    InternetService = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    OnlineSecurity = st.sidebar.selectbox('Online Security', ('Yes', 'No', 'No internet service'))
    OnlineBackup = st.sidebar.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
    DeviceProtection = st.sidebar.selectbox('Device Protection', ('Yes', 'No', 'No internet service'))
    TechSupport = st.sidebar.selectbox('Tech Support', ('Yes', 'No', 'No internet service'))
    StreamingTV = st.sidebar.selectbox('Streaming TV', ('Yes', 'No', 'No internet service'))
    StreamingMovies = st.sidebar.selectbox('Streaming Movies', ('Yes', 'No', 'No internet service'))
    Contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
    PaymentMethod = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    MonthlyCharges = st.sidebar.number_input('Monthly Charges ($)', 0.0, 150.0, 50.0)
    TotalCharges = st.sidebar.number_input('Total Charges ($)', 0.0, 10000.0, 500.0)

    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

# Display user input
st.subheader("Customer Details Entered:")
st.write(df_input)

# Preprocessing user input using saved encoders and scaler
input_df_encoded = df_input.copy()
for col in input_df_encoded.select_dtypes(include=['object']).columns:
    if col in encoders:
        le = encoders[col]
        # Handle unseen labels safely
        input_df_encoded[col] = input_df_encoded[col].apply(lambda s: s if s in le.classes_ else le.classes_[0])
        input_df_encoded[col] = le.transform(input_df_encoded[col])

# Scale input
input_scaled = scaler.transform(input_df_encoded)

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ **This customer is likely to CHURN!** (Probability: {prediction_proba[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ **This customer is likely to STAY (No Churn).** (Probability: {prediction_proba[0][0]*100:.2f}%)")

        import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- DATA VISUALIZATION SECTION -----------------
st.markdown("---")
st.subheader("📈 Customer Churn Analytics & Visualizations")

# Dataset එක Load කරගැනීම (Graphs ඇඳීමට)
@st.cache_data
def load_data():
    return pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df_viz = load_data()

# 1. Churn Count Bar Chart (Advanced Seaborn Plot)
st.write("### Total Churn vs Non-Churn Customers")

fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor('#0E1117')
ax.set_facecolor('#0E1117')

# 'df_viz' DataFrame එක පාවිච්චි කරමින් Count Plot එක ඇඳීම (කොළ සහ රතු වර්ණවලින්)
sns.countplot(data=df_viz, x='Churn', palette=['#2ECC71', '#E74C3C'], ax=ax)

ax.set_xlabel("Churn Status", color='white', fontsize=11)
ax.set_ylabel("Customer Count", color='white', fontsize=11)

# අංක සහ මායිම් (Axis & Borders) සුදු පැහැයට හැරවීම
ax.tick_params(colors='white', which='both')
for spine in ax.spines.values():
    spine.set_edgecolor('#555555')

st.pyplot(fig)

# 2. Monthly Charges vs Churn (Advanced Seaborn Histogram)
st.write("#### Monthly Charges Distribution by Churn Status")

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0E1117')
ax.set_facecolor('#0E1117')

sns.histplot(
    data=df_viz, 
    x='MonthlyCharges', 
    hue='Churn', 
    element="bars", 
    stat="count", 
    kde=True, 
    palette={'No': '#1E88E5', 'Yes': '#D81B60'}, 
    alpha=0.5, 
    ax=ax
)

ax.set_title('Distribution of Monthly Charges by Churn Status', color='white', fontsize=16, pad=20)
ax.set_xlabel('Monthly Charges ($)', color='white', fontsize=12)
ax.set_ylabel('Count of Customers', color='white', fontsize=12)

legend = ax.legend(title='Customer Churn', loc='upper right', bbox_to_anchor=(1, 1.15), frameon=False)
plt.setp(legend.get_title(), color='white', fontsize=12)
for text in legend.get_texts():
    text.set_color('white')
    text.set_fontsize(11)

ax.tick_params(axis='x', colors='white', labelsize=11)
ax.tick_params(axis='y', colors='white', labelsize=11)

for spine in ax.spines.values():
    spine.set_edgecolor('#555555')

st.pyplot(fig)
# ----------------- BULK PREDICTION (CSV FILE UPLOAD) -----------------
st.markdown("---")
st.subheader("📁 Bulk Customer Churn Prediction (CSV Upload)")
st.write("Upload a CSV file containing details of multiple customers to obtain batch predictions at once.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Upload කළ CSV file එක Read කිරීම
    bulk_df = pd.read_csv(uploaded_file)
    st.write("### Upload කළ Data වල Sample එකක්:")
    st.dataframe(bulk_df.head())
    
    if st.button("Predict for All Customers"):
        try:
            # 1. Data Preprocessing (Model එකට අවශ්‍ය විදිහට Encode කිරීම)
            processed_df = bulk_df.copy()
            
            # Encoder අදාළ Column වලට Apply කිරීම
            for col, le in encoders.items():
                if col in processed_df.columns:
                    processed_df[col] = le.transform(processed_df[col].astype(str))
            
            # Feature Scaling (Scaler එක ඇත්නම්)
            scaled_data = scaler.transform(processed_df)
            
            # 2. Bulk Predictions ලබාගැනීම
            predictions = model.predict(scaled_data)
            probabilities = model.predict_proba(scaled_data)[:, 1]
            
            # Results මුල් DataFrame එකට එකතු කිරීම
            bulk_df['Churn_Prediction'] = ['Yes (Likely to Churn)' if p == 1 else 'No (Likely to Stay)' for p in predictions]
            bulk_df['Churn_Probability (%)'] = (probabilities * 100).round(2)
            
            st.success("✅ සියලුම Customers ලා සඳහා Predictions සෑදීම සාර්ථකයි!")
            
            # Results Table එක පෙන්වීම
            st.write("### 📊 Prediction Results:")
            st.dataframe(bulk_df)
            
            # 3. Processed Results CSV එකක් ලෙස Download කිරීමට Button එකක් ලබාදීම
            csv_result = bulk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_result,
                file_name="Customer_Churn_Predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error during bulk prediction: {e}")
            st.info("💡 Upload කළ CSV එකෙහි Column නම් Model එක Train කිරීමට පාවිච්චි කළ Dataset එකට සමාන දැයි තහවුරු කරගන්න.")
            from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# PDF එක generate කිරීමට Function එකක්
def generate_pdf_report(customer_data, prediction_text, probability):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # PDF Title & Header
    p.setFont("Helvetica-Bold", 20)
    p.drawString(50, height - 50, "Telco Customer Churn Prediction Report")
    
    p.setFont("Helvetica", 10)
    p.drawString(50, height - 70, "Generated by Churn Analytics Dashboard")
    p.line(50, height - 80, width - 50, height - 80)
    
    # Prediction Result Section
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, height - 120, "Prediction Result:")
    
    p.setFont("Helvetica", 12)
    p.drawString(70, height - 145, f"Status: {prediction_text}")
    p.drawString(70, height - 165, f"Churn Probability: {probability:.2f}%")
    
    # Customer Details Section
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, height - 210, "Customer Details:")
    
    y_offset = height - 235
    p.setFont("Helvetica", 11)
    for key, value in customer_data.items():
        p.drawString(70, y_offset, f"- {key}: {value}")
        y_offset -= 20
        if y_offset < 50:
            p.showPage()
            y_offset = height - 50
            
    # Footer
    p.setFont("Helvetica-Oblique", 9)
    p.drawString(50, 30, "Confidential - Telco Customer Retention Analytics")
    
    p.showPage()
    p.save()
    
    buffer.seek(0)
    return buffer.getvalue()

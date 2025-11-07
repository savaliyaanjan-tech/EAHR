import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import io

st.set_page_config(page_title="Employee Attrition Analytics Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("EA.csv")
    return df

df = load_data()

st.title("👥 Employee Attrition Analytics Dashboard")

# --- Filters
roles = st.multiselect("Select Job Roles", df["JobRole"].unique(), default=df["JobRole"].unique())
satisfaction = st.slider("Select Minimum Job Satisfaction", int(df["JobSatisfaction"].min()), int(df["JobSatisfaction"].max()), int(df["JobSatisfaction"].min()))
filtered = df[(df["JobRole"].isin(roles)) & (df["JobSatisfaction"] >= satisfaction)]

# --- KPIs
total_emp = len(filtered)
attrition_count = filtered["Attrition"].value_counts().get("Yes", 0)
attrition_rate = (attrition_count / total_emp) * 100 if total_emp else 0
avg_income = round(filtered["MonthlyIncome"].mean(), 2)
avg_satisfaction = round(filtered["JobSatisfaction"].mean(), 2)
avg_balance = round(filtered["WorkLifeBalance"].mean(), 2)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Employees", total_emp)
col2.metric("Attrition (%)", f"{attrition_rate:.2f}")
col3.metric("Avg Monthly Income", avg_income)
col4.metric("Avg Job Satisfaction", avg_satisfaction)
col5.metric("Avg Work-Life Balance", avg_balance)

# --- Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 ML Models", "📁 Upload & Predict"])

# --- Dashboard Charts
with tab1:
    st.subheader("HR Insights & Analytics")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.bar(filtered, x="JobRole", color="Attrition", barmode="group", title="Attrition by Job Role")
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.box(filtered, x="Attrition", y="JobSatisfaction", color="Attrition", title="Job Satisfaction vs Attrition")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = px.scatter(filtered, x="YearsAtCompany", y="MonthlyIncome", color="Attrition", title="Monthly Income vs Years at Company")
        st.plotly_chart(fig3, use_container_width=True)
        fig4 = px.density_heatmap(filtered, x="Department", y="WorkLifeBalance", color_continuous_scale="Blues", title="Work-Life Balance by Department")
        st.plotly_chart(fig4, use_container_width=True)
    fig5 = px.line(filtered, x="YearsSinceLastPromotion", y="YearsWithCurrManager", color="Attrition", title="Promotion vs Current Manager Years")
    st.plotly_chart(fig5, use_container_width=True)

# --- ML Models
with tab2:
    st.subheader("Machine Learning Model Comparison")

    @st.cache_data
    def preprocess(df):
        df_encoded = df.copy()
        le = LabelEncoder()
        for col in df_encoded.select_dtypes("object"):
            df_encoded[col] = le.fit_transform(df_encoded[col])
        X = df_encoded.drop("Attrition", axis=1)
        y = df_encoded["Attrition"]
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_train, X_test, y_train, y_test = preprocess(df)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosted Tree": GradientBoostingClassifier(random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred), 3),
            "Recall": round(recall_score(y_test, y_pred), 3),
            "F1 Score": round(f1_score(y_test, y_pred), 3)
        })

    st.dataframe(pd.DataFrame(results))

# --- Upload & Predict
with tab3:
    st.subheader("Upload New Dataset for Attrition Prediction")
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded:
        new_df = pd.read_csv(uploaded)
        st.write("Uploaded Data Preview:", new_df.head())

        # encode columns
        le = LabelEncoder()
        for col in new_df.select_dtypes("object"):
            new_df[col] = le.fit_transform(new_df[col])
        rf = RandomForestClassifier(random_state=42)
        X = df.drop("Attrition", axis=1)
        y = LabelEncoder().fit_transform(df["Attrition"])
        rf.fit(X, y)
        preds = rf.predict(new_df)
        new_df["Predicted_Attrition"] = preds

        buffer = io.BytesIO()
        new_df.to_csv(buffer, index=False)
        st.download_button("Download Predictions", data=buffer.getvalue(), file_name="Predicted_Attrition.csv", mime="text/csv")

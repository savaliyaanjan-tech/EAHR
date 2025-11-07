
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

st.title("👥 Employee Attrition Prediction & HR Insights")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # Preprocess Data
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # Label Encoding
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Sidebar Filters
    st.sidebar.header("🔍 Filters")
    job_roles = df['JobRole'].unique().tolist() if 'JobRole' in df.columns else []
    selected_roles = st.sidebar.multiselect("Select Job Roles", job_roles, default=job_roles)
    satisfaction_cols = [col for col in df.columns if "Satisfaction" in col]
    selected_satisfaction = st.sidebar.slider("Filter by Satisfaction", 0, 5, (0, 5))

    filtered_df = df.copy()
    if 'JobRole' in df.columns:
        filtered_df = filtered_df[filtered_df['JobRole'].isin(selected_roles)]
    for col in satisfaction_cols:
        filtered_df = filtered_df[(filtered_df[col] >= selected_satisfaction[0]) & (filtered_df[col] <= selected_satisfaction[1])]

    # Charts Section
    st.header("📊 HR Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition by Job Role")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='JobRole', hue='Attrition', data=filtered_df, ax=ax1, palette='Blues')
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        st.subheader("Average Monthly Income by Department")
        fig2, ax2 = plt.subplots()
        sns.barplot(x='Department', y='MonthlyIncome', data=filtered_df, ax=ax2, palette='coolwarm')
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        st.subheader("Attrition Rate by Age Group")
        filtered_df['AgeGroup'] = pd.cut(filtered_df['Age'], bins=[18,25,35,45,55,65], labels=['18-25','26-35','36-45','46-55','56-65'])
        fig3, ax3 = plt.subplots()
        sns.countplot(x='AgeGroup', hue='Attrition', data=filtered_df, ax=ax3, palette='mako')
        st.pyplot(fig3)

    with col2:
        st.subheader("Attrition by Gender and Overtime")
        fig4, ax4 = plt.subplots()
        sns.countplot(x='Gender', hue='OverTime', data=filtered_df, ax=ax4, palette='Set2')
        st.pyplot(fig4)

        st.subheader("Years at Company vs Attrition")
        fig5, ax5 = plt.subplots()
        sns.boxplot(x='Attrition', y='YearsAtCompany', data=filtered_df, ax=ax5, palette='Spectral')
        st.pyplot(fig5)

    # Algorithm Section
    st.header("🧠 Model Training and Performance")
    if st.button("Run Models"):
        X = df.drop('Attrition', axis=1)
        y = df['Attrition']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_pred)

            results.append({
                "Model": name,
                "Train Accuracy": model.score(X_train, y_train),
                "Test Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_proba)
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

    # Prediction Tab
    st.header("📂 Upload New Data for Prediction")
    new_file = st.file_uploader("Upload new data for prediction", type=["csv"], key="newdata")
    if new_file is not None:
        new_df = pd.read_csv(new_file)
        for col in new_df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            new_df[col] = le.fit_transform(new_df[col])
        model = GradientBoostingClassifier(random_state=42)
        X = df.drop('Attrition', axis=1)
        y = df['Attrition']
        model.fit(X, y)
        new_df['Predicted_Attrition'] = model.predict(new_df)
        st.write(new_df.head())

        csv = new_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predicted Data", csv, "Predicted_Attrition.csv", "text/csv")
else:
    st.info("👆 Upload a dataset to begin analysis.")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('EA.csv')

data = load_data()

st.sidebar.title('Navigation')
tabs = st.sidebar.radio('Go to:', ['📊 Insights Dashboard', '🤖 Model Performance', '📂 Predict New Data'])

# ------------------------ Tab 1: Insights Dashboard ------------------------ #
if tabs == '📊 Insights Dashboard':
    st.title('Employee Attrition Insights Dashboard')
    job_roles = st.multiselect('Select Job Role(s):', data['JobRole'].unique(), default=data['JobRole'].unique())
    sat_min, sat_max = st.slider('Filter by Job Satisfaction:', int(data['JobSatisfaction'].min()), int(data['JobSatisfaction'].max()), (int(data['JobSatisfaction'].min()), int(data['JobSatisfaction'].max())))
    filtered = data[(data['JobRole'].isin(job_roles)) & (data['JobSatisfaction'].between(sat_min, sat_max))]

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(filtered.groupby('JobRole')['Attrition'].value_counts(normalize=True).rename('Rate').reset_index(),
                      x='JobRole', y='Rate', color='Attrition', title='Attrition Rate by Job Role')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.pie(filtered, names='JobRole', title='Employee Distribution by Job Role')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.scatter(filtered, x='Age', y='MonthlyIncome', color='Attrition', title='Age vs Income by Attrition')
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = px.box(filtered, x='JobRole', y='MonthlyIncome', color='Attrition', title='Income Distribution by Role')
        st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.density_heatmap(filtered, x='Age', y='JobSatisfaction', color_continuous_scale='Blues', title='Satisfaction vs Age Heatmap')
    st.plotly_chart(fig5, use_container_width=True)

# ------------------------ Tab 2: Model Performance ------------------------ #
elif tabs == '🤖 Model Performance':
    st.title('Model Performance Metrics')

    def preprocess(df):
        df = df.copy()
        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = le.fit_transform(df[col])
        X = df.drop('Attrition', axis=1)
        y = df['Attrition']
        return X, y

    X, y = preprocess(data)

    if len(y.unique()) < 2:
        st.error("Attrition column must have at least two unique classes for model training.")
        st.stop()

    if y.value_counts().min() < 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {'Decision Tree': DecisionTreeClassifier(),
              'Random Forest': RandomForestClassifier(),
              'Gradient Boosting': GradientBoostingClassifier()}

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores = cross_val_score(model, X, y, cv=5)
        results.append({
            'Model': name,
            'Train Acc': model.score(X_train, y_train),
            'Test Acc': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'AUC': roc_auc_score(y_test, y_pred),
            'CV Mean': scores.mean()
        })

    st.dataframe(pd.DataFrame(results))

    for name, model in models.items():
        y_pred = model.fit(X_train, y_train).predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        st.subheader(f'{name} Confusion Matrix')
        st.write(cm)

# ------------------------ Tab 3: Predict New Data ------------------------ #
elif tabs == '📂 Predict New Data':
    st.title('Predict Attrition for New Data')
    upload = st.file_uploader('Upload New Dataset (CSV)', type=['csv'])
    if upload:
        new_data = pd.read_csv(upload)
        le = LabelEncoder()
        for col in new_data.select_dtypes(include='object').columns:
            new_data[col] = le.fit_transform(new_data[col])
        X, y = preprocess(data)
        model = RandomForestClassifier().fit(X, y)
        preds = model.predict(new_data)
        new_data['Predicted_Attrition'] = preds
        st.dataframe(new_data.head())
        csv = new_data.to_csv(index=False).encode('utf-8')
        st.download_button('Download Predicted File', data=csv, file_name='predicted_attrition.csv', mime='text/csv')

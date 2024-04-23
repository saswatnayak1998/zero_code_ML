import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, classification_report
import altair as alt

def plot_parity(y_true, y_pred):
    df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    chart = alt.Chart(df).mark_point().encode(
        x='Actual',
        y='Predicted',
        tooltip=['Actual', 'Predicted']
    ).properties(
        title='Parity Plot'
    ) + alt.Chart(df).mark_line(color='red').encode(
        x='Actual',
        y='Actual'
    )
    chart = chart.properties(width=600, height=600).interactive()
    st.altair_chart(chart, use_container_width=True)

def plot_data(X, y):
    df = pd.DataFrame({
        'Feature': X.squeeze(),
        'Target': y
    })
    chart = alt.Chart(df).mark_point(color='blue').encode(
        x='Feature',
        y='Target',
        tooltip=['Feature', 'Target']
    ).properties(
        title='Scatter Plot of the Example Data',
        width=600,
        height=400
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

st.title('Zero Code Machine Learning- Saswat K Nayak')
task_type = st.radio('Select Task Type', ('Regression', 'Classification'))
uploaded_file = st.file_uploader("Choose a CSV file. Make sure the last column is the Y value and the rest are Xs. All columns should be numerical.", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if data is not None:
        st.write(data.head())

        # Extracting features and target from the data
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Check if the data is 2D and plot if true
        if X.shape[1] == 1:
            plot_data(X, y)

        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model and kernel selection
        if task_type == 'Regression':
            model_option = st.selectbox(
                'Select Model Type',
                ('Ridge Regression', 'Random Forest Regression', 'SVM Regression')
            )
            kernel_option = 'linear'
            if 'SVM' in model_option:
                kernel_option = st.selectbox('Select Kernel Type', ['linear', 'poly', 'rbf', 'sigmoid'])
            if 'Ridge' in model_option:
                kernel_option = st.selectbox('Select Kernel Type', ['linear', 'polynomial', 'rbf', 'sigmoid'])
                # Hyperparameters for polynomial, rbf, and sigmoid kernels
                if kernel_option in ['polynomial', 'sigmoid']:
                    coef0 = st.slider('Coefficient 0', 0.0, 10.0, 1.0)
                    if kernel_option == 'polynomial':
                        degree = st.slider('Degree of the polynomial', 2, 5)

                if kernel_option in ['rbf', 'sigmoid']:
                    gamma = st.slider('Gamma', 0.01, 1.0, 0.1)
        else:
            model_option = st.selectbox(
                'Select Model Type',
                ('Logistic Regression', 'Random Forest Classification', 'SVM Classification')
            )
            if 'SVM' in model_option:
                kernel_option = st.selectbox('Select Kernel Type', ['linear', 'poly', 'rbf', 'sigmoid'])

        if st.button('Train Model'):
            model_kwargs = {'kernel': kernel_option} if kernel_option else {}
            if kernel_option in ['polynomial', 'sigmoid']:
                model_kwargs['coef0'] = coef0
                if kernel_option == 'polynomial':
                    model_kwargs['degree'] = degree
            if kernel_option in ['rbf', 'sigmoid']:
                model_kwargs['gamma'] = gamma

            model = None
            if task_type == 'Regression':
                if model_option == 'Ridge Regression':
                    model = KernelRidge(alpha=1.0, **model_kwargs)
                elif model_option == 'Random Forest Regression':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_option == 'SVM Regression':
                    model = SVR(**model_kwargs)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                st.write(f'Mean Squared Error: {mse}')
                plot_parity(y_test, y_pred)

            else:
                if model_option == 'Logistic Regression':
                    model = LogisticRegression(max_iter=1000)
                elif model_option == 'Random Forest Classification':
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_option == 'SVM Classification':
                    model = SVC(**model_kwargs)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                st.write(f'Accuracy: {accuracy}')
                st.write(f'F1 Score: {f1}')
                st.text('Classification Report:')
                st.text(classification_report(y_test, y_pred))

else:
    uri = "https://raw.githubusercontent.com/saswatnayak1998/zero_code_ML/main/Kernel_Ridge_Regression_Data.csv"
    data = pd.read_csv(uri)
    if data is not None:
        st.write(data.head())

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        if X.shape[1] == 1:
            plot_data(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if task_type == 'Regression':
            model_option = st.selectbox(
                'Select Model Type',
                ('Ridge Regression', 'Random Forest Regression', 'SVM Regression')
            )
            kernel_option = 'linear'
            if 'SVM' in model_option or 'Ridge' in model_option:
                kernel_option = st.selectbox('Select Kernel Type', ['linear', 'poly', 'rbf', 'sigmoid'])

                # Hyperparameters for polynomial, rbf, and sigmoid kernels
                if kernel_option in ['polynomial', 'sigmoid']:
                    coef0 = st.slider('Coefficient 0', 0.0, 10.0, 1.0)
                    if kernel_option == 'polynomial':
                        degree = st.slider('Degree of the polynomial', 2, 5)

                if kernel_option in ['rbf', 'sigmoid']:
                    gamma = st.slider('Gamma', 0.01, 1.0, 0.1)
        else:
            model_option = st.selectbox(
                'Select Model Type',
                ('Logistic Regression', 'Random Forest Classification', 'SVM Classification')
            )
            if 'SVM' in model_option:
                kernel_option = st.selectbox('Select Kernel Type', ['linear', 'poly', 'rbf', 'sigmoid'])

        # Create and train model
        if st.button('Train Model'):
            model_kwargs = {'kernel': kernel_option} if kernel_option else {}
            if kernel_option in ['polynomial', 'sigmoid']:
                model_kwargs['coef0'] = coef0
                if kernel_option == 'polynomial':
                    model_kwargs['degree'] = degree
            if kernel_option in ['rbf', 'sigmoid']:
                model_kwargs['gamma'] = gamma

            model = None
            if task_type == 'Regression':
                if model_option == 'Ridge Regression':
                    model = KernelRidge(alpha=1.0, **model_kwargs)
                elif model_option == 'Random Forest Regression':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_option == 'SVM Regression':
                    model = SVR(**model_kwargs)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                st.write(f'Mean Squared Error: {mse}')
                plot_parity(y_test, y_pred)

            else:
                if model_option == 'Logistic Regression':
                    model = LogisticRegression(max_iter=1000)
                elif model_option == 'Random Forest Classification':
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_option == 'SVM Classification':
                    model = SVC(**model_kwargs)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                st.write(f'Accuracy: {accuracy}')
                st.write(f'F1 Score: {f1}')
                st.text('Classification Report:')
                st.text(classification_report(y_test, y_pred))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import inspect
import os

# Function to load data
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Function to calculate remaining percentage
def calculate_remaining_percentage(age):
    return np.exp(-0.0903 * age + 4.3994)  # No need to multiply by 100

# Function to get the full source code
def get_full_source_code():
    current_file = inspect.getfile(inspect.currentframe())
    with open(current_file, "r") as file:
        return file.read()

# Main function to run the Streamlit app
def main():
    st.title("Vehicle Depreciation Analysis")

    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        st.header("1. Data Overview")
        st.write(data.head())

        st.header("2. Scatter Plot of Leftover Percentage vs Age")
        if 'Age' in data.columns and 'Leftover_Percentage' in data.columns and 'Brand_Type' in data.columns:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.scatterplot(x="Age", y="Leftover_Percentage", hue="Brand_Type", data=data, palette="tab10", s=60, edgecolor=None, ax=ax)
            ax.set_title("Scatter Plot of Leftover Percentage vs Age by Brand Type")
            ax.set_xlabel("Age")
            ax.set_ylabel("Leftover Percentage (%)")
            st.pyplot(fig)
        else:
            st.warning("The uploaded dataset doesn't have the required columns for this plot.")

        st.header("3. Linear Regression on Log-Transformed Data")
        if 'Age' in data.columns and 'Leftover_Percentage' in data.columns:
            # Avoid log(0) issue by adding a small constant
            data['Leftover_Percentage'] = data['Leftover_Percentage'].replace(0, np.nan)
            data = data.dropna(subset=['Leftover_Percentage'])
            data['Log_Leftover_Percentage'] = np.log(data['Leftover_Percentage'] + 1e-5)  # Added a small constant to avoid log(0)

            X = data[['Age']]
            y = data['Log_Leftover_Percentage']
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            fig, ax = plt.subplots(figsize=(14, 8))
            sns.scatterplot(x="Age", y="Log_Leftover_Percentage", data=data, palette="tab20", s=60, edgecolor=None, ax=ax)
            ax.plot(X, y_pred, color="red", linewidth=2, label="Regression Line")
            ax.set_title("Linear Regression of Log-Transformed Leftover Percentage vs Age")
            ax.set_xlabel("Age")
            ax.set_ylabel("Log-Transformed Leftover Percentage")
            ax.legend()
            st.pyplot(fig)

            st.write(f"Equation: log(Leftover_Percentage + 1e-5) = {model.intercept_:.4f} + {-model.coef_[0]:.4f} * Age")
        else:
            st.warning("The uploaded dataset doesn't have the required columns for this analysis.")

        st.header("4. Model Comparison (Using Default Parameters)")
        future_years = np.arange(0, 15)  # Extended to 15 years for better visualization
        old_x_values = np.exp(-0.1179 * future_years + 4.2620)
        new_x_values = np.exp(-0.0903 * future_years + 4.3994)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(future_years, old_x_values, color="red", linestyle='--', linewidth=2, label="Old X: y = exp(-0.1179x + 4.2620)")
        ax.plot(future_years, new_x_values, color="blue", linestyle='--', linewidth=2, label="New X: y = exp(-0.0903x + 4.3994)")
        ax.set_title("Comparison of Old and New X Models")
        ax.set_xlabel("Year")
        ax.set_ylabel("Leftover Percentage")
        ax.legend()
        st.pyplot(fig)

        st.header("5. Model Equations")
        model_equations = pd.DataFrame({
            'Model': ['Old X', 'New X'],
            'Equation': [
                'y = exp(-0.1179x + 4.2620)',
                'y = exp(-0.0903x + 4.3994)'
            ]
        })
        st.write(model_equations)

        st.header("6. Depreciation Calculator")
        st.write("Calculate the remaining percentage of the car's value using the new equation.")
        age = st.number_input("Enter the age of the car (in years):", min_value=0, max_value=100, value=0, step=1)
        if st.button("Calculate"):
            remaining_percentage = calculate_remaining_percentage(age)
            st.write(f"The remaining value of the car at age {age} is approximately {remaining_percentage:.2f}%")

        st.header("7. Conclusion")
        st.write("""
        - This analysis is based on the uploaded dataset and may differ from the original study.
        - The scatter plot and linear regression provide insights into the relationship between vehicle age and leftover percentage.
        - The model comparison uses default parameters and may not reflect the specifics of the uploaded data.
        - The depreciation calculator allows you to estimate the remaining value of a car based on its age.
        - For a more accurate analysis, consider adjusting the parameters based on your specific dataset.
        """)
    else:
        st.info("Please upload a CSV file to begin the analysis.")

    # Display the full source code
    st.header("8. Full Source Code")
    st.code(get_full_source_code(), language="python")

if __name__ == "__main__":
    main()

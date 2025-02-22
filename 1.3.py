import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mannwhitneyu, wilcoxon

# Function to calculate additional statistics
def calculate_advanced_statistics(df):
    statistics = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        statistics[col] = {
            "Mean": df[col].mean(),
            "Median": df[col].median(),
            "Mode": df[col].mode()[0] if not df[col].mode().empty else None,
            "Standard Deviation": df[col].std(),
            "Variance": df[col].var(),
            "Range": df[col].max() - df[col].min(),
            "IQR": df[col].quantile(0.75) - df[col].quantile(0.25),
            "Skewness": df[col].skew(),
            "Kurtosis": df[col].kurt(),
            "Coefficient of Variation": df[col].std() / df[col].mean() if df[col].mean() != 0 else None
        }
    return statistics

# Function to perform hypothesis tests
def perform_hypothesis_tests(df, column1, column2):
    results = {}

    # Pearson Correlation
    pearson_corr, pearson_p = stats.pearsonr(df[column1].dropna(), df[column2].dropna())
    results["Pearson Correlation"] = {"Correlation": pearson_corr, "p-value": pearson_p}

    # Spearman Correlation
    spearman_corr, spearman_p = stats.spearmanr(df[column1].dropna(), df[column2].dropna())
    results["Spearman Correlation"] = {"Correlation": spearman_corr, "p-value": spearman_p}

    # Shapiro-Wilk Test (Normality Test)
    shapiro_stat, shapiro_p = stats.shapiro(df[column1].dropna())
    results["Shapiro-Wilk Test (Normality)"] = {"Statistic": shapiro_stat, "p-value": shapiro_p}

    return results

# Regression Analysis Functions
def simple_linear_regression(df, x_col, y_col):
    X = df[[x_col]]
    y = df[y_col]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.summary()

def multiple_linear_regression(df, x_cols, y_col):
    X = df[x_cols]
    y = df[y_col]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.summary()

def logistic_regression(df, x_cols, y_col):
    X = df[x_cols]
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, cm

# Hypothesis Testing Functions
def one_sample_ttest(df, column, value):
    stat, p = stats.ttest_1samp(df[column].dropna(), value)
    return stat, p

def two_sample_ttest(df, column1, column2):
    stat, p = stats.ttest_ind(df[column1].dropna(), df[column2].dropna())
    return stat, p

def paired_ttest(df, column1, column2):
    stat, p = stats.ttest_rel(df[column1].dropna(), df[column2].dropna())
    return stat, p

def anova_test(df, column, group_col):
    groups = [df[df[group_col] == group][column] for group in df[group_col].unique()]
    stat, p = stats.f_oneway(*groups)
    return stat, p

def chi_square_test(df, column1, column2):
    contingency_table = pd.crosstab(df[column1], df[column2])
    stat, p, _, _ = stats.chi2_contingency(contingency_table)
    return stat, p

def mann_whitney_u_test(df, column1, column2):
    stat, p = mannwhitneyu(df[column1].dropna(), df[column2].dropna())
    return stat, p

def wilcoxon_test(df, column1, column2):
    stat, p = wilcoxon(df[column1].dropna(), df[column2].dropna())
    return stat, p

# Streamlit UI
st.title("Advanced Statistical Report Generator")
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview")
    st.write(df.head())

    # Page navigation
    page = st.sidebar.radio("Choose a page", (
        "Descriptive Statistics", 
        "Distribution Analysis", 
        "Regression Analysis", 
        "Hypothesis Testing"
    ))

    if page == "Descriptive Statistics":
        # Perform advanced statistics
        advanced_stats = calculate_advanced_statistics(df)
        st.write("### Advanced Descriptive Statistics")
        st.write(pd.DataFrame(advanced_stats).T)

        # Histogram for each numeric column
        st.write("### Data Distributions (Histograms)")
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_columns:
            fig, ax = plt.subplots()
            sns.histplot(df[col], bins=20, kde=True, ax=ax)
            ax.set_title(f"Histogram of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    elif page == "Distribution Analysis":
        st.write("### Distribution Analysis")

        # Select a column for distribution analysis
        column = st.selectbox("Select a Numeric Column", df.select_dtypes(include=[np.number]).columns.tolist())

        # Histogram & Boxplot
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.histplot(df[column], kde=True, ax=ax[0])
        sns.boxplot(x=df[column], ax=ax[1])
        st.pyplot(fig)

        # Q-Q Plot
        st.write("### Q-Q Plot")
        fig, ax = plt.subplots(figsize=(6, 6))
        stats.probplot(df[column].dropna(), dist="norm", plot=ax)
        st.pyplot(fig)

        # Kernel Density Estimation (KDE)
        #st.write("### Kernel Density Estimation (KDE)")
        #fig, ax = plt.subplots(figsize=(6, 6))
        #sns.kdeplot(df[column], shade=True, ax=ax)
        #st.pyplot(fig)

        # Shapiro-Wilk Test (Normality Test)
        st.write("### Shapiro-Wilk Test (Normality Test)")
        stat, p = stats.shapiro(df[column].dropna())
        st.write(f"Shapiro-Wilk Statistic: {stat}, p-value: {p}")
        
        if p > 0.05:
            st.write("Interpretation: The data seems to be normally distributed.")
        else:
            st.write("Interpretation: The data does not follow a normal distribution.")

        # Kolmogorov-Smirnov Test
        st.write("### Kolmogorov-Smirnov Test")
        stat, p = stats.ks_2samp(df[column].dropna(), np.random.normal(0, 1, size=len(df[column].dropna())))
        st.write(f"KS Statistic: {stat}, p-value: {p}")

    elif page == "Regression Analysis":
        st.write("### Regression Analysis")
        regression_type = st.radio("Choose Regression Type", ("Simple Linear Regression", "Multiple Linear Regression", "Logistic Regression"))

        if regression_type == "Simple Linear Regression":
            st.write("### Simple Linear Regression")
            x_col = st.selectbox("Select Predictor Variable", df.select_dtypes(include=[np.number]).columns.tolist())
            y_col = st.selectbox("Select Response Variable", df.select_dtypes(include=[np.number]).columns.tolist())
            
            if st.button("Perform Simple Linear Regression"):
                st.write(f"### Results for Simple Linear Regression (Predicting {y_col} from {x_col})")
                regression_results = simple_linear_regression(df, x_col, y_col)
                st.write(regression_results)

        elif regression_type == "Multiple Linear Regression":
            st.write("### Multiple Linear Regression")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            x_cols = st.multiselect("Select Predictor Variables", numeric_columns)
            y_col = st.selectbox("Select Response Variable", numeric_columns)
            
            if st.button("Perform Multiple Linear Regression"):
                st.write(f"### Results for Multiple Linear Regression (Predicting {y_col} from {', '.join(x_cols)})")
                regression_results = multiple_linear_regression(df, x_cols, y_col)
                st.write(regression_results)

        elif regression_type == "Logistic Regression":
            st.write("### Logistic Regression")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            x_cols = st.multiselect("Select Predictor Variables", numeric_columns)
            y_col = st.selectbox("Select Response Variable (binary)", numeric_columns)

            if st.button("Perform Logistic Regression"):
                st.write(f"### Results for Logistic Regression (Predicting {y_col} from {', '.join(x_cols)})")
                accuracy, cm = logistic_regression(df, x_cols, y_col)
                st.write(f"Accuracy: {accuracy}")
                st.write(f"Confusion Matrix: \n{cm}")

    elif page == "Hypothesis Testing":
        st.write("### Hypothesis Testing")
        
        test_type = st.radio("Choose Test Type", ("One-sample t-test", "Two-sample t-test", "Paired t-test", "ANOVA", "Chi-square Test", "Mann-Whitney U Test", "Wilcoxon Signed-Rank Test"))

        if test_type == "One-sample t-test":
            column = st.selectbox("Select Column", df.select_dtypes(include=[np.number]).columns.tolist())
            value = st.number_input("Enter the value for comparison", value=0)
            if st.button("Perform One-sample t-test"):
                stat, p = one_sample_ttest(df, column, value)
                st.write(f"t-statistic: {stat}, p-value: {p}")

        if test_type == "Two-sample t-test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[np.number]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[np.number]).columns.tolist())
            if st.button("Perform Two-sample t-test"):
                stat, p = two_sample_ttest(df, column1, column2)
                st.write(f"t-statistic: {stat}, p-value: {p}")

        if test_type == "Paired t-test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[np.number]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[np.number]).columns.tolist())
            if st.button("Perform Paired t-test"):
                stat, p = paired_ttest(df, column1, column2)
                st.write(f"t-statistic: {stat}, p-value: {p}")

        if test_type == "ANOVA":
            column = st.selectbox("Select the Numeric Column", df.select_dtypes(include=[np.number]).columns.tolist())
            group_col = st.selectbox("Select the Grouping Column", df.select_dtypes(include=[object]).columns.tolist())
            if st.button("Perform ANOVA"):
                stat, p = anova_test(df, column, group_col)
                st.write(f"F-statistic: {stat}, p-value: {p}")

        if test_type == "Chi-square Test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[object]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[object]).columns.tolist())
            if st.button("Perform Chi-square Test"):
                stat, p = chi_square_test(df, column1, column2)
                st.write(f"Chi-square statistic: {stat}, p-value: {p}")

        if test_type == "Mann-Whitney U Test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[np.number]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[np.number]).columns.tolist())
            if st.button("Perform Mann-Whitney U Test"):
                stat, p = mann_whitney_u_test(df, column1, column2)
                st.write(f"U-statistic: {stat}, p-value: {p}")

        if test_type == "Wilcoxon Signed-Rank Test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[np.number]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[np.number]).columns.tolist())
            if st.button("Perform Wilcoxon Test"):
                stat, p = wilcoxon_test(df, column1, column2)
                st.write(f"Wilcoxon statistic: {stat}, p-value: {p}")

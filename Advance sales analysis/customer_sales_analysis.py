"""
Multi-Faceted Customer and Sales Data Analysis
==============================================

This comprehensive analysis includes:
1. Data Preprocessing and Feature Engineering
2. Exploratory Data Analysis (EDA)
3. Customer Segmentation
4. Sales Forecasting
5. Predictive Modeling for Customer Churn
6. Product Analysis and Cross-Selling
7. Interactive Reporting and Visualization

Author: Data Analysis Team
Date: August 2025
"""

from scipy.stats import zscore
from scipy import stats
import random
from datetime import datetime, timedelta
import datetime
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier, XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Data Preprocessing and Feature Engineering

# Clustering and Segmentation

# Time Series Analysis

# Machine Learning Models

# Association Rule Mining

# Neural Networks


class CustomerSalesAnalyzer:
    """
    Comprehensive Customer and Sales Data Analyzer
    """

    def __init__(self):
        """Initialize the analyzer with default settings."""
        self.data = None
        self.processed_data = None
        self.customer_segments = None
        self.forecasting_models = {}
        self.churn_models = {}
        self.association_rules = None

        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def generate_sample_dataset(self, n_customers=5000, n_transactions=20000):
        """
        Generate a comprehensive sample dataset for analysis.

        Args:
            n_customers (int): Number of customers to generate
            n_transactions (int): Number of transactions to generate

        Returns:
            pd.DataFrame: Generated dataset
        """
        print("🔄 Generating sample dataset...")

        # Set seed for reproducibility
        np.random.seed(42)
        random.seed(42)

        # Generate customer data
        customer_data = []

        for i in range(1, n_customers + 1):
            customer = {
                'Customer_ID': f'CUST_{i:05d}',
                'Customer_Name': f'Customer_{i}',
                'Age': np.random.normal(40, 12),
                'Gender': np.random.choice(['Male', 'Female'], p=[0.52, 0.48]),
                'Income_Level': np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2]),
                'Location': np.random.choice(['North', 'South', 'East', 'West', 'Central'], p=[0.2, 0.2, 0.2, 0.2, 0.2]),
                'Registration_Date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=np.random.randint(0, 1400))
            }
            customer_data.append(customer)

        customers_df = pd.DataFrame(customer_data)

        # Generate product data
        categories = ['Electronics', 'Clothing', 'Home & Garden',
                      'Sports', 'Books', 'Beauty', 'Automotive']
        subcategories = {
            'Electronics': ['Mobile', 'Laptop', 'TV', 'Audio', 'Gaming'],
            'Clothing': ['Men', 'Women', 'Kids', 'Accessories'],
            'Home & Garden': ['Furniture', 'Kitchen', 'Garden', 'Decor'],
            'Sports': ['Fitness', 'Outdoor', 'Team Sports', 'Water Sports'],
            'Books': ['Fiction', 'Non-Fiction', 'Educational', 'Children'],
            'Beauty': ['Skincare', 'Makeup', 'Hair', 'Fragrance'],
            'Automotive': ['Parts', 'Accessories', 'Tools', 'Care']
        }

        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD',
                  'BrandE', 'BrandF', 'BrandG', 'BrandH']

        products = []
        for i in range(1, 201):  # 200 products
            category = np.random.choice(categories)
            subcategory = np.random.choice(subcategories[category])
            cost_price = np.random.uniform(10, 500)
            selling_price = cost_price * np.random.uniform(1.2, 2.5)

            product = {
                'Product_ID': f'PROD_{i:03d}',
                'Category': category,
                'Sub_Category': subcategory,
                'Brand': np.random.choice(brands),
                'Cost_Price': cost_price,
                'Selling_Price': selling_price,
                'Margin': selling_price - cost_price
            }
            products.append(product)

        products_df = pd.DataFrame(products)

        # Generate transaction data
        transactions = []

        for i in range(1, n_transactions + 1):
            customer_id = np.random.choice(customers_df['Customer_ID'])
            product_id = np.random.choice(products_df['Product_ID'])
            product_info = products_df[products_df['Product_ID']
                                       == product_id].iloc[0]

            # Create some seasonality and trends
            base_date = pd.Timestamp('2020-01-01')
            days_offset = np.random.randint(0, 1800)  # ~5 years of data
            purchase_date = base_date + pd.Timedelta(days=days_offset)

            # Add seasonal effects
            month = purchase_date.month
            seasonal_multiplier = 1.0
            if month in [11, 12]:  # Holiday season
                seasonal_multiplier = 1.5
            elif month in [6, 7, 8]:  # Summer
                seasonal_multiplier = 1.2

            quantity = max(1, int(np.random.poisson(2) * seasonal_multiplier))
            unit_price = product_info['Selling_Price']

            # Apply discount
            discount_rate = np.random.choice([0, 0.05, 0.10, 0.15, 0.20, 0.25],
                                             p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
            discount_amount = unit_price * quantity * discount_rate

            transaction = {
                'Transaction_ID': f'TXN_{i:06d}',
                'Customer_ID': customer_id,
                'Product_ID': product_id,
                'Purchase_Date': purchase_date,
                'Purchase_Quantity': quantity,
                'Unit_Price': unit_price,
                'Discount_Rate': discount_rate,
                'Discount_Amount': discount_amount,
                'Total_Amount': (unit_price * quantity) - discount_amount,
                'Payment_Method': np.random.choice(['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet'],
                                                   p=[0.4, 0.3, 0.15, 0.15]),
                'Sales_Channel': np.random.choice(['Online', 'Store', 'Mobile App'], p=[0.5, 0.3, 0.2]),
                'Payment_Status': np.random.choice(['Completed', 'Pending', 'Failed'], p=[0.92, 0.05, 0.03])
            }
            transactions.append(transaction)

        transactions_df = pd.DataFrame(transactions)

        # Merge all data
        # First merge transactions with products
        merged_data = transactions_df.merge(
            products_df, on='Product_ID', how='left')

        # Then merge with customers
        final_data = merged_data.merge(
            customers_df, on='Customer_ID', how='left')

        # Add some missing values to simulate real-world data
        missing_indices = np.random.choice(
            final_data.index, size=int(0.02 * len(final_data)), replace=False)
        final_data.loc[missing_indices, 'Age'] = np.nan

        missing_indices = np.random.choice(
            final_data.index, size=int(0.01 * len(final_data)), replace=False)
        final_data.loc[missing_indices, 'Income_Level'] = np.nan

        self.data = final_data
        print(
            f"✅ Generated dataset with {len(final_data)} transactions, {len(customers_df)} customers, and {len(products_df)} products")
        return final_data

    def preprocess_data(self):
        """
        Advanced data preprocessing and feature engineering.
        """
        print("\n🔄 Starting data preprocessing and feature engineering...")

        if self.data is None:
            raise ValueError(
                "No data loaded. Please load or generate data first.")

        data = self.data.copy()

        # 1. Handle Missing Values
        print("📋 Handling missing values...")

        # Check missing values
        missing_summary = data.isnull().sum()
        print(
            f"Missing values found: \n{missing_summary[missing_summary > 0]}")

        # Handle categorical missing values
        if 'Income_Level' in data.columns and data['Income_Level'].isnull().any():
            data['Income_Level'].fillna(
                data['Income_Level'].mode()[0], inplace=True)

        # Handle numerical missing values using KNN Imputer
        if data['Age'].isnull().any():
            knn_imputer = KNNImputer(n_neighbors=5)
            data['Age'] = knn_imputer.fit_transform(data[['Age']]).flatten()

        # 2. Detect and Handle Outliers
        print("🎯 Detecting and handling outliers...")

        # Outlier detection using IQR method for numerical columns
        numerical_cols = ['Age', 'Purchase_Quantity',
                          'Unit_Price', 'Total_Amount']

        for col in numerical_cols:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers instead of removing them
                data[col] = np.clip(data[col], lower_bound, upper_bound)

        # 3. Feature Engineering
        print("🛠️ Creating new features...")

        # Convert date column
        data['Purchase_Date'] = pd.to_datetime(data['Purchase_Date'])

        # Extract date features
        data['Year'] = data['Purchase_Date'].dt.year
        data['Month'] = data['Purchase_Date'].dt.month
        data['Day_of_Week'] = data['Purchase_Date'].dt.dayofweek
        data['Quarter'] = data['Purchase_Date'].dt.quarter
        data['Is_Weekend'] = data['Day_of_Week'].isin([5, 6]).astype(int)
        data['Hour'] = data['Purchase_Date'].dt.hour

        # Customer-level features
        customer_features = data.groupby('Customer_ID').agg({
            'Total_Amount': ['sum', 'mean', 'count'],
            'Purchase_Date': ['min', 'max'],
            'Discount_Amount': 'sum',
            'Purchase_Quantity': 'sum'
        }).round(2)

        # Flatten column names
        customer_features.columns = [
            '_'.join(col).strip() for col in customer_features.columns.values]
        customer_features = customer_features.reset_index()

        # Calculate additional customer metrics
        current_date = data['Purchase_Date'].max()
        customer_features['Days_Since_Last_Purchase'] = (
            current_date - customer_features['Purchase_Date_max']
        ).dt.days

        customer_features['Customer_Lifetime_Days'] = (
            customer_features['Purchase_Date_max'] -
            customer_features['Purchase_Date_min']
        ).dt.days + 1

        customer_features['Purchase_Frequency'] = (
            customer_features['Total_Amount_count'] /
            customer_features['Customer_Lifetime_Days'] * 30
        ).round(2)

        customer_features['Average_Order_Value'] = customer_features['Total_Amount_mean']
        customer_features['Total_Spent'] = customer_features['Total_Amount_sum']
        customer_features['Total_Discount_Used'] = customer_features['Discount_Amount_sum']
        customer_features['Discount_Usage_Rate'] = (
            customer_features['Total_Discount_Used'] /
            customer_features['Total_Spent']
        ).round(3)

        # RFM Analysis components
        customer_features['Recency'] = customer_features['Days_Since_Last_Purchase']
        customer_features['Frequency'] = customer_features['Total_Amount_count']
        customer_features['Monetary'] = customer_features['Total_Spent']

        # Calculate Customer Lifetime Value (CLV)
        customer_features['CLV'] = (
            customer_features['Average_Order_Value'] *
            customer_features['Purchase_Frequency'] *
            customer_features['Customer_Lifetime_Days'] / 365
        ).round(2)

        # Loyalty Score (based on multiple factors)
        customer_features['Loyalty_Score'] = (
            (customer_features['Frequency'] / customer_features['Frequency'].max() * 0.4) +
            (customer_features['Monetary'] / customer_features['Monetary'].max() * 0.4) +
            ((customer_features['Customer_Lifetime_Days'] /
             customer_features['Customer_Lifetime_Days'].max()) * 0.2)
        ).round(3)

        # Merge customer features back to main dataset
        data = data.merge(customer_features[['Customer_ID', 'Purchase_Frequency', 'CLV',
                                             'Loyalty_Score', 'Recency', 'Frequency', 'Monetary',
                                             'Discount_Usage_Rate']],
                          on='Customer_ID', how='left')

        # Product-level features
        data['Profit_Margin'] = (
            (data['Selling_Price'] - data['Cost_Price']) / data['Selling_Price']).round(3)
        data['Discounted_Price'] = data['Unit_Price'] - \
            (data['Unit_Price'] * data['Discount_Rate'])

        # 4. Encode Categorical Variables
        print("🔢 Encoding categorical variables...")

        # Create label encoders for categorical variables
        categorical_cols = ['Gender', 'Income_Level', 'Location', 'Category', 'Sub_Category',
                            'Brand', 'Payment_Method', 'Sales_Channel', 'Payment_Status']

        label_encoders = {}
        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[f'{col}_Encoded'] = le.fit_transform(
                    data[col].astype(str))
                label_encoders[col] = le

        # 5. Normalize Numerical Features
        print("📊 Normalizing numerical features...")

        # Select numerical columns for scaling
        numerical_features = ['Age', 'Purchase_Quantity', 'Unit_Price', 'Total_Amount',
                              'CLV', 'Loyalty_Score', 'Purchase_Frequency', 'Discount_Usage_Rate']

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[numerical_features])
        scaled_df = pd.DataFrame(scaled_features, columns=[
                                 f'{col}_Scaled' for col in numerical_features])

        # Combine with original data
        self.processed_data = pd.concat(
            [data.reset_index(drop=True), scaled_df], axis=1)

        print("✅ Data preprocessing completed successfully!")
        print(f"📊 Final dataset shape: {self.processed_data.shape}")

        return self.processed_data

    def exploratory_data_analysis(self):
        """
        Comprehensive Exploratory Data Analysis with advanced visualizations.
        """
        print("\n🔍 Starting Exploratory Data Analysis...")

        if self.processed_data is None:
            raise ValueError(
                "Data not processed. Please run preprocess_data() first.")

        data = self.processed_data

        # Create output directory for plots
        import os
        plot_dir = "analysis_plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # 1. Dataset Overview
        print("📋 Dataset Overview:")
        print(f"Shape: {data.shape}")
        print(
            f"Date range: {data['Purchase_Date'].min()} to {data['Purchase_Date'].max()}")
        print(f"Unique customers: {data['Customer_ID'].nunique()}")
        print(f"Unique products: {data['Product_ID'].nunique()}")
        print(f"Total revenue: ${data['Total_Amount'].sum():,.2f}")

        # 2. Sales Trends Over Time
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Sales Trend', 'Monthly Sales Distribution',
                            'Sales by Day of Week', 'Sales by Quarter'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        # Daily sales trend
        daily_sales = data.groupby(data['Purchase_Date'].dt.date)[
            'Total_Amount'].sum().reset_index()
        fig.add_trace(
            go.Scatter(x=daily_sales['Purchase_Date'], y=daily_sales['Total_Amount'],
                       mode='lines', name='Daily Sales'),
            row=1, col=1
        )

        # Monthly distribution
        monthly_sales = data.groupby(
            'Month')['Total_Amount'].sum().reset_index()
        fig.add_trace(
            go.Bar(x=monthly_sales['Month'], y=monthly_sales['Total_Amount'],
                   name='Monthly Sales'),
            row=1, col=2
        )

        # Day of week
        dow_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                   4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        data['Day_Name'] = data['Day_of_Week'].map(dow_map)
        dow_sales = data.groupby('Day_Name')[
            'Total_Amount'].sum().reset_index()
        fig.add_trace(
            go.Bar(x=dow_sales['Day_Name'], y=dow_sales['Total_Amount'],
                   name='Sales by Day'),
            row=2, col=1
        )

        # Quarterly sales
        quarterly_sales = data.groupby(
            'Quarter')['Total_Amount'].sum().reset_index()
        fig.add_trace(
            go.Bar(x=quarterly_sales['Quarter'], y=quarterly_sales['Total_Amount'],
                   name='Quarterly Sales'),
            row=2, col=2
        )

        fig.update_layout(
            height=800, title_text="Sales Analysis Across Time Dimensions")
        fig.write_html(f"{plot_dir}/sales_time_analysis.html")

        # 3. Customer Demographics Analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Age distribution
        data['Age'].hist(bins=30, ax=axes[0, 0], alpha=0.7)
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Frequency')

        # Gender distribution
        gender_counts = data['Gender'].value_counts()
        axes[0, 1].pie(gender_counts.values,
                       labels=gender_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Gender Distribution')

        # Income level distribution
        income_counts = data['Income_Level'].value_counts()
        axes[0, 2].bar(income_counts.index, income_counts.values)
        axes[0, 2].set_title('Income Level Distribution')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # Location distribution
        location_counts = data['Location'].value_counts()
        axes[1, 0].bar(location_counts.index, location_counts.values)
        axes[1, 0].set_title('Customer Location Distribution')

        # CLV distribution
        data['CLV'].hist(bins=50, ax=axes[1, 1], alpha=0.7)
        axes[1, 1].set_title('Customer Lifetime Value Distribution')
        axes[1, 1].set_xlabel('CLV ($)')

        # Loyalty Score distribution
        data['Loyalty_Score'].hist(bins=30, ax=axes[1, 2], alpha=0.7)
        axes[1, 2].set_title('Loyalty Score Distribution')
        axes[1, 2].set_xlabel('Loyalty Score')

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/customer_demographics.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Product Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Category performance
        category_sales = data.groupby(
            'Category')['Total_Amount'].sum().sort_values(ascending=False)
        category_sales.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Sales by Product Category')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Profit margin analysis
        profit_by_category = data.groupby(
            'Category')['Profit_Margin'].mean().sort_values(ascending=False)
        profit_by_category.plot(
            kind='bar', ax=axes[0, 1], color='green', alpha=0.7)
        axes[0, 1].set_title('Average Profit Margin by Category')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Discount impact
        discount_impact = data.groupby(pd.cut(data['Discount_Rate'], bins=5))[
            'Purchase_Quantity'].mean()
        discount_impact.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Purchase Quantity vs Discount Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Brand performance
        brand_sales = data.groupby(
            'Brand')['Total_Amount'].sum().sort_values(ascending=False)
        brand_sales.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Sales by Brand')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/product_analysis.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Correlation Analysis
        plt.figure(figsize=(14, 10))

        # Select numerical columns for correlation
        correlation_cols = ['Age', 'Purchase_Quantity', 'Unit_Price', 'Total_Amount',
                            'CLV', 'Loyalty_Score', 'Purchase_Frequency', 'Discount_Usage_Rate',
                            'Profit_Margin', 'Recency', 'Frequency', 'Monetary']

        correlation_matrix = data[correlation_cols].corr()

        # Create correlation heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                    center=0, square=True, fmt='.2f')
        plt.title('Correlation Matrix of Key Variables')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/correlation_matrix.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 6. Advanced Statistical Analysis
        print("\n📊 Statistical Summary:")
        print(data[correlation_cols].describe())

        # Seasonal decomposition
        daily_sales = data.groupby(data['Purchase_Date'].dt.date)[
            'Total_Amount'].sum()
        daily_sales.index = pd.to_datetime(daily_sales.index)
        daily_sales = daily_sales.sort_index()

        if len(daily_sales) > 365:  # Only if we have enough data
            plt.figure(figsize=(15, 10))
            decomposition = seasonal_decompose(
                daily_sales, model='additive', period=365)

            plt.subplot(411)
            decomposition.observed.plot(title='Original Sales Data')
            plt.subplot(412)
            decomposition.trend.plot(title='Trend')
            plt.subplot(413)
            decomposition.seasonal.plot(title='Seasonal')
            plt.subplot(414)
            decomposition.resid.plot(title='Residual')

            plt.tight_layout()
            plt.savefig(f"{plot_dir}/seasonal_decomposition.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

        print("✅ Exploratory Data Analysis completed!")
        print(f"📊 Plots saved in '{plot_dir}' directory")

        return {
            'daily_sales': daily_sales,
            'correlation_matrix': correlation_matrix,
            'category_performance': category_sales,
            'customer_metrics': data[correlation_cols].describe()
        }

    def customer_segmentation(self):
        """
        Advanced customer segmentation using multiple clustering algorithms.
        """
        print("\n👥 Starting Customer Segmentation Analysis...")

        if self.processed_data is None:
            raise ValueError(
                "Data not processed. Please run preprocess_data() first.")

        # Prepare customer-level data for clustering
        customer_data = self.processed_data.groupby('Customer_ID').agg({
            'Age': 'first',
            'CLV': 'first',
            'Loyalty_Score': 'first',
            'Purchase_Frequency': 'first',
            'Recency': 'first',
            'Frequency': 'first',
            'Monetary': 'first',
            'Discount_Usage_Rate': 'first',
            'Gender': 'first',
            'Income_Level': 'first',
            'Location': 'first'
        }).reset_index()

        # Prepare features for clustering
        clustering_features = ['Age', 'CLV', 'Loyalty_Score', 'Purchase_Frequency',
                               'Recency', 'Frequency', 'Monetary', 'Discount_Usage_Rate']

        # Handle any missing values
        for col in clustering_features:
            customer_data[col].fillna(
                customer_data[col].median(), inplace=True)

        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(
            customer_data[clustering_features])

        # 1. RFM Analysis with K-Means
        print("🎯 Performing RFM Analysis with K-Means clustering...")

        rfm_features = ['Recency', 'Frequency', 'Monetary']
        rfm_scaled = scaler.fit_transform(customer_data[rfm_features])

        # Determine optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, 11)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(rfm_scaled)
            inertias.append(kmeans.inertia_)

        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k (RFM)')
        plt.grid(True)
        plt.savefig("analysis_plots/elbow_curve_rfm.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Use k=5 for RFM segmentation
        kmeans_rfm = KMeans(n_clusters=5, random_state=42, n_init=10)
        customer_data['RFM_Segment'] = kmeans_rfm.fit_predict(rfm_scaled)

        # 2. Advanced Clustering with Gaussian Mixture Models
        print("🧬 Applying Gaussian Mixture Models...")

        # Determine optimal number of components using BIC
        n_components_range = range(2, 11)
        bic_scores = []
        aic_scores = []

        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(scaled_features)
            bic_scores.append(gmm.bic(scaled_features))
            aic_scores.append(gmm.aic(scaled_features))

        # Plot BIC scores
        plt.figure(figsize=(10, 6))
        plt.plot(n_components_range, bic_scores, 'ro-', label='BIC')
        plt.plot(n_components_range, aic_scores, 'bo-', label='AIC')
        plt.xlabel('Number of Components')
        plt.ylabel('Information Criterion')
        plt.title('Model Selection for Gaussian Mixture Model')
        plt.legend()
        plt.grid(True)
        plt.savefig("analysis_plots/gmm_model_selection.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Use optimal number of components (minimum BIC)
        optimal_components = n_components_range[np.argmin(bic_scores)]
        gmm = GaussianMixture(n_components=optimal_components, random_state=42)
        customer_data['GMM_Segment'] = gmm.fit_predict(scaled_features)

        # 3. Agglomerative Clustering
        print("🌳 Applying Agglomerative Clustering...")

        agg_clustering = AgglomerativeClustering(n_clusters=5, linkage='ward')
        customer_data['Agg_Segment'] = agg_clustering.fit_predict(
            scaled_features)

        # 4. Analyze Segments
        print("📊 Analyzing customer segments...")

        # RFM Segment Analysis
        rfm_analysis = customer_data.groupby('RFM_Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Age': 'mean',
            'CLV': 'mean',
            'Loyalty_Score': 'mean',
            'Customer_ID': 'count'
        }).round(2)
        rfm_analysis.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary',
                                'Avg_Age', 'Avg_CLV', 'Avg_Loyalty', 'Customer_Count']

        print("\n🎯 RFM Segment Analysis:")
        print(rfm_analysis)

        # Define segment names based on RFM characteristics
        segment_names = {
            'RFM_Segment': {
                0: 'Champions',
                1: 'Loyal Customers',
                2: 'Potential Loyalists',
                3: 'At Risk',
                4: 'Hibernating'
            }
        }

        # Map segment names
        customer_data['RFM_Segment_Name'] = customer_data['RFM_Segment'].map(
            segment_names['RFM_Segment'])

        # 5. Visualization
        # Create comprehensive segmentation plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # RFM Segments visualization
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        # Recency vs Frequency
        for segment in customer_data['RFM_Segment'].unique():
            segment_data = customer_data[customer_data['RFM_Segment'] == segment]
            axes[0, 0].scatter(segment_data['Recency'], segment_data['Frequency'],
                               c=colors[segment], label=f'Segment {segment}', alpha=0.6)
        axes[0, 0].set_xlabel('Recency (Days)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('RFM Segments: Recency vs Frequency')
        axes[0, 0].legend()

        # Frequency vs Monetary
        for segment in customer_data['RFM_Segment'].unique():
            segment_data = customer_data[customer_data['RFM_Segment'] == segment]
            axes[0, 1].scatter(segment_data['Frequency'], segment_data['Monetary'],
                               c=colors[segment], label=f'Segment {segment}', alpha=0.6)
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('Monetary Value')
        axes[0, 1].set_title('RFM Segments: Frequency vs Monetary')
        axes[0, 1].legend()

        # CLV vs Loyalty Score
        for segment in customer_data['RFM_Segment'].unique():
            segment_data = customer_data[customer_data['RFM_Segment'] == segment]
            axes[0, 2].scatter(segment_data['CLV'], segment_data['Loyalty_Score'],
                               c=colors[segment], label=f'Segment {segment}', alpha=0.6)
        axes[0, 2].set_xlabel('Customer Lifetime Value')
        axes[0, 2].set_ylabel('Loyalty Score')
        axes[0, 2].set_title('CLV vs Loyalty Score by Segment')
        axes[0, 2].legend()

        # Segment size distribution
        segment_counts = customer_data['RFM_Segment_Name'].value_counts()
        axes[1, 0].pie(segment_counts.values,
                       labels=segment_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Customer Segment Distribution')

        # Age distribution by segment
        customer_data.boxplot(column='Age', by='RFM_Segment', ax=axes[1, 1])
        axes[1, 1].set_title('Age Distribution by Segment')
        axes[1, 1].set_xlabel('RFM Segment')

        # CLV distribution by segment
        customer_data.boxplot(column='CLV', by='RFM_Segment', ax=axes[1, 2])
        axes[1, 2].set_title('CLV Distribution by Segment')
        axes[1, 2].set_xlabel('RFM Segment')

        plt.tight_layout()
        plt.savefig("analysis_plots/customer_segmentation.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 6. Segment Profiling
        print("\n👥 Detailed Segment Profiles:")

        for segment_id, segment_name in segment_names['RFM_Segment'].items():
            segment_data = customer_data[customer_data['RFM_Segment']
                                         == segment_id]

            print(f"\n🎯 {segment_name} (Segment {segment_id}):")
            print(
                f"   Size: {len(segment_data)} customers ({len(segment_data)/len(customer_data)*100:.1f}%)")
            print(f"   Avg Recency: {segment_data['Recency'].mean():.1f} days")
            print(
                f"   Avg Frequency: {segment_data['Frequency'].mean():.1f} purchases")
            print(f"   Avg Monetary: ${segment_data['Monetary'].mean():.2f}")
            print(f"   Avg CLV: ${segment_data['CLV'].mean():.2f}")
            print(
                f"   Avg Loyalty Score: {segment_data['Loyalty_Score'].mean():.3f}")
            print(
                f"   Most common location: {segment_data['Location'].mode().iloc[0] if not segment_data['Location'].mode().empty else 'N/A'}")
            print(
                f"   Most common income level: {segment_data['Income_Level'].mode().iloc[0] if not segment_data['Income_Level'].mode().empty else 'N/A'}")

        self.customer_segments = customer_data

        print("✅ Customer segmentation completed!")

        return {
            'customer_segments': customer_data,
            'rfm_analysis': rfm_analysis,
            'segment_names': segment_names,
            'clustering_models': {
                'kmeans_rfm': kmeans_rfm,
                'gmm': gmm,
                'agg_clustering': agg_clustering
            }
        }

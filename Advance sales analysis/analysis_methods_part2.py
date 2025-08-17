"""
Continuation of the CustomerSalesAnalyzer class - Part 2
This file contains the remaining methods for the comprehensive analysis.
"""

# Add these methods to the CustomerSalesAnalyzer class


def sales_forecasting(self):
    """
    Advanced sales forecasting using multiple time-series models.
    """
    print("\n📈 Starting Sales Forecasting Analysis...")

    if self.processed_data is None:
        raise ValueError(
            "Data not processed. Please run preprocess_data() first.")

    # Prepare time series data
    daily_sales = self.processed_data.groupby(
        self.processed_data['Purchase_Date'].dt.date
    )['Total_Amount'].sum().reset_index()
    daily_sales['Purchase_Date'] = pd.to_datetime(daily_sales['Purchase_Date'])
    daily_sales = daily_sales.sort_values(
        'Purchase_Date').reset_index(drop=True)

    # Split data for training and testing
    train_size = int(len(daily_sales) * 0.8)
    train_data = daily_sales[:train_size]
    test_data = daily_sales[train_size:]

    print(f"📊 Training data: {len(train_data)} days")
    print(f"🧪 Testing data: {len(test_data)} days")

    forecasting_results = {}

    # 1. ARIMA Model
    print("🔮 Training ARIMA model...")
    try:
        # Auto-determine ARIMA parameters
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller

        # Check stationarity
        adf_result = adfuller(train_data['Total_Amount'])
        print(f"ADF Statistic: {adf_result[0]:.6f}")
        print(f"p-value: {adf_result[1]:.6f}")

        # Fit ARIMA model
        arima_model = ARIMA(train_data['Total_Amount'], order=(1, 1, 1))
        arima_fitted = arima_model.fit()

        # Forecast
        arima_forecast = arima_fitted.forecast(steps=len(test_data))
        arima_forecast_df = pd.DataFrame({
            'Date': test_data['Purchase_Date'],
            'Actual': test_data['Total_Amount'].values,
            'ARIMA_Forecast': arima_forecast
        })

        # Calculate metrics
        arima_rmse = np.sqrt(mean_squared_error(
            test_data['Total_Amount'], arima_forecast))
        arima_mae = mean_absolute_error(
            test_data['Total_Amount'], arima_forecast)
        arima_mape = np.mean(np.abs(
            (test_data['Total_Amount'] - arima_forecast) / test_data['Total_Amount'])) * 100

        forecasting_results['ARIMA'] = {
            'model': arima_fitted,
            'forecast': arima_forecast_df,
            'rmse': arima_rmse,
            'mae': arima_mae,
            'mape': arima_mape
        }

        print(
            f"✅ ARIMA - RMSE: ${arima_rmse:.2f}, MAE: ${arima_mae:.2f}, MAPE: {arima_mape:.2f}%")

    except Exception as e:
        print(f"❌ ARIMA model failed: {str(e)}")

    # 2. Prophet Model
    print("🔮 Training Prophet model...")
    try:
        # Prepare data for Prophet
        prophet_train = train_data[['Purchase_Date', 'Total_Amount']].copy()
        prophet_train.columns = ['ds', 'y']

        # Initialize and fit Prophet model
        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        prophet_model.fit(prophet_train)

        # Create future dataframe
        future = prophet_model.make_future_dataframe(periods=len(test_data))

        # Make predictions
        prophet_forecast = prophet_model.predict(future)

        # Extract test predictions
        prophet_test_forecast = prophet_forecast.tail(len(test_data))[
            'yhat'].values

        prophet_forecast_df = pd.DataFrame({
            'Date': test_data['Purchase_Date'],
            'Actual': test_data['Total_Amount'].values,
            'Prophet_Forecast': prophet_test_forecast
        })

        # Calculate metrics
        prophet_rmse = np.sqrt(mean_squared_error(
            test_data['Total_Amount'], prophet_test_forecast))
        prophet_mae = mean_absolute_error(
            test_data['Total_Amount'], prophet_test_forecast)
        prophet_mape = np.mean(np.abs(
            (test_data['Total_Amount'] - prophet_test_forecast) / test_data['Total_Amount'])) * 100

        forecasting_results['Prophet'] = {
            'model': prophet_model,
            'forecast': prophet_forecast_df,
            'full_forecast': prophet_forecast,
            'rmse': prophet_rmse,
            'mae': prophet_mae,
            'mape': prophet_mape
        }

        print(
            f"✅ Prophet - RMSE: ${prophet_rmse:.2f}, MAE: ${prophet_mae:.2f}, MAPE: {prophet_mape:.2f}%")

    except Exception as e:
        print(f"❌ Prophet model failed: {str(e)}")

    # 3. LSTM Neural Network
    print("🧠 Training LSTM model...")
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from sklearn.preprocessing import MinMaxScaler

        # Prepare data for LSTM
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(daily_sales[['Total_Amount']])

        # Create sequences
        def create_sequences(data, sequence_length):
            X, y = [], []
            for i in range(len(data) - sequence_length):
                X.append(data[i:(i + sequence_length)])
                y.append(data[i + sequence_length])
            return np.array(X), np.array(y)

        sequence_length = 30  # Use 30 days to predict next day
        X, y = create_sequences(scaled_data, sequence_length)

        # Split for training
        train_size_lstm = int(len(X) * 0.8)
        X_train, X_test = X[:train_size_lstm], X[train_size_lstm:]
        y_train, y_test = y[:train_size_lstm], y[train_size_lstm:]

        # Build LSTM model
        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        # Train model
        lstm_model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)

        # Make predictions
        lstm_predictions = lstm_model.predict(X_test)

        # Inverse transform predictions
        lstm_predictions = scaler.inverse_transform(lstm_predictions)
        y_test_actual = scaler.inverse_transform(y_test)

        # Create forecast dataframe
        test_dates = daily_sales['Purchase_Date'].iloc[-len(
            lstm_predictions):].values
        lstm_forecast_df = pd.DataFrame({
            'Date': test_dates,
            'Actual': y_test_actual.flatten(),
            'LSTM_Forecast': lstm_predictions.flatten()
        })

        # Calculate metrics
        lstm_rmse = np.sqrt(mean_squared_error(
            y_test_actual, lstm_predictions))
        lstm_mae = mean_absolute_error(y_test_actual, lstm_predictions)
        lstm_mape = np.mean(
            np.abs((y_test_actual - lstm_predictions) / y_test_actual)) * 100

        forecasting_results['LSTM'] = {
            'model': lstm_model,
            'scaler': scaler,
            'forecast': lstm_forecast_df,
            'rmse': lstm_rmse,
            'mae': lstm_mae,
            'mape': lstm_mape
        }

        print(
            f"✅ LSTM - RMSE: ${lstm_rmse:.2f}, MAE: ${lstm_mae:.2f}, MAPE: {lstm_mape:.2f}%")

    except Exception as e:
        print(f"❌ LSTM model failed: {str(e)}")

    # 4. Visualization of Results
    print("📊 Creating forecast visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Plot actual vs predicted for each model
    models = ['ARIMA', 'Prophet', 'LSTM']

    for i, model_name in enumerate(models):
        if model_name in forecasting_results:
            forecast_df = forecasting_results[model_name]['forecast']

            row = i // 2
            col = i % 2

            axes[row, col].plot(forecast_df['Date'], forecast_df['Actual'],
                                label='Actual', linewidth=2)
            axes[row, col].plot(forecast_df['Date'], forecast_df[f'{model_name}_Forecast'],
                                label=f'{model_name} Forecast', linewidth=2, linestyle='--')
            axes[row, col].set_title(f'{model_name} Sales Forecast')
            axes[row, col].set_xlabel('Date')
            axes[row, col].set_ylabel('Sales Amount')
            axes[row, col].legend()
            axes[row, col].tick_params(axis='x', rotation=45)

    # Model comparison
    if len(forecasting_results) > 0:
        model_names = list(forecasting_results.keys())
        rmse_values = [forecasting_results[model]['rmse']
                       for model in model_names]
        mae_values = [forecasting_results[model]['mae']
                      for model in model_names]
        mape_values = [forecasting_results[model]['mape']
                       for model in model_names]

        x = np.arange(len(model_names))
        width = 0.25

        axes[1, 1].bar(x - width, rmse_values, width, label='RMSE', alpha=0.8)
        axes[1, 1].bar(x, mae_values, width, label='MAE', alpha=0.8)
        axes[1, 1].bar(x + width, mape_values, width,
                       label='MAPE (%)', alpha=0.8)

        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Error Metrics')
        axes[1, 1].set_title('Model Performance Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names)
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("analysis_plots/sales_forecasting.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Future Predictions
    print("🔮 Generating future predictions...")

    # Generate predictions for next 30 days
    future_dates = pd.date_range(start=daily_sales['Purchase_Date'].max() + pd.Timedelta(days=1),
                                 periods=30, freq='D')

    future_predictions = pd.DataFrame({'Date': future_dates})

    # Prophet future predictions
    if 'Prophet' in forecasting_results:
        prophet_model = forecasting_results['Prophet']['model']
        future_prophet = prophet_model.make_future_dataframe(periods=30)
        prophet_future_forecast = prophet_model.predict(future_prophet)
        future_predictions['Prophet_Forecast'] = prophet_future_forecast.tail(30)[
            'yhat'].values

    print("📊 Future Sales Predictions (Next 30 days):")
    if 'Prophet_Forecast' in future_predictions.columns:
        print(
            f"Average daily sales forecast: ${future_predictions['Prophet_Forecast'].mean():.2f}")
        print(
            f"Total monthly forecast: ${future_predictions['Prophet_Forecast'].sum():.2f}")

    self.forecasting_models = forecasting_results

    print("✅ Sales forecasting completed!")

    return {
        'models': forecasting_results,
        'future_predictions': future_predictions,
        'daily_sales': daily_sales
    }


def churn_prediction(self):
    """
    Predictive modeling for customer churn using advanced machine learning algorithms.
    """
    print("\n🚨 Starting Customer Churn Prediction Analysis...")

    if self.customer_segments is None:
        print("⚠️ Customer segmentation not performed. Running segmentation first...")
        self.customer_segmentation()

    # Define churn based on recency (customers who haven't purchased in last 90 days)
    churn_threshold = 90
    customer_data = self.customer_segments.copy()
    customer_data['Is_Churned'] = (
        customer_data['Recency'] > churn_threshold).astype(int)

    print(f"📊 Churn Analysis:")
    print(f"   Total customers: {len(customer_data)}")
    print(
        f"   Churned customers: {customer_data['Is_Churned'].sum()} ({customer_data['Is_Churned'].mean()*100:.1f}%)")
    print(
        f"   Active customers: {len(customer_data) - customer_data['Is_Churned'].sum()} ({(1-customer_data['Is_Churned'].mean())*100:.1f}%)")

    # Prepare features for modeling
    feature_columns = ['Age', 'CLV', 'Loyalty_Score', 'Purchase_Frequency',
                       'Frequency', 'Monetary', 'Discount_Usage_Rate']

    # Handle missing values
    for col in feature_columns:
        customer_data[col].fillna(customer_data[col].median(), inplace=True)

    X = customer_data[feature_columns]
    y = customer_data['Is_Churned']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    churn_models = {}

    # 1. Logistic Regression
    print("🔍 Training Logistic Regression...")

    lr_param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_param_grid,
                           cv=5, scoring='roc_auc', n_jobs=-1)
    lr_grid.fit(X_train_scaled, y_train)

    lr_best = lr_grid.best_estimator_
    lr_pred = lr_best.predict(X_test_scaled)
    lr_pred_proba = lr_best.predict_proba(X_test_scaled)[:, 1]

    churn_models['Logistic_Regression'] = {
        'model': lr_best,
        'predictions': lr_pred,
        'probabilities': lr_pred_proba,
        'accuracy': lr_best.score(X_test_scaled, y_test),
        'auc_roc': roc_auc_score(y_test, lr_pred_proba)
    }

    # 2. Random Forest
    print("🌲 Training Random Forest...")

    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf_grid = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param_grid,
                                 n_iter=10, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)
    rf_grid.fit(X_train, y_train)

    rf_best = rf_grid.best_estimator_
    rf_pred = rf_best.predict(X_test)
    rf_pred_proba = rf_best.predict_proba(X_test)[:, 1]

    churn_models['Random_Forest'] = {
        'model': rf_best,
        'predictions': rf_pred,
        'probabilities': rf_pred_proba,
        'accuracy': rf_best.score(X_test, y_test),
        'auc_roc': roc_auc_score(y_test, rf_pred_proba),
        'feature_importance': rf_best.feature_importances_
    }

    # 3. XGBoost
    print("⚡ Training XGBoost...")

    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    xgb_grid = RandomizedSearchCV(XGBClassifier(random_state=42), xgb_param_grid,
                                  n_iter=10, cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)
    xgb_grid.fit(X_train, y_train)

    xgb_best = xgb_grid.best_estimator_
    xgb_pred = xgb_best.predict(X_test)
    xgb_pred_proba = xgb_best.predict_proba(X_test)[:, 1]

    churn_models['XGBoost'] = {
        'model': xgb_best,
        'predictions': xgb_pred,
        'probabilities': xgb_pred_proba,
        'accuracy': xgb_best.score(X_test, y_test),
        'auc_roc': roc_auc_score(y_test, xgb_pred_proba),
        'feature_importance': xgb_best.feature_importances_
    }

    # 4. Model Evaluation and Comparison
    print("\n📊 Model Performance Comparison:")

    results_df = pd.DataFrame({
        'Model': list(churn_models.keys()),
        'Accuracy': [churn_models[model]['accuracy'] for model in churn_models.keys()],
        'AUC-ROC': [churn_models[model]['auc_roc'] for model in churn_models.keys()]
    })

    print(results_df.round(4))

    # 5. Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ROC Curves
    for i, (model_name, model_data) in enumerate(churn_models.items()):
        fpr, tpr, _ = roc_curve(y_test, model_data['probabilities'])
        axes[0, 0].plot(
            fpr, tpr, label=f"{model_name} (AUC = {model_data['auc_roc']:.3f})")

    axes[0, 0].plot([0, 1], [0, 1], 'k--')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curves Comparison')
    axes[0, 0].legend()

    # Feature Importance (Random Forest)
    if 'Random_Forest' in churn_models:
        feature_imp = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': churn_models['Random_Forest']['feature_importance']
        }).sort_values('Importance', ascending=True)

        axes[0, 1].barh(feature_imp['Feature'], feature_imp['Importance'])
        axes[0, 1].set_title('Feature Importance (Random Forest)')
        axes[0, 1].set_xlabel('Importance')

    # Confusion Matrix (Best Model)
    best_model_name = max(churn_models.keys(),
                          key=lambda x: churn_models[x]['auc_roc'])
    best_predictions = churn_models[best_model_name]['predictions']

    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
    axes[0, 2].set_title(f'Confusion Matrix ({best_model_name})')
    axes[0, 2].set_xlabel('Predicted')
    axes[0, 2].set_ylabel('Actual')

    # Model Accuracy Comparison
    model_names = list(churn_models.keys())
    accuracies = [churn_models[model]['accuracy'] for model in model_names]
    auc_scores = [churn_models[model]['auc_roc'] for model in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    axes[1, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    axes[1, 0].bar(x + width/2, auc_scores, width, label='AUC-ROC', alpha=0.8)
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Model Performance Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45)
    axes[1, 0].legend()

    # Churn Risk Distribution
    best_probabilities = churn_models[best_model_name]['probabilities']
    axes[1, 1].hist(best_probabilities, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Churn Probability')
    axes[1, 1].set_ylabel('Number of Customers')
    axes[1, 1].set_title('Distribution of Churn Probabilities')

    # Churn by Segment
    customer_data['Churn_Probability'] = np.concatenate([
        churn_models[best_model_name]['model'].predict_proba(
            scaler.transform(customer_data[feature_columns])
        )[:, 1]
    ])

    churn_by_segment = customer_data.groupby('RFM_Segment_Name')[
        'Is_Churned'].mean()
    churn_by_segment.plot(kind='bar', ax=axes[1, 2])
    axes[1, 2].set_title('Churn Rate by Customer Segment')
    axes[1, 2].set_xlabel('Customer Segment')
    axes[1, 2].set_ylabel('Churn Rate')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("analysis_plots/churn_prediction.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Customer Risk Scoring
    print("\n🎯 Customer Risk Analysis:")

    # Categorize customers by churn risk
    customer_data['Risk_Category'] = pd.cut(
        customer_data['Churn_Probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    risk_analysis = customer_data.groupby('Risk_Category').agg({
        'Customer_ID': 'count',
        'CLV': 'mean',
        'Monetary': 'mean',
        'Loyalty_Score': 'mean'
    }).round(2)

    print(risk_analysis)

    self.churn_models = churn_models

    print("✅ Churn prediction analysis completed!")

    return {
        'models': churn_models,
        'customer_risk_data': customer_data,
        'risk_analysis': risk_analysis,
        'best_model': best_model_name,
        'feature_importance': feature_columns
    }


def market_basket_analysis(self):
    """
    Perform Market Basket Analysis to identify cross-selling opportunities.
    """
    print("\n🛒 Starting Market Basket Analysis...")

    if self.processed_data is None:
        raise ValueError(
            "Data not processed. Please run preprocess_data() first.")

    # Create transaction-product matrix
    transaction_data = self.processed_data.groupby(
        ['Transaction_ID', 'Product_ID']).size().reset_index(name='Quantity')

    # Create a binary matrix for association rule mining
    basket = transaction_data.pivot_table(
        index='Transaction_ID',
        columns='Product_ID',
        values='Quantity',
        fill_value=0
    )

    # Convert to binary (bought or not bought)
    basket_binary = (basket > 0).astype(int)

    print(f"📊 Market Basket Analysis Overview:")
    print(f"   Total transactions: {len(basket_binary)}")
    print(f"   Total products: {len(basket_binary.columns)}")
    print(
        f"   Average items per transaction: {basket_binary.sum(axis=1).mean():.2f}")

    # Apply Apriori algorithm
    print("🔍 Finding frequent itemsets...")

    # Find frequent itemsets with minimum support of 1%
    frequent_itemsets = apriori(
        basket_binary, min_support=0.01, use_colnames=True)

    if len(frequent_itemsets) > 0:
        print(f"✅ Found {len(frequent_itemsets)} frequent itemsets")

        # Generate association rules
        print("📋 Generating association rules...")

        rules = association_rules(
            frequent_itemsets, metric="lift", min_threshold=1.1)

        if len(rules) > 0:
            # Sort rules by lift
            rules = rules.sort_values('lift', ascending=False)

            print(f"✅ Generated {len(rules)} association rules")

            # Add product names for better readability
            product_info = self.processed_data[[
                'Product_ID', 'Category', 'Sub_Category', 'Brand']].drop_duplicates()

            # Display top rules
            print("\n🎯 Top 10 Association Rules:")
            top_rules = rules.head(
                10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

            for idx, rule in top_rules.iterrows():
                antecedent = list(rule['antecedents'])[0] if len(
                    rule['antecedents']) == 1 else str(rule['antecedents'])
                consequent = list(rule['consequents'])[0] if len(
                    rule['consequents']) == 1 else str(rule['consequents'])

                print(f"   {antecedent} → {consequent}")
                print(
                    f"   Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")
                print()

            # Visualizations
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # Support vs Confidence
            axes[0, 0].scatter(
                rules['support'], rules['confidence'], alpha=0.6)
            axes[0, 0].set_xlabel('Support')
            axes[0, 0].set_ylabel('Confidence')
            axes[0, 0].set_title('Support vs Confidence')

            # Support vs Lift
            axes[0, 1].scatter(rules['support'], rules['lift'], alpha=0.6)
            axes[0, 1].set_xlabel('Support')
            axes[0, 1].set_ylabel('Lift')
            axes[0, 1].set_title('Support vs Lift')

            # Confidence vs Lift
            axes[1, 0].scatter(rules['confidence'], rules['lift'], alpha=0.6)
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_ylabel('Lift')
            axes[1, 0].set_title('Confidence vs Lift')

            # Top rules by lift
            top_10_rules = rules.head(10)
            rule_labels = [f"Rule {i+1}" for i in range(len(top_10_rules))]
            axes[1, 1].barh(rule_labels, top_10_rules['lift'])
            axes[1, 1].set_xlabel('Lift')
            axes[1, 1].set_title('Top 10 Rules by Lift')

            plt.tight_layout()
            plt.savefig("analysis_plots/market_basket_analysis.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

            self.association_rules = rules

        else:
            print("❌ No association rules found with the current thresholds")
            rules = pd.DataFrame()

    else:
        print("❌ No frequent itemsets found with minimum support of 1%")
        rules = pd.DataFrame()

    # Category-level analysis
    print("\n📊 Category-level Cross-selling Analysis:")

    # Analyze which categories are often bought together
    transaction_categories = self.processed_data.groupby(
        ['Transaction_ID', 'Category']).size().reset_index(name='Count')
    category_basket = transaction_categories.pivot_table(
        index='Transaction_ID',
        columns='Category',
        values='Count',
        fill_value=0
    )
    category_basket_binary = (category_basket > 0).astype(int)

    # Category co-occurrence matrix
    category_cooccurrence = category_basket_binary.T.dot(
        category_basket_binary)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(category_cooccurrence, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Category Co-occurrence Matrix')
    plt.tight_layout()
    plt.savefig("analysis_plots/category_cooccurrence.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Market Basket Analysis completed!")

    return {
        'association_rules': rules,
        'frequent_itemsets': frequent_itemsets if 'frequent_itemsets' in locals() else pd.DataFrame(),
        'category_cooccurrence': category_cooccurrence,
        'basket_stats': {
            'total_transactions': len(basket_binary),
            'total_products': len(basket_binary.columns),
            'avg_items_per_transaction': basket_binary.sum(axis=1).mean()
        }
    }


def product_profitability_analysis(self):
    """
    Comprehensive product analysis including profitability and performance metrics.
    """
    print("\n💰 Starting Product Profitability Analysis...")

    if self.processed_data is None:
        raise ValueError(
            "Data not processed. Please run preprocess_data() first.")

    # Aggregate product-level metrics
    product_analysis = self.processed_data.groupby('Product_ID').agg({
        'Total_Amount': ['sum', 'count', 'mean'],
        'Purchase_Quantity': 'sum',
        'Cost_Price': 'first',
        'Selling_Price': 'first',
        'Category': 'first',
        'Sub_Category': 'first',
        'Brand': 'first',
        'Discount_Amount': 'sum',
        'Profit_Margin': 'first'
    }).round(2)

    # Flatten column names
    product_analysis.columns = [
        '_'.join(col).strip() for col in product_analysis.columns.values]
    product_analysis = product_analysis.reset_index()

    # Calculate additional metrics
    product_analysis['Total_Revenue'] = product_analysis['Total_Amount_sum']
    product_analysis['Total_Units_Sold'] = product_analysis['Purchase_Quantity_sum']
    product_analysis['Avg_Order_Value'] = product_analysis['Total_Amount_mean']
    product_analysis['Transaction_Count'] = product_analysis['Total_Amount_count']

    # Calculate total profit
    product_analysis['Unit_Profit'] = (
        product_analysis['Selling_Price_first'] -
        product_analysis['Cost_Price_first']
    )
    product_analysis['Total_Profit'] = (
        product_analysis['Unit_Profit'] * product_analysis['Total_Units_Sold']
    ) - product_analysis['Discount_Amount_sum']

    # Calculate profitability metrics
    product_analysis['Profit_Margin_Percent'] = (
        product_analysis['Total_Profit'] /
        product_analysis['Total_Revenue'] * 100
    ).round(2)

    product_analysis['ROI'] = (
        product_analysis['Total_Profit'] /
        (product_analysis['Cost_Price_first'] *
         product_analysis['Total_Units_Sold']) * 100
    ).round(2)

    # Performance categorization
    product_analysis['Revenue_Quartile'] = pd.qcut(
        product_analysis['Total_Revenue'],
        q=4,
        labels=['Low', 'Medium-Low', 'Medium-High', 'High']
    )

    product_analysis['Profit_Quartile'] = pd.qcut(
        product_analysis['Total_Profit'],
        q=4,
        labels=['Low', 'Medium-Low', 'Medium-High', 'High']
    )

    # Create product performance matrix
    product_analysis['Performance_Category'] = 'Unknown'

    high_revenue = product_analysis['Revenue_Quartile'].isin(
        ['Medium-High', 'High'])
    high_profit = product_analysis['Profit_Quartile'].isin(
        ['Medium-High', 'High'])

    product_analysis.loc[high_revenue & high_profit,
                         'Performance_Category'] = 'Star Products'
    product_analysis.loc[high_revenue & ~high_profit,
                         'Performance_Category'] = 'Question Marks'
    product_analysis.loc[~high_revenue & high_profit,
                         'Performance_Category'] = 'Cash Cows'
    product_analysis.loc[~high_revenue & ~
                         high_profit, 'Performance_Category'] = 'Dogs'

    print("📊 Product Performance Overview:")
    performance_summary = product_analysis['Performance_Category'].value_counts(
    )
    print(performance_summary)

    # Category-level analysis
    category_analysis = product_analysis.groupby('Category_first').agg({
        'Total_Revenue': 'sum',
        'Total_Profit': 'sum',
        'Total_Units_Sold': 'sum',
        'Product_ID': 'count'
    }).round(2)

    category_analysis['Avg_Profit_Margin'] = (
        category_analysis['Total_Profit'] /
        category_analysis['Total_Revenue'] * 100
    ).round(2)

    category_analysis = category_analysis.sort_values(
        'Total_Revenue', ascending=False)

    print("\n📈 Top 5 Categories by Revenue:")
    print(category_analysis.head())

    # Visualizations
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))

    # Revenue vs Profit scatter plot
    scatter = axes[0, 0].scatter(
        product_analysis['Total_Revenue'],
        product_analysis['Total_Profit'],
        c=product_analysis['Profit_Margin_Percent'],
        cmap='RdYlGn',
        alpha=0.6
    )
    axes[0, 0].set_xlabel('Total Revenue ($)')
    axes[0, 0].set_ylabel('Total Profit ($)')
    axes[0, 0].set_title('Product Revenue vs Profit')
    plt.colorbar(scatter, ax=axes[0, 0], label='Profit Margin %')

    # Performance category distribution
    performance_counts = product_analysis['Performance_Category'].value_counts(
    )
    axes[0, 1].pie(performance_counts.values,
                   labels=performance_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Product Performance Distribution')

    # Top 10 products by revenue
    top_revenue = product_analysis.nlargest(10, 'Total_Revenue')
    axes[1, 0].barh(range(len(top_revenue)), top_revenue['Total_Revenue'])
    axes[1, 0].set_yticks(range(len(top_revenue)))
    axes[1, 0].set_yticklabels(top_revenue['Product_ID'])
    axes[1, 0].set_xlabel('Total Revenue ($)')
    axes[1, 0].set_title('Top 10 Products by Revenue')

    # Top 10 products by profit
    top_profit = product_analysis.nlargest(10, 'Total_Profit')
    axes[1, 1].barh(range(len(top_profit)), top_profit['Total_Profit'])
    axes[1, 1].set_yticks(range(len(top_profit)))
    axes[1, 1].set_yticklabels(top_profit['Product_ID'])
    axes[1, 1].set_xlabel('Total Profit ($)')
    axes[1, 1].set_title('Top 10 Products by Profit')

    # Category revenue comparison
    category_analysis.plot(kind='bar', y='Total_Revenue', ax=axes[2, 0])
    axes[2, 0].set_title('Revenue by Category')
    axes[2, 0].set_xlabel('Category')
    axes[2, 0].set_ylabel('Total Revenue ($)')
    axes[2, 0].tick_params(axis='x', rotation=45)

    # Category profit margin comparison
    category_analysis.plot(
        kind='bar', y='Avg_Profit_Margin', ax=axes[2, 1], color='green')
    axes[2, 1].set_title('Average Profit Margin by Category')
    axes[2, 1].set_xlabel('Category')
    axes[2, 1].set_ylabel('Profit Margin (%)')
    axes[2, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("analysis_plots/product_profitability.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # Recommendations
    print("\n💡 Product Strategy Recommendations:")

    star_products = product_analysis[product_analysis['Performance_Category']
                                     == 'Star Products']
    question_marks = product_analysis[product_analysis['Performance_Category']
                                      == 'Question Marks']
    cash_cows = product_analysis[product_analysis['Performance_Category'] == 'Cash Cows']
    dogs = product_analysis[product_analysis['Performance_Category'] == 'Dogs']

    print(f"\n⭐ Star Products ({len(star_products)} products):")
    print("   - Maintain high inventory levels")
    print("   - Invest in marketing and promotion")
    print("   - Consider premium positioning")

    print(f"\n❓ Question Marks ({len(question_marks)} products):")
    print("   - Analyze cost structure")
    print("   - Consider price optimization")
    print("   - Evaluate market positioning")

    print(f"\n🐄 Cash Cows ({len(cash_cows)} products):")
    print("   - Optimize pricing for higher revenue")
    print("   - Increase marketing investment")
    print("   - Bundle with other products")

    print(f"\n🐕 Dogs ({len(dogs)} products):")
    print("   - Consider discontinuation")
    print("   - Reduce inventory levels")
    print("   - Focus on liquidation strategies")

    print("✅ Product profitability analysis completed!")

    return {
        'product_analysis': product_analysis,
        'category_analysis': category_analysis,
        'performance_summary': performance_summary,
        'recommendations': {
            'star_products': star_products['Product_ID'].tolist(),
            'question_marks': question_marks['Product_ID'].tolist(),
            'cash_cows': cash_cows['Product_ID'].tolist(),
            'dogs': dogs['Product_ID'].tolist()
        }
    }


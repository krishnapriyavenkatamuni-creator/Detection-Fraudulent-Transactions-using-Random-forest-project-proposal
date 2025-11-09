import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')

print("=" * 60)
print("FRAUDULENT TRANSACTION DETECTION USING RANDOM FOREST")
print("=" * 60)

# ============================================================================
# STEP 1: GENERATE SYNTHETIC DATASET (Or load your own dataset)
# ============================================================================
print("\n[STEP 1] Generating Synthetic Transaction Dataset...")

np.random.seed(42)
n_samples = 10000

# Generate features
data = {
    'transaction_amount': np.random.exponential(100, n_samples),
    'transaction_hour': np.random.randint(0, 24, n_samples),
    'day_of_week': np.random.randint(0, 7, n_samples),
    'age': np.random.randint(18, 80, n_samples),
    'num_transactions_last_month': np.random.poisson(15, n_samples),
    'account_age_days': np.random.randint(30, 3650, n_samples),
    'distance_from_home': np.random.exponential(50, n_samples),
    'distance_from_last_transaction': np.random.exponential(30, n_samples),
    'ratio_to_median_purchase': np.random.gamma(2, 0.5, n_samples),
    'used_chip': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
    'used_pin_number': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'online_order': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
}

df = pd.DataFrame(data)

# Generate fraud labels (5% fraud rate)
fraud_probability = (
    0.001 +
    0.02 * (df['transaction_amount'] > 500).astype(int) +
    0.03 * (df['transaction_hour'] > 22).astype(int) +
    0.04 * (df['distance_from_home'] > 100).astype(int) +
    0.02 * (df['online_order'] == 1).astype(int) +
    0.03 * (df['used_chip'] == 0).astype(int)
)

df['is_fraud'] = (np.random.random(n_samples) < fraud_probability).astype(int)

print(f"Dataset created with {len(df)} transactions")
print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].sum()/len(df)*100:.2f}%)")
print(f"Legitimate cases: {(1-df['is_fraud']).sum()} ({(1-df['is_fraud']).sum()/len(df)*100:.2f}%)")

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n[STEP 2] Performing Exploratory Data Analysis...")

print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Fraud distribution
df['is_fraud'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
axes[0, 0].set_title('Distribution of Fraud vs Legitimate Transactions')
axes[0, 0].set_xlabel('Transaction Type (0=Legitimate, 1=Fraud)')
axes[0, 0].set_ylabel('Count')

# 2. Transaction amount distribution
df.boxplot(column='transaction_amount', by='is_fraud', ax=axes[0, 1])
axes[0, 1].set_title('Transaction Amount by Fraud Status')
axes[0, 1].set_xlabel('Is Fraud')
axes[0, 1].set_ylabel('Transaction Amount')

# 3. Correlation heatmap
correlation = df.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0], cbar_kws={'shrink': 0.8})
axes[1, 0].set_title('Feature Correlation Heatmap')

# 4. Fraud by transaction hour
fraud_by_hour = df.groupby('transaction_hour')['is_fraud'].mean()
fraud_by_hour.plot(kind='line', ax=axes[1, 1], marker='o', color='red')
axes[1, 1].set_title('Fraud Rate by Hour of Day')
axes[1, 1].set_xlabel('Hour')
axes[1, 1].set_ylabel('Fraud Rate')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ EDA visualizations saved as 'eda_analysis.png'")

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n[STEP 3] Preprocessing Data...")

# Separate features and target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print(f"Training fraud rate: {y_train.sum()/len(y_train)*100:.2f}%")
print(f"Testing fraud rate: {y_test.sum()/len(y_test)*100:.2f}%")

# Feature scaling (optional for Random Forest, but good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# STEP 4: BUILD AND TRAIN RANDOM FOREST MODEL
# ============================================================================
print("\n[STEP 4] Building Random Forest Model...")

# Create Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',  # Handle imbalanced data
    random_state=42,
    n_jobs=-1
)

print("Training model...")
rf_model.fit(X_train_scaled, y_train)
print("✓ Model training completed!")

# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================
print("\n[STEP 5] Evaluating Model Performance...")

# Make predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE METRICS")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
print("=" * 60)

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ============================================================================
# STEP 6: VISUALIZE RESULTS
# ============================================================================
print("\n[STEP 6] Creating Performance Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# 2. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend(loc="lower right")

# 3. Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance.plot(x='feature', y='importance', kind='barh', ax=axes[1, 0], legend=False)
axes[1, 0].set_title('Feature Importance')
axes[1, 0].set_xlabel('Importance Score')

# 4. Performance Metrics Bar Chart
metrics_data = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'ROC-AUC': roc_auc}
axes[1, 1].bar(metrics_data.keys(), metrics_data.values(), color=['blue', 'green', 'orange', 'red', 'purple'])
axes[1, 1].set_title('Model Performance Metrics')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_ylim([0, 1])
for i, (k, v) in enumerate(metrics_data.items()):
    axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
print("✓ Performance visualizations saved as 'model_performance.png'")

# ============================================================================
# STEP 7: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n[STEP 7] Feature Importance Analysis...")
print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# ============================================================================
# STEP 8: SAVE THE MODEL
# ============================================================================
print("\n[STEP 8] Saving Model...")
import joblib

joblib.dump(rf_model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✓ Model saved as 'fraud_detection_model.pkl'")
print("✓ Scaler saved as 'scaler.pkl'")

# ============================================================================
# STEP 9: TEST WITH SAMPLE PREDICTIONS
# ============================================================================
print("\n[STEP 9] Testing with Sample Predictions...")

sample_transactions = X_test_scaled[:5]
sample_predictions = rf_model.predict(sample_transactions)
sample_probabilities = rf_model.predict_proba(sample_transactions)

print("\nSample Predictions:")
for i in range(len(sample_predictions)):
    print(f"\nTransaction {i+1}:")
    print(f"  Predicted: {'FRAUD' if sample_predictions[i] == 1 else 'LEGITIMATE'}")
    print(f"  Fraud Probability: {sample_probabilities[i][1]*100:.2f}%")
    print(f"  Actual: {'FRAUD' if y_test.iloc[i] == 1 else 'LEGITIMATE'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nGenerated Files:")
print("1. eda_analysis.png - Exploratory Data Analysis visualizations")
print("2. model_performance.png - Model performance visualizations")
print("3. fraud_detection_model.pkl - Trained Random Forest model")
print("4. scaler.pkl - Feature scaler")
print("\nNext Steps:")
print("1. Review the visualizations and metrics")
print("2. Use the web application code to deploy the model")
print("3. Prepare your report with these results")
print("=" * 60)
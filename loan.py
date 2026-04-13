# ============================================
# LOAN DEFAULT RISK PREDICTOR
# Advanced Machine Learning with Professional Visualisations
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

print("="*70)
print("LOAN DEFAULT RISK PREDICTOR - ADVANCED ML MODEL")
print("="*70)

# ============================================
# STEP 1: GENERATE REALISTIC LOAN DATASET
# ============================================

np.random.seed(42)
n_samples = 10000

print("\n📊 Generating synthetic loan dataset...")

# Create realistic features
data = pd.DataFrame({
    # Credit history features
    'credit_score': np.random.normal(680, 50, n_samples).clip(300, 850),
    'num_late_payments': np.random.poisson(2, n_samples),
    'credit_utilization': np.random.beta(2, 5, n_samples) * 100,
    'credit_age_years': np.random.exponential(8, n_samples).clip(0, 40),
    
    # Loan features
    'loan_amount': np.random.lognormal(10, 1, n_samples),
    'interest_rate': np.random.normal(12, 4, n_samples).clip(3, 30),
    'loan_term_months': np.random.choice([12, 24, 36, 48, 60], n_samples),
    
    # Borrower features
    'annual_income': np.random.lognormal(11, 0.6, n_samples),
    'debt_to_income': np.random.beta(2, 8, n_samples) * 100,
    'employment_years': np.random.exponential(5, n_samples).clip(0, 30),
    'num_open_accounts': np.random.poisson(8, n_samples),
    'num_credit_inquiries': np.random.poisson(1, n_samples),
    
    # Demographic (simplified for fairness)
    'homeowner': np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),
})

# Create target variable (default) based on logical rules
# Higher risk = higher chance of default
risk_score = (
    (data['credit_score'] < 620) * 30 +
    (data['credit_score'] < 580) * 20 +
    (data['debt_to_income'] > 40) * 15 +
    (data['num_late_payments'] > 2) * 20 +
    (data['num_late_payments'] > 5) * 15 +
    (data['credit_utilization'] > 50) * 10 +
    (data['loan_amount'] / data['annual_income'] > 0.5) * 10 +
    (data['employment_years'] < 1) * 10
)

# Convert to probability (0-100 scale)
risk_prob = np.clip(risk_score / 100 + np.random.normal(0, 0.1, n_samples), 0, 1)
data['default'] = (risk_prob > 0.5).astype(int)

default_rate = data['default'].mean() * 100
print(f"✓ Dataset created: {n_samples} loans, {default_rate:.1f}% default rate")
print(f"✓ Features: {data.shape[1]-1} variables, 1 target column")

# ============================================
# STEP 2: EXPLORATORY DATA ANALYSIS - PROFESSIONAL PLOTS
# ============================================

print("\n📈 Generating exploratory visualisations...")

# Create a beautiful correlation heatmap
fig, ax = plt.subplots(figsize=(14, 10))
correlation_matrix = data.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='RdBu_r', center=0, square=True, 
            linewidths=0.5, cbar_kws={"shrink": 0.8},
            ax=ax)
ax.set_title('Feature Correlation Matrix\n(Understanding Relationships in Loan Data)', 
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Distribution of defaults by key features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Loan Default Risk Analysis\nKey Feature Distributions', 
             fontsize=16, fontweight='bold')

features = ['credit_score', 'debt_to_income', 'num_late_payments', 
            'credit_utilization', 'annual_income', 'employment_years']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4A6C']

for idx, (feature, color) in enumerate(zip(features, colors)):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    
    # Plot histograms by default status
    default_0 = data[data['default'] == 0][feature]
    default_1 = data[data['default'] == 1][feature]
    
    ax.hist(default_0, bins=40, alpha=0.6, label='No Default', color='green', density=True)
    ax.hist(default_1, bins=40, alpha=0.6, label='Default', color='red', density=True)
    ax.set_xlabel(feature.replace('_', ' ').title())
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# STEP 3: PREPARE DATA FOR ML
# ============================================

print("\n🔧 Preparing data for machine learning...")

# Separate features and target
X = data.drop('default', axis=1)
y = data['default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")
print(f"✓ Features scaled to standard normal distribution")

# ============================================
# STEP 4: TRAIN MULTIPLE MODELS
# ============================================

print("\n🤖 Training multiple ML models...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

results = {}
predictions = {}
probabilities = {}

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob)
    }
    predictions[name] = y_pred
    probabilities[name] = y_prob
    
    print(f"    ✓ AUC: {results[name]['auc']:.4f}")

# ============================================
# STEP 5: PROFESSIONAL MODEL COMPARISON VISUALISATIONS
# ============================================

print("\n📊 Generating professional visualisations...")

# Figure 1: Model Performance Comparison (Radar Chart)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='polar')

metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Close the loop

for name, metrics_dict in results.items():
    values = [metrics_dict[m] for m in metrics]
    values += values[:1]  # Close the loop
    ax.plot(angles, values, 'o-', linewidth=2, label=name)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([m.capitalize() for m in metrics])
ax.set_ylim(0, 1)
ax.set_title('Model Performance Comparison\nRadar Chart', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)
plt.tight_layout()
plt.show()

# Figure 2: ROC Curves Comparison
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#2E86AB', '#A23B72']

for (name, color) in zip(models.keys(), colors):
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    auc = results[name]['auc']
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc:.3f})')

ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('ROC Curves - Model Comparison\nHigher AUC = Better Performance', 
             fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 3: Confusion Matrices Heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Confusion Matrices\nUnderstanding Prediction Errors', fontsize=16, fontweight='bold')

for idx, (name, color) in enumerate(zip(models.keys(), colors)):
    cm = confusion_matrix(y_test, predictions[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'],
                ax=axes[idx], cbar=False)
    axes[idx].set_title(f'{name}\nAccuracy: {results[name]["accuracy"]:.3f}', fontsize=12)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Figure 4: Feature Importance (Random Forest)
fig, ax = plt.subplots(figsize=(12, 8))
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
ax.barh(feature_importance['feature'], feature_importance['importance'], 
        color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_title('Random Forest - Top Predictors of Loan Default\nWhat Factors Matter Most?', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# Figure 5: Distribution of Predictions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Prediction Distribution Analysis', fontsize=16, fontweight='bold')

for idx, (name, color) in enumerate(zip(models.keys(), colors)):
    axes[idx].hist(probabilities[name][y_test == 0], bins=30, alpha=0.6, 
                   label='Actual No Default', color='green', density=True)
    axes[idx].hist(probabilities[name][y_test == 1], bins=30, alpha=0.6, 
                   label='Actual Default', color='red', density=True)
    axes[idx].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    axes[idx].set_xlabel('Predicted Default Probability')
    axes[idx].set_ylabel('Density')
    axes[idx].set_title(f'{name}\nAUC: {results[name]["auc"]:.3f}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# STEP 6: PERFORMANCE SUMMARY TABLE
# ============================================

print("\n" + "="*70)
print("MODEL PERFORMANCE SUMMARY")
print("="*70)

results_df = pd.DataFrame(results).T
print(results_df.round(4).to_string())

# ============================================
# STEP 7: CUSTOM RISK ASSESSMENT
# ============================================

print("\n" + "="*70)
print("CUSTOM RISK ASSESSMENT")
print("="*70)

# Create a sample borrower
sample_borrower = pd.DataFrame({
    'credit_score': [650],
    'num_late_payments': [3],
    'credit_utilization': [65],
    'credit_age_years': [5],
    'loan_amount': [25000],
    'interest_rate': [15],
    'loan_term_months': [36],
    'annual_income': [50000],
    'debt_to_income': [45],
    'employment_years': [2],
    'num_open_accounts': [10],
    'num_credit_inquiries': [3],
    'homeowner': [0]
})

# Scale and predict
sample_scaled = scaler.transform(sample_borrower)
sample_prob = models['Random Forest'].predict_proba(sample_scaled)[0, 1]

print(f"\n📋 Sample Borrower Profile:")
for col in sample_borrower.columns:
    print(f"  {col}: {sample_borrower[col].values[0]}")

print(f"\n🎯 Predicted Default Risk: {sample_prob:.1%}")
if sample_prob > 0.5:
    print(f"  ⚠️ HIGH RISK - Loan should be reviewed carefully")
else:
    print(f"  ✅ LOW RISK - Loan likely to be repaid")

# ============================================
# STEP 8: SAVE MODEL
# ============================================

import joblib
joblib.dump(rf_model, 'loan_default_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print(f"\n💾 Model saved as 'loan_default_model.pkl'")
print(f"💾 Scaler saved as 'scaler.pkl'")

print("\n" + "="*70)
print("✅ SIMULATION COMPLETE!")
print("="*70)
print("\n📊 Generated Visualisations:")
print("  1. Correlation Heatmap")
print("  2. Feature Distributions")
print("  3. Radar Chart (Model Comparison)")
print("  4. ROC Curves")
print("  5. Confusion Matrices")
print("  6. Feature Importance Bar Chart")
print("  7. Prediction Distribution Analysis")
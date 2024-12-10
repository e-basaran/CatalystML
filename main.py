import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('data.csv')

# Print basic information about the dataset
print(df.info())

# Create feature encoders
le_surface = LabelEncoder()
le_ab = LabelEncoder()
le_a = LabelEncoder()
le_b = LabelEncoder()

# Encode categorical variables
df['surface_encoded'] = le_surface.fit_transform(df['surface'])
df['ab_encoded'] = le_ab.fit_transform(df['ab'])
df['a_encoded'] = le_a.fit_transform(df['a'])
df['b_encoded'] = le_b.fit_transform(df['b'])

# Create chemical complexity features
df['has_asterisk_a'] = df['a'].str.contains(r'\*', regex=True).astype(int)
df['has_asterisk_b'] = df['b'].str.contains(r'\*', regex=True).astype(int)
df['has_asterisk_ab'] = df['ab'].str.contains(r'\*', regex=True).astype(int)

# Create length-based features
df['len_a'] = df['a'].str.len()
df['len_b'] = df['b'].str.len()
df['len_ab'] = df['ab'].str.len()

# Advanced chemical features
df['carbon_count_a'] = df['a'].str.count('C')
df['carbon_count_b'] = df['b'].str.count('C')
df['oxygen_count_a'] = df['a'].str.count('O')
df['oxygen_count_b'] = df['b'].str.count('O')
df['hydrogen_count_a'] = df['a'].str.count('H')
df['hydrogen_count_b'] = df['b'].str.count('H')
df['nitrogen_count_a'] = df['a'].str.count('N')
df['nitrogen_count_b'] = df['b'].str.count('N')

# Surface type features
df['surface_metal'] = df['surface'].str.extract(r'([A-Z][a-z]?)')[0]
df['surface_metal_encoded'] = LabelEncoder().fit_transform(df['surface_metal'])
df['surface_facet'] = df['surface'].str.extract(r'\((\d+)\)')[0]
df['surface_facet'] = pd.to_numeric(df['surface_facet'])

# Create interaction features
df['surface_a_interaction'] = df['surface_encoded'] * df['a_encoded']
df['surface_b_interaction'] = df['surface_encoded'] * df['b_encoded']
df['a_b_interaction'] = df['a_encoded'] * df['b_encoded']
df['metal_facet_interaction'] = df['surface_metal_encoded'] * df['surface_facet']

# Molecular complexity scores
df['complexity_a'] = (df['carbon_count_a'] + df['oxygen_count_a'] + 
                     df['nitrogen_count_a'] + df['hydrogen_count_a'])
df['complexity_b'] = (df['carbon_count_b'] + df['oxygen_count_b'] + 
                     df['nitrogen_count_b'] + df['hydrogen_count_b'])

# Fill any NaN values with 0 for numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(0)

# Select features for modeling
feature_columns = [
    'surface_encoded', 'ab_encoded', 'a_encoded', 'b_encoded',
    'has_asterisk_a', 'has_asterisk_b', 'has_asterisk_ab',
    'len_a', 'len_b', 'len_ab',
    'carbon_count_a', 'carbon_count_b',
    'oxygen_count_a', 'oxygen_count_b',
    'hydrogen_count_a', 'hydrogen_count_b',
    'nitrogen_count_a', 'nitrogen_count_b',
    'surface_metal_encoded', 'surface_facet',
    'surface_a_interaction', 'surface_b_interaction',
    'a_b_interaction', 'metal_facet_interaction',
    'complexity_a', 'complexity_b'
]

# Prepare X and y
X = df[feature_columns]
y = df['reaction_energy']

# Split the data
print("\nSplitting the data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Create preprocessing pipeline
numeric_features = [col for col in feature_columns 
                   if col not in ['surface_encoded', 'ab_encoded', 'a_encoded', 'b_encoded',
                                'has_asterisk_a', 'has_asterisk_b', 'has_asterisk_ab']]
categorical_features = ['surface_encoded', 'ab_encoded', 'a_encoded', 'b_encoded',
                       'has_asterisk_a', 'has_asterisk_b', 'has_asterisk_ab']

# Create polynomial features for numeric columns
poly = PolynomialFeatures(degree=2, include_bias=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('poly', poly)
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=-1))
        ]), categorical_features)
    ])

# Fit the preprocessor on the training data to ensure the PolynomialFeatures is fitted
preprocessor.fit(X_train)

# Get feature names after polynomial transformation from the fitted pipeline
numeric_features_poly = preprocessor.named_transformers_['num'].named_steps['poly'].get_feature_names_out(numeric_features)
all_features = list(numeric_features_poly) + categorical_features

# Create model pipelines
def create_model_pipeline(model):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

# Initialize models with better hyperparameters
base_models = {
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features='sqrt',
        learning_rate=0.08,
        random_state=42
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42
    )
}

models = {name: create_model_pipeline(model) for name, model in base_models.items()}

# Perform cross-validation
print("\nCross-Validation Results (R² scores):")
cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_results[name] = cv_scores
    print(f"{name}:")
    print(f"Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Find and train best model
best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k].mean())
print(f"\nBest model: {best_model_name}")

# Ensure best_model is defined
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# Check if the model has feature_importances_ attribute
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    # Get feature names after polynomial transformation
    numeric_features_poly = poly.get_feature_names_out(numeric_features)
    all_features = list(numeric_features_poly) + categorical_features
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': best_model.named_steps['model'].feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15))
else:
    print("The selected model does not support feature importances.")

# Make predictions
train_pred = best_model.predict(X_train)
val_pred = best_model.predict(X_val)
test_pred = best_model.predict(X_test)

# Calculate comprehensive metrics
def calculate_metrics(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Calculate custom metrics for reaction energy ranges
    errors = np.abs(y_true - y_pred)
    within_05 = np.mean(errors < 0.5) * 100
    within_1 = np.mean(errors < 1.0) * 100
    
    print(f"\n{dataset_name} Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"Predictions within 0.5 eV: {within_05:.1f}%")
    print(f"Predictions within 1.0 eV: {within_1:.1f}%")
    
    return errors, y_true, y_pred

# Print metrics and analyze errors
train_errors, train_true, train_pred = calculate_metrics(y_train, train_pred, "Training")
val_errors, val_true, val_pred = calculate_metrics(y_val, val_pred, "Validation")
test_errors, test_true, test_pred = calculate_metrics(y_test, test_pred, "Test")

# Analyze error patterns
print("\nError Analysis:")
# Create a DataFrame with true values, predictions, and errors
error_analysis = pd.DataFrame({
    'True_Energy': y_test,
    'Predicted_Energy': test_pred,
    'Absolute_Error': np.abs(y_test - test_pred),
    'Surface': df.loc[y_test.index, 'surface'],
    'Reactant_A': df.loc[y_test.index, 'a'],
    'Reactant_B': df.loc[y_test.index, 'b'],
    'Product': df.loc[y_test.index, 'ab']
})

# Find worst predictions
print("\nWorst 10 Predictions:")
worst_predictions = error_analysis.nlargest(10, 'Absolute_Error')
print(worst_predictions[['Surface', 'Reactant_A', 'Reactant_B', 'Product', 
                        'True_Energy', 'Predicted_Energy', 'Absolute_Error']])

# Analyze performance by reaction type
print("\nPerformance by Surface Type:")
surface_performance = error_analysis.groupby('Surface')['Absolute_Error'].agg(
    ['mean', 'std', 'count']).sort_values('mean', ascending=False)
print(surface_performance.head())


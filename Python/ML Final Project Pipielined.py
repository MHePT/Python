import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mutual_info_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectPercentile as SP
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_explore_data():
    cancer = load_breast_cancer()
    df = pd.DataFrame(
        data=cancer.data, 
        columns=cancer.feature_names
    )
    df['target'] = cancer.target
    
    print(f"""
    Dataset Shape: {df.shape}
    Feature Names: {cancer['feature_names']}
    Target Names: {cancer['target_names']}
    Missing Values:\n{df.isnull().sum()}
    Data Types:\n{df.dtypes}
    """)
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
    
    return df


def preprocess_data_pipeline(df):
    # Create preprocessing pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier

    # Identify numeric and categorical columns
    numeric_features = pd.DataFrame(df).select_dtypes(exclude=['object']).columns
    categorical_features = pd.DataFrame(df).select_dtypes(include=['object']).columns

    # Create preprocessing steps for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Create preprocessing steps for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                     ('classifier', RandomForestClassifier(random_state=42))])
    
    return clf


def perform_grid_search(model, X, y):
    param_grid = {
        'classifier__n_estimators': [202,203,204],  # Reduced parameter space
        'classifier__max_depth': [6, 7],
        'classifier__min_samples_split': [2, 3]
    }
    
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1  # Utilize all available cores
    )
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def evaluate_models(y_true, y_pred):
    
    print(f"Mutual Information Score: {mutual_info_score(y_true, y_pred):.4f}")

    print(classification_report(y_true, y_pred))

    print(confusion_matrix(y_true,y_pred))  
    

df = load_and_explore_data()
        
X = df.iloc[:, :-1].to_numpy()
Target = df.iloc[:, -1].values

#Reduce Attributes by 50%
selector = SP(percentile=50)
selector.fit(X,Target)
X = selector.transform(X)
 
X_temp, X_test, y_temp, y_test = train_test_split(
    X, Target, 
    test_size=0.2,
    random_state=42
)
    
X_train, X_val, Target_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42
)
    
print("Data split sizes:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

Model = preprocess_data_pipeline(X)    
best_model = perform_grid_search(Model, X_train, Target_train)

print("\nValidation Set Performance:")
Target_val_pred = best_model.predict(X_val)
evaluate_models(y_val, Target_val_pred)
    
print("\nTest Set Performance:")
Target_test_pred = best_model.predict(X_test)
evaluate_models(y_test, Target_test_pred)

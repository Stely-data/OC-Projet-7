from feature_pipeline import FeatureEngineeringPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import pandas as pd
import numpy as np
import re
from feature_pipeline import FeatureEngineeringPipeline

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier, threshold=0.5):
        self.base_classifier = base_classifier
        self.threshold = threshold

    def fit(self, df, y=None):
        X = df.drop(columns=['TARGET'])
        y = df['TARGET']
        self.base_classifier.fit(X, y)
        return self

    def predict(self, X):
        proba = self.base_classifier.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.base_classifier.predict_proba(X)

class FeatureEngineeringPipelineWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, feature_engineering_pipeline, selected_features):
        self.feature_engineering_pipeline = feature_engineering_pipeline
        self.selected_features = selected_features

    def fit(self, df, y=None):
        df_transformed = self.feature_engineering_pipeline.fit(df)
        df_transformed = clean_column_names(df_transformed)
        self.feature_engineering_pipeline.save("data/Cleaned/param")
        return df_transformed[['TARGET'] + self.selected_features]

    def transform(self, X):
        self.feature_engineering_pipeline.load("data/Cleaned/param")
        X_transformed = self.feature_engineering_pipeline.transform(X)
        X_transformed = clean_column_names(X_transformed)
        if 'TARGET' in X_transformed.columns:
            X_transformed = X_transformed.drop(columns=['TARGET'])
        return X_transformed[self.selected_features]

class PipelineWithDriftDetection:
    def __init__(self, pipeline, reference_data):
        self.pipeline = pipeline
        self.reference_data = reference_data

    def fit(self, df):
        self.pipeline.fit(df)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def check_data_drift(self, current_data):
        if 'TARGET' not in current_data.columns:
            ref_data = self.reference_data.drop(columns=['TARGET'])
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_data, current_data=current_data)
        report.save_html("data_drift_report.html")

# Ajoutez également les fonctions auxiliaires nécessaires ici, par exemple:
def clean_column_names(df):
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns]
    return df

def replace_infinite_values(df):
    return df.replace([np.inf, -np.inf], np.nan)

def prepare_pip_data(df):
    # Gestion des valeurs infinies
    df = replace_infinite_values(df)

    # Initialisation du pipeline de feature engineering
    feature_engineering_pipeline = FeatureEngineeringPipeline()
    selected_features = pd.read_csv('data/Featured/features.csv')['Selected Features'].tolist()
    feature_engineering_pipeline_wrapper = FeatureEngineeringPipelineWrapper(feature_engineering_pipeline,
                                                                             selected_features)

    if 'TARGET' in df.columns:
        print("fit...")
        df_transformed = feature_engineering_pipeline_wrapper.fit(df)
    else:
        df['TARGET'] = np.nan
        print("predict...")
        df_transformed = feature_engineering_pipeline_wrapper.transform(df)

    return df_transformed

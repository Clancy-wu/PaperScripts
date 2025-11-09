import pandas as pd
import numpy as np
import logging
import sys
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import optuna

def objective(trial):
    df = pd.read_csv('machine_WithClinic_prepare.csv') # 'health': 2, 'patient': 1
    X, y = df.iloc[:, 0:103], df.iloc[:, 103]
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    pca = PCA(n_components=0.95)
    X_scaled_pca = pca.fit_transform(X_scaled)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_pca, y, test_size=0.3, random_state=42, stratify=y)

    #n_estimators; criterion; max_depth; max_features
    n_estimators = trial.suggest_int("n_estimators", 10, 500)
    max_depth = trial.suggest_int("max_depth", 1, 100)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)


    model = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', max_depth=max_depth, max_features='sqrt',
                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                   random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='balanced_accuracy', n_jobs=5).mean()

    trial.set_user_attr("score", score)
    trial.set_user_attr("model", model)

    return score


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
sampler = optuna.samplers.TPESampler(seed=42)  # Make the sampler behave in a deterministic way.

study = optuna.create_study(direction="maximize", storage="sqlite:///RandomForestPCA.db", study_name='RandomForest_pca95', load_if_exists=True, sampler=sampler)
study.set_user_attr("Contributors", 'Kang Wu')
study.set_user_attr("Time", 'July 22 2025')
study.set_user_attr("Dataset", "Chronic fatigue syndrome patients and healthy controls")
study.optimize(objective, n_trials=100000)

from optuna.visualization.matplotlib import plot_optimization_history
import matplotlib.pyplot as plt
fig = plot_optimization_history(study)
plt.show() 

import joblib  # or use pickle
# Get the model stored in user attributes
best_model = study.best_trial.user_attrs["model"]
joblib.dump(best_model, "best_model.pkl")
###########################################################################


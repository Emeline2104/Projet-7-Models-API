from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

# Partie non agrégée 
initial_data = pd.read_csv("Data/raw/application_test.csv")
current_data = pd.read_csv("Data/raw/application_train.csv")

data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=initial_data, current_data=current_data)
data_drift_report.save_html("data_drift_report.html")

print("Ah")

# Partie agrégée 
# Chargement des données
all_data_aggreg = pd.read_csv("Data/cleaned/data_agregg.csv")

# Séparation des données initiales et actuelles
initial_data_aggreg = all_data_aggreg[all_data_aggreg['TARGET'].notna()]
current_data_aggreg = all_data_aggreg[all_data_aggreg['TARGET'].isna()]

data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=initial_data_aggreg, current_data=current_data_aggreg)
data_drift_report.save_html("data_drift_report_aggreg.html")

print("B")

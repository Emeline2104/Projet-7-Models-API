import sys
sys.path.append("/Users/beatricetapin/Documents/2023/Data Science/Projet_7_Modele_API/")
from config import INITIAL_DATA_FILENAME, CURRENT_DATA_FILENAME
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

# Partie non agrégée 
initial_data = pd.read_csv(INITIAL_DATA_FILENAME)
current_data = pd.read_csv(CURRENT_DATA_FILENAME)

data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=initial_data, current_data=current_data)
data_drift_report.save_html("data_drift_report.html")

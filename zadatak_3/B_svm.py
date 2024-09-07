from A_prep import *
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix







class Models(DataOverview):

    def __init__(self, csv_data, mode="validation"):
        super().__init__(csv_data)



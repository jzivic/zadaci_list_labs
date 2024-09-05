import csv
import pandas as pd
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix





project_path = os.getcwd()
csv_input = os.path.join(project_path, "data/training_data.csv")    # set this




class DataOverview:
    def __init__(self, csv_data):

        df_data, self.features = self.read_csv_to_list(csv_data)
        # self.df_all_data = df_data.apply(pd.to_numeric, errors='coerce')  # transform  str -> float

        # self.check_missing_data() # all ok
        # self.check_data_distribution(save_diagrams=False)
        # self.make_sets()



    def read_csv_to_list(self, csv_input, save_to_picke=True):

        print(project_path)

        with open(csv_input, newline='') as csvfile:
            reader = csv.reader(csvfile)
            features = next(reader)
            rows = list(reader)
        all_data_dict = {feature: [row[i] for row in rows] for i, feature in enumerate(features)}
        all_data_df = pd.DataFrame(all_data_dict)

        if save_to_picke is True:
            all_data_df.to_pickle("all_data.pickle")

        return pd.DataFrame(all_data_dict), features


    def check_missing_data(self):
        if self.df_all_data.isnull().values.any():
            print("DataFrame has missing values.")
        else:
            print("DataFrame has no missing values.")


    def check_data_distribution(self):

        data_dist_dir = os.path.join(project_path, "zadatak_1/data_distribution")

        if os.path.exists(data_dist_dir):
            shutil.rmtree(data_dist_dir)
        os.makedirs(data_dist_dir)  # Creates a new directory

        for feature in self.features:
            plt.figure()
            sns.histplot(self.df_all_data[feature], kde=True if feature != "Class" else False)
            plt.title(f'Distribution of {feature}')
            file_path = os.path.join(data_dist_dir, f'{feature}_distribution.png')
            plt.savefig(file_path)
            plt.close()


    def make_sets(self):

        y_data = self.df_all_data["Class"]
        X_data = self.df_all_data.drop(columns=["Class"])

        X_train, X_rest, y_train, y_rest = train_test_split(X_data, y_data, random_state=True, test_size=0.3)
        X_test, X_validation, y_test, y_validation = train_test_split(X_rest, y_rest, random_state=True, test_size=0.5)

        return X_train, X_test, X_validation, y_train, y_test, y_validation









if __name__ == "__main__":

    print(project_path)


    # DataOverview(csv_data=csv_input)

    # all_data_df.to_pickle("all_data.pickle")

    # df_class_1 = df[df['Class'] == 1]  # DataFrame with Class 1
    # df_class_2 = df[df['Class'] == 2]  # DataFrame with Class 2









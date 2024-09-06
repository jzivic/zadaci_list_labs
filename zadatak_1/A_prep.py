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
csv_input = os.path.abspath(os.path.join(project_path, "../data/training_data.csv"))





class DataOverview:
    def __init__(self, csv_data):

        df_data, self.features = self.read_csv_to_list(csv_data)
        df_shuffled = df_data.sample(frac=1, random_state=None).reset_index(drop=True)
        self.df_all_data = df_shuffled.apply(pd.to_numeric, errors='coerce')  # transform  str -> float

        # self.check_missing_data()
        # self.check_data_distribution(save_diagrams=True)

        (self.X_train, self.X_test, self.X_validation,
         self.y_train, self.y_test, self.y_validation) = self.make_data_sets()



    def read_csv_to_list(self, csv_input, save_to_pickle=True):
        with open(csv_input, newline='') as csvfile:
            reader = csv.reader(csvfile)
            features = next(reader)
            rows = list(reader)
        all_data_dict = {feature: [row[i] for row in rows] for i, feature in enumerate(features)}
        all_data_df = pd.DataFrame(all_data_dict)

        if save_to_pickle is True:
            all_data_df.to_pickle("all_data.pickle")

        return pd.DataFrame(all_data_dict), features


    def check_missing_data(self):
        if self.df_all_data.isnull().values.any():
            print("DataFrame has missing values.")
        else:
            print("DataFrame has no missing values.")


    def check_data_distribution(self, save_diagrams=False):
        data_dist_dir = os.path.abspath(os.path.join(project_path, "../zadatak_1/output/data_distribution"))

        if os.path.exists(data_dist_dir):
            shutil.rmtree(data_dist_dir)
        os.makedirs(data_dist_dir)  # Creates a new directory

        if save_diagrams is True:
            for feature in self.features:
                plt.figure()
                sns.histplot(self.df_all_data[feature], kde=True if feature != "Class" else False)
                plt.title(f'Distribution of {feature}')
                distribution_dir_path = os.path.join(data_dist_dir, f'{feature}_distribution.png')

                plt.savefig(distribution_dir_path)
                plt.close()


    def make_data_sets(self):
        y_data = self.df_all_data["Class"]
        X_data = self.df_all_data.drop(columns=["Class"])

        X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data,
                    test_size=0.3, random_state=42, stratify=y_data)
        X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp,
                  test_size=0.5, random_state=42, stratify=y_temp)

        return X_train, X_test, X_validation, y_train, y_test, y_validation









if __name__ == "__main__":
    DataOverview(csv_data=csv_input)









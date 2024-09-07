import csv
import pandas as pd
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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
        data_dist_dir = os.path.abspath(os.path.join(project_path, "../zadatak_3/output/data_distribution"))

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


    def set_model_acc_dir(self, mode):
        if mode == "validation":
            self.model_acc_dir = os.path.abspath(os.path.join(project_path, "../zadatak_3/output/model_acc_valid_1/"))
        elif mode == "test":
            self.model_acc_dir = os.path.abspath(os.path.join(project_path, "../zadatak_3/output/model_acc_test_1/"))
        os.makedirs(self.model_acc_dir, exist_ok=True)



    # for post processing
    def draw_calc_matrix(self, y_true, y_pred, model_name):

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # print(f"Model: {model_name}")
        # print(f"accuracy: {accuracy}")
        # print(f"precision: {precision}")
        # print(f"recall: {recall}")
        # print(f"f1: {f1}")
        # print(f"cm: {cm}")
        # print(f"cm_normalized: {cm_normalized}")

        class_labels = ['Class 0', 'Class 1', 'Class 2']  # or use the actual labels of your classes
        plt.ylabel('Prediction', fontsize=12)
        plt.xlabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=16)
        plt.title(f"Confusion Matrix: {model_name}", fontsize=16)

        file_path = os.path.join(self.model_acc_dir, f"CM_{model_name}.png")


        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlGnBu', xticklabels=class_labels,
                yticklabels=class_labels)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.show()

        return accuracy, precision, recall, f1, cm, cm_normalized







if __name__ == "__main__":
    DataOverview(csv_data=csv_input)









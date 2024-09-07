import csv
import pandas as pd
import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


project_path = os.getcwd()

# data file
csv_input = os.path.abspath(os.path.join(project_path, "../data/training_data.csv"))

#  csv file to store the models comparisson
csv_compare_models = os.path.abspath(os.path.join(project_path, "../zadatak_3/output/model_compaison.csv"))


class DataOverview:
    def __init__(self, csv_data):


        # store data on the accuracy of all models
        self.models_data = []

        df_data, self.features = self.read_csv_to_list(csv_data)

        # shuffle data just to be sure
        df_shuffled = df_data.sample(frac=1, random_state=None).reset_index(drop=True)
        self.df_all_data = df_shuffled.apply(pd.to_numeric, errors='coerce')  # transform  str -> float

        # self.check_missing_data()
        # self.plot_data_distribution()

        (self.X_train, self.X_test, self.X_validation,
         self.y_train, self.y_test, self.y_validation) = self.make_data_sets()




    # data input to feed the data from csv file
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


    # check if some data is missing and number of each class
    def check_missing_data(self):
        if self.df_all_data.isnull().values.any():
            print("DataFrame has missing values.")
        else:
            print("DataFrame has no missing values.")

        n_c = lambda c: len(self.df_all_data[self.df_all_data['Class'] == c ])
        n_class_0, n_class_1, n_class_2  = n_c(0), n_c(1), n_c(2)
        print(n_class_0, n_class_1, n_class_2)


    # plot histograms for every feature
    def plot_data_distribution(self):
        data_dist_dir = os.path.abspath(os.path.join(project_path, "../zadatak_3/output/data_distribution"))

        if os.path.exists(data_dist_dir):
            shutil.rmtree(data_dist_dir)
        os.makedirs(data_dist_dir)

        for feature in self.features:
            plt.figure()
            sns.histplot(self.df_all_data[feature], kde=True if feature != "Class" else False)
            plt.title(f'Distribution of {feature}')
            distribution_dir_path = os.path.join(data_dist_dir, f'{feature}_distribution.png')

            plt.savefig(distribution_dir_path)
            plt.close()


    # separate data into train, validation and test data
    def make_data_sets(self):
        y_data = self.df_all_data["Class"]
        X_data = self.df_all_data.drop(columns=["Class"])
        X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data,
                    test_size=0.3, random_state=42, stratify=y_data)
        X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp,
                  test_size=0.5, random_state=42, stratify=y_temp)

        return X_train, X_test, X_validation, y_train, y_test, y_validation


    # function to create two separate directories, depending on whether it is for the validation or test data set
    def set_model_acc_dir(self, mode):
        if mode == "validation":
            self.model_acc_dir = os.path.abspath(os.path.join(project_path, "../zadatak_3/output/model_acc_valid/"))
        elif mode == "test":
            self.model_acc_dir = os.path.abspath(os.path.join(project_path, "../zadatak_3/output/model_acc_test/"))
        os.makedirs(self.model_acc_dir, exist_ok=True)


    # for post-processing, to compare accuracy, plot confusion matrix
    def draw_calc_matrix(self, y_true, y_pred, model_name):
        accuracy = round(accuracy_score(y_true, y_pred), 4)
        precision = round(precision_score(y_true, y_pred, average='weighted'), 4)
        recall = round(recall_score(y_true, y_pred, average='weighted'), 4)
        f1 = round(f1_score(y_true, y_pred, average='weighted'), 4)
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        class_labels = ['Class 0', 'Class 1', 'Class 2']  # or use the actual labels of your classes
        plt.ylabel('Prediction', fontsize=12)
        plt.xlabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=16)
        plt.title(f"Confusion Matrix: {model_name}", fontsize=16)

        file_path = os.path.join(self.model_acc_dir, f"CM_{model_name}.png")
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlGnBu', xticklabels=class_labels,
                yticklabels=class_labels)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

        # This has to be turned on for ploting diagrams:
        plt.show()

        model_metrics = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Confusion Matrix': cm.tolist()  # Save confusion matrix as a list for Excel
        }
        self.models_data.append(model_metrics)

        return model_metrics


    def save_results_to_excel(self, file_name="output/model_metrics.xlsx"):
        df = pd.DataFrame(self.models_data)
        df.to_excel(file_name, index=False)
        # print(f"Results saved to {file_name}")

    # calculate time for training every model. Plotting matrices impacts the time, so turn off when measuring time
    @staticmethod
    def calc_timing(arg):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)  # Call the actual function
                elapsed_time = round(time.time() - start_time, 2)
                print(f"{arg.upper()} time: {elapsed_time} seconds")
                return result
            return wrapper
        return decorator






if __name__ == "__main__":
    DataOverview(csv_data=csv_input)









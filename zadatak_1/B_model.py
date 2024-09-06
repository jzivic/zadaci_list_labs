from A_prep import *

import openpyxl
from sklearn.svm import SVC






gamma_range = range(2)
C_range = range(2)

C_f = lambda n: 2 ** n
gamma_f = lambda n: 2 ** n


xlsx_name = "output/Acc_SVM.xlsx"  # xlsx file path for saving results
Acc_grid_search = pd.ExcelWriter(xlsx_name)  # creates xlsx file








class Model_1(DataOverview):
    def __init__(self, csv_data):
        super().__init__(csv_data)

        self.decision_tree_f()
        # self.random_forest_f()
        # self.gaussian_NB_f()





    def draw_calc_matrix(self, y_pred, ratio=True):
        cm = confusion_matrix(self.y_validation, y_pred)

        tp = np.diag(cm)
        fn = cm.sum(axis=1) - tp
        fp = cm.sum(axis=0) - tp
        tn = cm.sum() - (tp + fn + fp)

        tpr = np.round((tp / (tp + fn + np.finfo(float).eps)) * 100, 2)  # True Positive Rate
        fpr = np.round((fp / (fp + tn + np.finfo(float).eps)) * 100, 2)  # False Positive Rate
        fnr = np.round((fn / (tp + fn + np.finfo(float).eps)) * 100, 2)  # False Negative Rate
        tnr = np.round((tn / (fp + tn + np.finfo(float).eps)) * 100, 2)  # True Negative Rate


        class_labels = ['Class 0', 'Class 1', 'Class 2']  # or use the actual labels of your classes
        plt.ylabel('Prediction', fontsize=12)
        plt.xlabel('Real', fontsize=12)
        plt.title('Confusion Matrix', fontsize=16)

        if ratio is True:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlGnBu', xticklabels=class_labels,
                    yticklabels=class_labels)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_labels, yticklabels=class_labels)
        plt.show()


    def decision_tree_f(self, draw_matrix=True):
        decision_tree_model = DecisionTreeClassifier(random_state=24)
        decision_tree_model.fit(self.X_train, self.y_train)

        y_pred = decision_tree_model.predict(self.X_validation)
        acc_train = round(decision_tree_model.score(self.X_train, self.y_train) * 100, 2)
        acc_valid = round(accuracy_score(self.y_validation, y_pred) * 100, 2)
        if draw_matrix is True:
            self.draw_calc_matrix(y_pred)
        return y_pred





if __name__ == "__main__":

    Model_1(csv_data=csv_input)











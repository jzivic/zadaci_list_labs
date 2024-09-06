from A_prep import *
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix





class Model_2(DataOverview):

    def __init__(self, csv_data):
        super().__init__(csv_data)

        self.model_dt = self.decision_tree_f()
        # self.model_rf = self.random_forest_f()
        # self.model_gnb = self.gaussian_NB_f()



    def draw_calc_matrix(self, y_true, y_pred, ratio=True):

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print(accuracy)
        # print(precision)
        # print(recall)
        # print(f1)
        # print(cm)
        # print(cm_normalized)
        class_labels = ['Class 0', 'Class 1', 'Class 2']  # or use the actual labels of your classes
        plt.ylabel('Prediction', fontsize=12)
        plt.xlabel('True', fontsize=12)
        plt.title('Confusion Matrix', fontsize=16)

        if ratio is True:
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlGnBu', xticklabels=class_labels,
                    yticklabels=class_labels)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_labels, yticklabels=class_labels)
        plt.show()

        return accuracy, precision, recall, f1, cm, cm_normalized



    def decision_tree_f(self, draw_matrix=True):
        decision_tree_model = DecisionTreeClassifier(random_state=24)
        decision_tree_model.fit(self.X_train, self.y_train)
        y_pred = decision_tree_model.predict(self.X_validation)

        if draw_matrix is True:
            self.draw_calc_matrix(self.y_validation, y_pred)

        return decision_tree_model


    def random_forest_f(self, draw_matrix=False):
        random_forest_model = RandomForestClassifier(n_estimators=100)
        random_forest_model.fit(self.X_train, self.y_train)
        y_pred = random_forest_model.predict(self.X_validation)
        # acc_train = round(random_forest_model.score(self.X_train, self.y_train) * 100, 2)
        # acc_valid = round(accuracy_score(self.y_validation, y_pred) * 100, 2)
        if draw_matrix is True:
            self.draw_calc_matrix(self.y_validation, y_pred)

        return random_forest_model


    def gaussian_NB_f(self, draw_matrix=False):
        gaussian_model = GaussianNB()
        gaussian_model.fit(self.X_train, self.y_train)
        y_pred = gaussian_model.predict(self.X_validation)
        # acc_train = round(gaussian_model.score(X_train, y_train) * 100, 2)
        # acc_valid = round(accuracy_score(y_validation, y_pred) * 100, 2)
        if draw_matrix is True:
            self.draw_calc_matrix(self.y_validation, y_pred)

        return gaussian_model









if __name__ == "__main__":

    Model_2(csv_data=csv_input)











from A_prep import *
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix




model_acc_dir = os.path.abspath(os.path.join(project_path, "../zadatak_1/output/model_acc_valid_1/"))
os.makedirs(model_acc_dir, exist_ok=True)




class Models(DataOverview):

    def __init__(self, csv_data):
        super().__init__(csv_data)

        self.model_dt = self.decision_tree_f()
        self.model_rf = self.random_forest_f()
        self.model_gnb = self.gaussian_NB_f()



    def draw_calc_matrix(self, y_true, y_pred, model_name, path):

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

        file_path = os.path.join(model_acc_dir, f"CM_{model_name}.png")

        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlGnBu', xticklabels=class_labels,
                yticklabels=class_labels)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.show()

        return accuracy, precision, recall, f1, cm, cm_normalized



    def decision_tree_f(self, draw_matrix=True):
        decision_tree_model = DecisionTreeClassifier(random_state=24)
        decision_tree_model.fit(self.X_train, self.y_train)
        y_pred = decision_tree_model.predict(self.X_validation)

        if draw_matrix is True:
            self.draw_calc_matrix(self.y_validation, y_pred, "dec_tree")

        return decision_tree_model


    def random_forest_f(self, draw_matrix=True):
        random_forest_model = RandomForestClassifier(n_estimators=100)
        random_forest_model.fit(self.X_train, self.y_train)
        y_pred = random_forest_model.predict(self.X_validation)

        if draw_matrix is True:
            self.draw_calc_matrix(self.y_validation, y_pred, "rand_forest")

        return random_forest_model


    def gaussian_NB_f(self, draw_matrix=True):
        gaussian_model = GaussianNB()
        gaussian_model.fit(self.X_train, self.y_train)
        y_pred = gaussian_model.predict(self.X_validation)

        if draw_matrix is True:
            self.draw_calc_matrix(self.y_validation, y_pred, "gaussian_NB")

        return gaussian_model









if __name__ == "__main__":
    Models(csv_data=csv_input)











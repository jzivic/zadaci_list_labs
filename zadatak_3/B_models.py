from A_prep import *
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB




class Models(DataOverview):

    def __init__(self, csv_data, mode="validation"):
        super().__init__(csv_data)

        self.set_model_acc_dir(mode)
        self.model_dt = self.decision_tree_f()
        # self.model_rf = self.random_forest_f()
        # self.model_gnb = self.gaussian_NB_f()


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











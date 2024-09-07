from A_prep import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier



class Models(DataOverview):

    def __init__(self, csv_data, mode="validation"):
        super().__init__(csv_data)
        self.set_model_acc_dir(mode)
        self.model_dt = self.decision_tree_f()
        self.model_rf = self.random_forest_f()
        self.model_gnb = self.gaussian_NB_f()
        self.model_svm = self.svm_f(c=1e4)
        self.model_knn = self.knn_f(n_neighbors=5)
        self.save_results_to_excel()



        # self.save_results_to_excel("zadatak_3/output/model_metrics.xlsx")



    @DataOverview.calc_timing("dc")
    def decision_tree_f(self, draw_matrix=True):
        decision_tree_model = DecisionTreeClassifier(random_state=24)
        decision_tree_model.fit(self.X_train, self.y_train)
        y_pred = decision_tree_model.predict(self.X_validation)

        if draw_matrix is True:
            self.draw_calc_matrix(self.y_validation, y_pred, "dec_tree")

        return decision_tree_model

    @DataOverview.calc_timing("rf")
    def random_forest_f(self, draw_matrix=True):
        random_forest_model = RandomForestClassifier(n_estimators=100)
        random_forest_model.fit(self.X_train, self.y_train)
        y_pred = random_forest_model.predict(self.X_validation)

        if draw_matrix is True:
           self.draw_calc_matrix(self.y_validation, y_pred, "rand_forest")

        return random_forest_model

    @DataOverview.calc_timing("gnb")
    def gaussian_NB_f(self, draw_matrix=True):
        gaussian_model = GaussianNB()
        gaussian_model.fit(self.X_train, self.y_train)
        y_pred = gaussian_model.predict(self.X_validation)

        if draw_matrix is True:
            self.draw_calc_matrix(self.y_validation, y_pred, "gaussian_NB")

        return gaussian_model

    @DataOverview.calc_timing("svm")
    def svm_f(self, c, gamma="scale", draw_matrix=True):
        svm_model = SVC(C=c, kernel="rbf", gamma=gamma)
        svm_model.fit(self.X_train, self.y_train)
        y_pred = svm_model.predict(self.X_validation)
        if draw_matrix is True:
            self.draw_calc_matrix(self.y_validation, y_pred, "svm")

        return svm_model

    @DataOverview.calc_timing("knn")
    def knn_f(self, n_neighbors, draw_matrix=True):
        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_model.fit(self.X_train, self.y_train)
        y_pred = knn_model.predict(self.X_validation)

        if draw_matrix is True:
            self.draw_calc_matrix(self.y_validation, y_pred, "knn")

        return knn_model










if __name__ == "__main__":
    Models(csv_data=csv_input)












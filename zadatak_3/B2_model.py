import time
from A_prep import *
from sklearn.svm import SVC



gamma_range = range(10, 11)
C_range = range(10, 11)
C_f = lambda n: 2 ** n
gamma_f = lambda n: 2 ** n





class Model_2(DataOverview):

    def __init__(self, csv_data, mode="validation"):
        super().__init__(csv_data)

        self.set_model_acc_dir(mode)
        self.svm()


    @DataOverview.calc_timing("svm")
    def svm(self):
        best_grid_acc = []  # index coordinates for best acc in grid search
        accuracity_matrix = {}  # matrix for saving acc
        best_acc = {"fpr": 100, "tpr": 0}  # acc initiation

        svcModel = SVC(kernel="rbf")
        svcModel.fit(self.X_train, self.y_train)
        y_pred = svcModel.predict(self.X_validation)

        self.draw_calc_matrix(self.y_validation, y_pred, "svm")
















if __name__ == "__main__":
    Model_2(csv_data=csv_input)



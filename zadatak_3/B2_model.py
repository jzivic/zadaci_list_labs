import time
from A_prep import *
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



# gamma_range = range(10, 11)
# C_range = range(10, 11)
# C_f = lambda n: 2 ** n
# gamma_f = lambda n: 2 ** n




C_f = lambda n: 2 ** n
gamma_f = lambda n: 2 ** n




class Model_Tuning(DataOverview):

    def __init__(self, csv_data, mode="validation"):
        super().__init__(csv_data)
        self.set_model_acc_dir(mode)

        # self.svm_grid_search()

        # self.grid_search()
        self.model_svm = self.svm(1e4)




    # after examination, C does not have significant impact and model times are similar. C=1e4
    def grid_search(self):
        for c in range(1,20000,1000):
            self.svm(c)



    @DataOverview.calc_timing("svm")
    def svm(self, c, gamma="scale", draw_matrix=True):
        svm_model = SVC(C=c, kernel="rbf", gamma=gamma)
        svm_model.fit(self.X_train, self.y_train)
        y_pred = svm_model.predict(self.X_validation)
        if draw_matrix is True:
            self.draw_calc_matrix(self.y_validation, y_pred, "scm")

        return svm_model







if __name__ == "__main__":
    Model_2(csv_data=csv_input)








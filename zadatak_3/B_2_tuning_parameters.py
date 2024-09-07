from B_1_models import *


class Model_Tuning(Models):

    def __init__(self, csv_data, mode="validation"):
        super().__init__(csv_data)

        self.set_model_acc_dir(mode)
        # self.svc_tuning()
        # self.knn_tuning()


    # after examination, C does not have significant impact and model times are similar. C=1e4
    def svc_tuning(self):
        for c in range(1,20000,1000):
            self.svm_f(c)

    # number of  neighbours does not play substantial role, time is also similar. n_neighbors=5
    def knn_tuning(self):
        for n_neighbor in range(3,10):
            print(n_neighbor)
            self.knn_f(n_neighbor)






if __name__ == "__main__":

    Model_Tuning(csv_data=csv_input)








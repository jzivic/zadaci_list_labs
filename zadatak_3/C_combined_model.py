from collections import Counter
from B_1_models import *




class CombinedModel(Models):

    def __init__(self, csv_data):
        super().__init__(csv_data, mode="test")
        self.create_combined_prediction()


    def create_combined_prediction(self):

        y_dt = self.model_dt.predict(self.X_test).tolist()
        y_rf = self.model_rf.predict(self.X_test).tolist()
        y_gnb = self.model_gnb.predict(self.X_test).tolist()
        y_svm = self.model_gnb.predict(self.X_test).tolist()
        y_knn = self.model_gnb.predict(self.X_test).tolist()

        avg_prediction = list(map(lambda x: Counter(x).most_common(1)[0][0], zip(y_dt, y_rf, y_gnb, y_svm, y_knn)))
        self.draw_calc_matrix(avg_prediction, self.y_test, "Combined_model")




if __name__ == "__main__":

    CombinedModel(csv_data=csv_input)











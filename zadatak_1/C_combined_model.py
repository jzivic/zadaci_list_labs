from collections import Counter

from B_models import *



model_acc_dir = os.path.abspath(os.path.join(project_path, "../zadatak_1/output/model_acc_test_1/"))
os.makedirs(model_acc_dir, exist_ok=True)



class CombinedModel(Models):

    def __init__(self, csv_data):
        super().__init__(csv_data)


        self.create_combined_prediction()





    def create_combined_prediction(self):

        y_dt = self.model_dt.predict(self.X_test).tolist()
        y_rf = self.model_rf.predict(self.X_test).tolist()
        y_gnb = self.model_gnb.predict(self.X_test).tolist()
        avg_prediction = list(map(lambda x: Counter(x).most_common(1)[0][0], zip(y_dt, y_rf, y_gnb)))

        self.draw_calc_matrix(avg_prediction, self.y_test, "Combined_model")













if __name__ == "__main__":

    CombinedModel(csv_data=csv_input)











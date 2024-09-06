
from C_model import *




class CombinedModel(Model_2):

    def __init__(self, csv_data):
        super().__init__(csv_data)

        self.proba()





    def proba(self):


        model = self.model_dt

        b = model.predict(self.X_test)


        a = self.draw_calc_matrix(self.y_test, b)







if __name__ == "__main__":

    CombinedModel(csv_data=csv_input)











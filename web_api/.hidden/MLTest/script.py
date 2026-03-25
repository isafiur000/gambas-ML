#import numpy
import pandas
#import sklearn
import sys

from mltest_model import lin_reg
from mltest_preprocess import preprocess_input

age = sys.argv[1]
gender = sys.argv[2]
bilirubindirect = sys.argv[3]
data = {"Age": age,"Gender": gender,"Bilirubin Direct": bilirubindirect}

input_df = pandas.DataFrame(data, index=[0])
X_input = preprocess_input.transform(input_df)
predict_val = lin_reg.predict(X_input)

print(predict_val[0])


import mariadb
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style

try:
    conn = mariadb.connect(
        user="fredrik",
        password="fredrik",
        host="localhost",
        database="fredrik"
    )
except mariadb.Error as e:
    print(f"Error: {e}")
    sys.exit(1)

cur = conn.cursor()
query = "select * from clean"
#df = pd.read_sql_query(query,conn, index_col="ssn")
df = pd.read_sql_query(query,conn)
df['sex_enum'] = np.where(df['sex'] == "Kvinna", 0, 1) # Kvinna = 0, Man = 1
df['nyanland_enum'] = np.where(df['mothertongue'] == "Svenska", 0, 1) # Svenska = 0, annat språk = 1

eights = df[df["ssn"].str.match('^05.*') == True]
eights = eights[["ssn", "g6_en", "g7_en", "g8_en"]].dropna()
eights.reset_index(inplace = True, drop = True)

#data = df[["sex_enum", "nyanland_enum", "g6_en", "g7_en", "g8_en", "g9_en"]]
data = df[["g6_en", "g7_en", "g8_en", "g9_en"]]
data = data.dropna()
#print(data)
predict = "g9_en"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    ## Train the model and save it to Pickle file
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    #print(acc)

    if acc > best:
        best = acc
        with open("la-mltest.pickle", "wb") as f:
            pickle.dump(linear, f)

#pickle_in = open("studentmodel.pickle", "rb")
#linear = pickle.load(pickle_in)

#print("Coefficient: ", linear.coef_)
#print("Intercept: ", linear.intercept_ )

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(" Grades: ", format(x_test[x]), " Guess: {:.2f}".format(predictions[x]), " Real: ", y_test[x], "::", "Diff: {:.2f}".format(y_test[x]-predictions[x]))


eights_predictions = linear.predict(eights[["g6_en", "g7_en", "g8_en"]].dropna())
eguess = pd.DataFrame({'g9_en_guess': eights_predictions})
eights_full = eights.join(eguess)
eights_full.to_csv('eights_full.csv')

#with np.printoptions(threshold=np.inf):
#    print(eights_full)

#for z in range(len(eights_predictions)):
    #print(z, " :: ", eights_predictions[z])

#newA = np.append(eights, eights_predictions, axis=1)

"""
single_student = np.array([20, 20, 15]).reshape(1, -1)
spred = linear.predict(single_student)
print(spred)
"""

"""
p = "g9_en"
style.use("ggplot")
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel(predict)
pyplot.show()
"""

"""
values = [np.average(data["g6_en"]), np.average(data["g7_en"]), np.average(data["g8_en"]), np.average(data["g9_en"])]
pyplot.bar(["6", "7", "8", "9"], values)
pyplot.show()
"""

#cur.execute("select sex from clean")
#print(cur)
#for(sex) in cur:
#    print(f"Sex: {sex}")


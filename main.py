import mariadb
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

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
df = pd.read_sql_query(query, conn)

df['sex_enum'] = np.where(df['sex'] == "Kvinna", 0, 1) # Kvinna = 0, Man = 1
df['language_enum'] = np.where(df['mothertongue'] == "Svenska", 0, 1) # Svenska = 0, annat språk = 1
df['total_points'] = df.drop(['ssn', 'sex', 'mothertongue', 'sex_enum', 'language_enum'], axis=1).sum(axis=1)

younger = df[df["ssn"].str.match('^06.*|^07.*') == True].copy()

eights = df[df["ssn"].str.match('^05.*') == True].copy()
eights.dropna(subset=['g8_ma'])

alumnis = df[df["ssn"].str.match('^04.*|03.*|02.*|01.*|00.*|99.*|98.*|97.*') == True].copy()
alumnis.dropna(subset=['g8_ma', 'g9_ma'], inplace=True) # Only keep students who actually got a grade in year 8 and 9

g9_subjects_all = ['g9_sv', 'g9_sva', 'g9_en', 'g9_ma', 'g9_bi', 'g9_fy', 'g9_ke', 'g9_tk', 'g9_ge', 'g9_hi', 'g9_re', 'g9_sh', 'g9_bd', 'g9_hkk', 'g9_idh', 'g9_mu', 'g9_sl', 'g9_mspr']
g9_subjects_core = ['g9_sv', 'g9_sva', 'g9_en', 'g9_ma']

alumnis['meritvarde'] = alumnis[g9_subjects_all].sum(axis=1) # 16 subjects + Mspr = 17 * 20 = 340 max
predict = "meritvarde"

#X = np.nan_to_num(np.array(df.drop(['ssn', 'sex', 'mothertongue', 'sex_enum', 'language_enum', predict], 1)))
#X = np.nan_to_num(np.array(alumnis[['g6_sv', 'g6_sva', 'g6_en', 'g6_ma', 'g7_sv', 'g7_sva', 'g7_en', 'g7_ma', 'g8_sv', 'g8_sva', 'g8_en', 'g8_ma']]))
X = np.nan_to_num(np.array(alumnis[['g8_sv', 'g8_sva', 'g8_en', 'g8_ma', 'g8_bi', 'g8_fy', 'g8_ke', 'g8_tk', 'g8_ge', 'g8_hi', 'g8_re', 'g8_sh', 'g8_bd', 'g8_hkk', 'g8_idh', 'g8_mu', 'g8_sl', 'g8_mspr']]))
y = np.array(alumnis[predict])

linear = linear_model.LinearRegression()
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # Train the model and save it to Pickle file
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc

print("\nCoefficient: ", linear.coef_)
print("Intercept: ", linear.intercept_ )
print("Best: ", best)

predictions = linear.predict(x_test)

predictions_file = open("predictions2.txt", "w+")
#for x in range(len(predictions)):
#    print(" Grades: ", format(x_test[x]), " Guess: {:.2f}".format(predictions[x]), " Real: ", y_test[x], "::", "Diff: {:.2f}".format(y_test[x]-predictions[x]))
#    print(",{:.2f}".format(predictions[x]), ",", y_test[x], ",{:.2f}".format(y_test[x] - predictions[x]), file=predictions_file)
predictions_file.close()

eights.reset_index(inplace=True, drop=True)
#eights_predictions = linear.predict(eights[['g6_sv', 'g7_sv', 'g8_sv', 'g9_sv', 'g6_sva', 'g7_sva', 'g8_sva', 'g9_sva', 'g6_en', 'g7_en', 'g8_en', 'g9_en', 'g6_ma', 'g7_ma', 'g8_ma', 'g9_ma']].fillna(0))
eights_predictions = linear.predict(eights[['g8_sv', 'g8_sva', 'g8_en', 'g8_ma', 'g8_bi', 'g8_fy', 'g8_ke', 'g8_tk', 'g8_ge', 'g8_hi', 'g8_re', 'g8_sh', 'g8_bd', 'g8_hkk', 'g8_idh', 'g8_mu', 'g8_sl', 'g8_mspr']].fillna(0))
eights_guess = pd.DataFrame({'g9_meritvarde_guess': eights_predictions})

eights_full = eights.join(eights_guess)
eights_full = eights_full[['ssn', 'g9_meritvarde_guess']]
eights_full.to_csv('eights_full.csv')
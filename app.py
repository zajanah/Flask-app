import pickle

from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek

app = Flask(__name__)

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split('X.csv', 'Y.csv', test_size=0.2, random_state=0)

# Scaling data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Oversample data
smk = SMOTETomek()
# Training data
x_train, y_train = smk.fit_sample(x_train, y_train)
# Testing data
x_test, y_test = smk.fit_sample(x_test, y_test)

# Fitting RandomForestClassifier to the Training set
rfr = RandomForestClassifier()
rfr.fit(x_train, x_train)

# Predicting the Test set results
y_pred = rfr.predict(x_test)

# Saving model to disk
pickle.dump(rfr, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


standard_to = StandardScaler()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        credit_score = int(request.form['CreditScore'])
        age = int(request.form['Age'])
        tenure = int(request.form['Tenure'])
        balance = float(request.form['Balance'])
        num_of_products = int(request.form['NumOfProducts'])
        has_cr_card = int(request.form['HasCrCard'])
        is_active_member = int(request.form['IsActiveMember'])
        estimated_salary = float(request.form['EstimatedSalary'])
        geography = request.form['Geography']
        if geography == 'Germany':
            geography = 1

        elif geography == 'Spain':
            geography = 2

        else:
            geography = 0
        gender = request.form['Gender']
        if gender == 'Male':
            gender = 0
        else:
            gender = 1
        prediction = model.predict([[credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member,
                                     estimated_salary, geography, gender]])
        if prediction == 1:
            return render_template('index.html', prediction_text="The Customer will leave the bank")
        else:
            return render_template('index.html', prediction_text="The Customer will not leave the bank")


if __name__ == "__main__":
    app.run(debug=True)

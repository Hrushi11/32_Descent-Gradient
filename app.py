import os
import csv
import pickle
import sklearn
import webbrowser
import numpy as np
import pandas as pd
import tensorflow as tf
from pnuemonia import pred_model
from flask import Flask, render_template, request


app = Flask(__name__)

# loading the saved models
model = tf.keras.model.load_model("models/pnuemonia")
diabetes_model = pickle.load(open('models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('models/parkinsons_model.sav', 'rb'))
otherdiseases_model = pickle.load(open('models/otherdiseases.sav', 'rb'))


# reading csv and converting the data to integer
def open_csv(filepath, filename):
    with open(filepath + filename, mode='r') as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            arr = lines

        data = list(map(float, arr))
        return data


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'csv'}


@app.route('/db_form', methods=['GET', 'POST'])
def db_form():
    global predictions, file_name, predictions1, predictions2, predictions3, predictions4, predictions5, data
    error = ''

    if request.method == "POST":

        fullname = request.form.get("fullname")

        preg = request.form.get("preg")

        glucose = request.form.get("glucose")

        bp = request.form.get("bp")

        skin = request.form.get("skin")

        insulin = request.form.get("insulin")

        bmi = request.form.get("bmi")

        dpf = request.form.get("dpf")

        age = request.form.get("age")

        data = [preg, glucose, bp, skin, insulin, bmi, dpf, age]
        data_conv = np.array([data])
        data_csv = pd.DataFrame(data_conv,
                                columns=['Pregnancies', 'Glucose', 'Blood Pressure', 'SkinThickness', 'Insulin', 'BMI',
                                         'Diabetes Pedigree Function', 'Age'])

        print(data_csv)
        data = list(map(float, data))

        # compare_data = compare_input(data, DB_Models)
        if (len(data) == 8):

            # changing the input_data_diabetes to numpy array
            input_data_diabetes_as_numpy_array = np.asarray(data)
            # reshape the array as we are predicting for one instance
            input_data_diabetes_reshaped = input_data_diabetes_as_numpy_array.reshape(1, -1)
            # prediction = diabetes_model.predict(input_data_diabetes_reshaped)
            print(input_data_diabetes_reshaped)
            prediction1 = diabetes_model.predict(input_data_diabetes_reshaped)

            # Accuracies
            model = diabetes_model.predict_proba(input_data_diabetes_reshaped)
            acc = model.max()

            if (prediction1[0] == 0):
                predictions = 'not diabetic' + f'   {(round(acc, 3) * 100)}%'
            else:
                predictions = 'diabetic' + f'   {(round(acc, 3) * 100)}%'

        if (len(error) == 0):
            return render_template('dep.html', type="csv", disease="db",
                                   predictions=predictions,
                                   data=data_csv.to_html(classes='mystyle', index=False))
        else:
            return render_template('index.html', error=error)

    return render_template("diabetes_form.html")


@app.route('/hd_form', methods=['GET', 'POST'])
def hd_form():
    global predictions, file_name, predictions1, predictions2, predictions3, predictions4, predictions5, data
    error = ''

    if request.method == "POST":

        fullname = request.form.get("fullname")

        age = request.form.get("age")

        sex = request.form.get("sex")

        cp = request.form.get("cp")

        trestbps = request.form.get("trestbps")

        chol = request.form.get("chol")

        fbs = request.form.get("fbs")

        restecg = request.form.get("restecg")

        thalach = request.form.get("thalach")

        exang = request.form.get("exang")

        oldpeak = request.form.get("oldpeak")

        slope = request.form.get("slope")

        ca = request.form.get("ca")

        thal = request.form.get("thal")

        data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        data_conv = np.array([data])
        data_csv = pd.DataFrame(data_conv,
                                columns=["age", " sex", " cp", " trestbps", " chol", " fbs", " restecg", " thalach",
                                         "exang", "oldpeak", "slope", "ca", "thal"])

        print(data_csv)
        data = list(map(float, data))

        if (len(data) == 13):

            input_data_heartd_as_numpy_array = np.asarray(data)
            # reshape the numpy array as we are predicting for only on instance
            input_data_heartd_reshaped = input_data_heartd_as_numpy_array.reshape(1, -1)

            prediction1 = heart_disease_model.predict(input_data_heartd_reshaped)

            # Accuracies
            model = heart_disease_model.predict_proba(input_data_heartd_reshaped)
            acc = model.max()

            if (prediction1[0] == 0):
                predictions = 'not Heart Disease' + f'   {(round(acc, 3) * 100)}%'
            else:
                predictions = 'Heart Disease' + f'   {(round(acc, 3) * 100)}%'

        if (len(error) == 0):
            return render_template('dep.html', type="csv", disease="hd",
                                   predictions=predictions,
                                   data=data_csv.to_html(classes='mystyle', index=False))
        else:
            return render_template('index.html', error=error)

    return render_template("heart_disease_form.html")


@app.route('/pk_form', methods=['GET', 'POST'])
def pk_form():
    global predictions, file_name, predictions1, predictions2, predictions3, predictions4, predictions5, data
    error = ''

    if request.method == "POST":

        fullname = request.form.get("fullname")

        a = request.form.get("1")

        b = request.form.get("2")

        c = request.form.get("3")

        d = request.form.get("4")

        e = request.form.get("5")

        f = request.form.get("6")

        g = request.form.get("7")

        h = request.form.get("8")

        i = request.form.get("9")

        j = request.form.get("10")

        k = request.form.get("11")

        l = request.form.get("12")

        m = request.form.get("13")

        n = request.form.get("14")

        o = request.form.get("15")

        p = request.form.get("16")

        q = request.form.get("17")

        r = request.form.get("18")

        s = request.form.get("19")

        t = request.form.get("20")

        u = request.form.get("21")

        v = request.form.get("22")

        data = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v]
        data_conv = np.array([data])
        data_csv = pd.DataFrame(data_conv, columns=["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
                                                    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
                                                    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
                                                    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1",
                                                    "spread2", "D2", "PPE"])

        print(data)
        data = list(map(float, data))

        if (len(data) == 22):
            input_data_parkinsons_as_numpy_array = np.round_(data, decimals=4)

            input_data_parkinsons_reshaped = input_data_parkinsons_as_numpy_array.reshape(1, -1)

            prediction1 = parkinsons_model.predict(input_data_parkinsons_reshaped)

            # Accuracies

            model = parkinsons_model.predict_proba(input_data_parkinsons_reshaped)
            acc = model.max()

            if (prediction1[0] == 0):
                predictions = 'not Parkinsons' + f'      {(round(acc, 3) * 100)}%'
            else:
                predictions = 'Parkinsons' + f'      {(round(acc, 3) * 100)}%'

        if (len(error) == 0):
            return render_template('dep.html', type="csv", disease="pk",
                                   predictions=predictions,
                                   data=data_csv.to_html(classes='mystyle', index=False))
        else:
            return render_template('index.html', error=error)

    return render_template("parkinsons_form.html")


@app.route('/od_form', methods=['GET', 'POST'])
def od_form():
    global predictions, file_name, predictions1, predictions2, predictions3, predictions4, predictions5, data
    error = ''

    if request.method == "POST":

        fullname = request.form.get("fullname")

        a = request.form.get("1")

        b = request.form.get("2")

        c = request.form.get("3")

        d = request.form.get("4")

        e = request.form.get("5")

        f = request.form.get("6")

        g = request.form.get("7")

        h = request.form.get("8")

        i = request.form.get("9")

        j = request.form.get("10")

        k = request.form.get("11")

        l = request.form.get("12")

        m = request.form.get("13")

        n = request.form.get("14")

        o = request.form.get("15")

        p = request.form.get("16")

        q = request.form.get("17")

        r = request.form.get("18")

        s = request.form.get("19")

        t = request.form.get("20")

        u = request.form.get("21")

        v = request.form.get("22")

        w = request.form.get("23")

        x = request.form.get("24")

        y = request.form.get("25")

        z = request.form.get("26")

        aa = request.form.get("27")

        ab = request.form.get("28")

        ac = request.form.get("29")

        ad = request.form.get("30")

        ae = request.form.get("31")

        af = request.form.get("32")

        ag = request.form.get("33")

        data = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, aa, ab, ac, ad, ae, af,
                ag]
        data_conv = np.array([data])
        data_csv = pd.DataFrame(data_conv,
                                columns=["itching", "skin_rash", "shivering", "chills", "vomiting", "fatigue",
                                         "high_fever", "headache", "yellowish_skin", "nausea", "loss_of_appetite",
                                         "pain_behind_the_eyes", "abdominial_pain", "diarrhoea", "mild_fever",
                                         "yellowing_of_eyes", "malaise", "runny_nose", "chest_pain",
                                         "pain_in_anal_region", "neck_pain", "dizziness", "swollen_extremeties",
                                         "slurred_speech", "loss_of_balance", "bladder_discomfort", "irritability",
                                         "increased_appetite", "stomach_bleeding", "painful_walking",
                                         "small_dents_in_nails", "blister", "prognosis"])

        print(data_csv)
        data = list(map(float, data))

        if (len(data) == 33):
            new_data = np.asarray(data)
            new_data2 = new_data.reshape(1, -1)
            # compute probabilities of assigning to each of the classes of prognosis
            probaDT = otherdiseases_model.predict_proba(new_data2)
            probaDT.round(4)  # round probabilities to four decimal places, if applicabl

            data = new_data2

            # Accuracies
            knn = otherdiseases_model.predict_proba(new_data2)
            acc = knn.max()

            pred1 = otherdiseases_model.predict(new_data2)
            predictions = f"{pred1[0]}" + f'{(round(acc, 3) * 100)}%'

        if (len(error) == 0):
            return render_template('dep.html', type="csv", disease="od",
                                   predictions=predictions,
                                   data=data_csv.to_html(classes='mystyle', index=False))
        else:
            return render_template('index.html', error=error)

    return render_template("other_diseases_form.html")


def open_csv_2(filepath, filename):
    with open(filepath + filename, mode='r') as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            arr = lines

        data = list(arr)
        return data


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


@app.route('/success', methods=['GET', 'POST'])
def success():
    global predictions, file_name, data, data_csv
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images/')

    if request.method == 'POST':

        if (request.files):
            file = request.files['file']

            if file and allowed_file(file.filename):

                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                file_name = file.filename

                if ".csv" in file.filename:

                    data = open_csv('static/images/', file.filename)

                    f = f"static/images/{file.filename}"
                    data_csv = []
                    with open(f) as file:
                        csvfile = csv.reader(file)
                        for row in csvfile:
                            data_csv.append(row)
                    data_csv = pd.DataFrame(data_csv)

                    if (len(data) == 8):

                        # changing the input_data_diabetes to numpy array
                        input_data_diabetes_as_numpy_array = np.asarray(data)
                        # reshape the array as we are predicting for one instance
                        input_data_diabetes_reshaped = input_data_diabetes_as_numpy_array.reshape(1, -1)
                        # prediction = diabetes_model.predict(input_data_diabetes_reshaped)
                        print(input_data_diabetes_reshaped)
                        prediction1 = diabetes_model.predict(input_data_diabetes_reshaped)

                        # Accuracies
                        model = diabetes_model.predict_proba(input_data_diabetes_reshaped)
                        acc = model.max()

                        if (prediction1[0] == 0):
                            predictions = 'not diabetic' + f'   {(round(acc, 3) * 100)}%'
                        else:
                            predictions = 'diabetic' + f'   {(round(acc, 3) * 100)}%'

                        if (len(error) == 0):
                            return render_template('dep.html', type="csv", disease="db",
                                                   predictions=predictions,
                                                   data=data_csv.to_html(classes='mystyle', index=False))
                        else:
                            return render_template('index.html', error=error)

                    elif (len(data) == 13):

                        input_data_heartd_as_numpy_array = np.asarray(data)
                        # reshape the numpy array as we are predicting for only on instance
                        input_data_heartd_reshaped = input_data_heartd_as_numpy_array.reshape(1, -1)

                        prediction1 = heart_disease_model.predict(input_data_heartd_reshaped)

                        # Accuracies
                        model = heart_disease_model.predict_proba(input_data_heartd_reshaped)
                        acc = model.max()

                        if prediction1[0] == 0:
                            predictions = 'not Heart Disease' + f'   {(round(acc, 3) * 100)}%'
                        else:
                            predictions = 'Heart Disease' + f'   {(round(acc, 3) * 100)}%'

                        if len(error) == 0:
                            return render_template('dep.html', type="csv", disease="hd",
                                                   predictions=predictions,
                                                   data=data_csv.to_html(classes='mystyle', index=False))
                        else:
                            return render_template('index.html', error=error)

                    elif len(data) == 22:
                        input_data_parkinsons_as_numpy_array = np.round_(data, decimals=4)
                        # reshape the numpy array
                        input_data_parkinsons_reshaped = input_data_parkinsons_as_numpy_array.reshape(1, -1)

                        prediction1 = parkinsons_model.predict(input_data_parkinsons_reshaped)

                        # Accuracies
                        model = parkinsons_model.predict_proba(input_data_parkinsons_reshaped)
                        acc = model.max()

                        if prediction1[0] == 0:
                            predictions = 'not Parkinsons' + f'      {(round(acc, 3) * 100)}%'
                        else:
                            predictions = 'Parkinsons' + f'      {(round(acc, 3) * 100)}%'

                        if len(error) == 0:
                            return render_template('dep.html', type="csv", disease="pk",
                                                   predictions=predictions,
                                                   data=data_csv.to_html(classes='mystyle', index=False))
                        else:
                            return render_template('index.html', error=error)

                    elif len(data) == 33:
                        new_data = np.asarray(data)
                        new_data2 = new_data.reshape(1, -1)
                        probaDT = otherdiseases_model.predict_proba(new_data2)
                        probaDT.round(4)  # round probabilities to four decimal places, if applicabl
                        data = new_data2

                        # Accuracies

                        model = otherdiseases_model.predict_proba(new_data2)
                        acc = model.max()

                        pred1 = otherdiseases_model.predict(new_data2)
                        predictions = f"{pred1[0]}" + f'{(round(acc, 3) * 100)}%'

                        if len(error) == 0:

                            return render_template('dep.html', type="csv", disease="od",
                                                   predictions=predictions,
                                                   data=data_csv.to_html(classes='mystyle', index=False))

                        else:
                            return render_template('index.html', error=error)


                else:
                    class_result, prob_result = pred_model(img_path)
                    predictions = (class_result, int(prob_result * 100))

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if len(error) == 0:
                if ".csv" in file_name:
                    return render_template('dep.html', type="csv", predictions=predictions,
                                           data=data_csv.to_html(classes='mystyle', header=False, index=False))
                else:
                    return render_template('dep.html', img=file_name, type="img", predictions=predictions)
            else:
                return render_template('index.html', error=error)
    else:
        return render_template('index.html')


@app.route('/')
def home():
    return render_template("index.html")


if __name__ == "__main__":
    webbrowser.open_new('http://127.0.0.1:2000/')
    app.run(debug=True, port=2000)

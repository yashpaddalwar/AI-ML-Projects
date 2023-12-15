from flask import Flask, render_template, request
import pickle
import sys
import logging
import numpy as np

model = pickle.load(open('finalmodelchurn.pickle','rb'))
model1_probability1 = pickle.load(open('probabilities.pickle','rb'))

app = Flask(__name__)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

@app.route('/',methods=['POST','GET'])
def home():
    return render_template('home.html')

@app.route('/test',methods=['POST','GET'])
def test():
    return render_template('test.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        # Senior-Citizen
        is_seniorcitizen = request.form['seniorcitizen']
        if is_seniorcitizen == 'yes':
            seniorcitizen = 1
        else:
            seniorcitizen = 0

        # Monthly-Charges
        monthlycharges = float(request.form['monthlycharges'])

        #totalcharges
        totalcharges = float(request.form['totalcharges'])

        # Gender
        is_gender = request.form['gender']
        if is_gender == 'Male':
            gender_Male = 1
        else:
            gender_Male = 0
        
        # Partner_Yes

        partner = request.form['partner']
        if partner == 'Yes':
            Partner_Yes = 1
        else:
            Partner_Yes = 0
        
        # Dependent_Yes
        dependent = request.form['dependent']
        if dependent == 'Yes':
            Dependent_Yes = 1
        else:
            Dependent_Yes = 0

        # PhoneService_Yes
        phoneservice = request.form['phoneservice']
        if phoneservice == 'Yes':
            PhoneService = 1
        else:
            PhoneService = 0

        # multiplelinesphoneservice
        multiplelinesphoneservice = request.form['multiplelinesphoneservice']
        if multiplelinesphoneservice == 'Nophoneservice':
            MultipleLinesNoPhoneService = 1
        else:
            MultipleLinesNoPhoneService = 0

        # MultipleLines_Yes
        if multiplelinesphoneservice == 'Yes':
            MultipleLines_Yes = 1
        else:
            MultipleLines_Yes = 0

        # internetservice
        internetservice = request.form['internetservice']
        if internetservice == 'FibreOptic':
            InternetServiceFibreOptics = 1
        else:
            InternetServiceFibreOptics = 0
        
        if internetservice == 'No':
            InternetService_No = 1
        else:
            InternetService_No = 0

        # OnlineSecurity
        onlinesecurity = request.form['onlinesecurity']
        if onlinesecurity == 'Noonlinesecurity':
            OnlineSecurity_NoInternetService = 1
        else:
            OnlineSecurity_NoInternetService = 0
        
        if onlinesecurity == 'Yes':
            OnlineSecurity_Yes = 1
        else:
            OnlineSecurity_Yes = 0

        # OnlineBackup
        onlinebackup = request.form['onlinebackup']
        if onlinebackup == 'Nointernetbackup':
            OnlineBackup_NoInternetService = 1
        else:
            OnlineBackup_NoInternetService = 0
        
        if onlinebackup == 'Yes':
            OnlineBackup_Yes = 1
        else:
            OnlineBackup_Yes = 0

        # DeviceProtection
        deviceprotection = request.form['deviceprotection']
        if deviceprotection == 'Nodeviceprotection':
            DeviceProtection_NoInternetService = 1
        else:
            DeviceProtection_NoInternetService = 0
        
        if deviceprotection == 'Yes':
            DeviceProtection_Yes = 1
        else:
            DeviceProtection_Yes = 0

        # Tech Support
        techsupport = request.form['techsupport']
        if techsupport == 'Notechsupport':
            TechSupport_NoInternetService = 1
        else:
            TechSupport_NoInternetService = 0
        
        if techsupport == 'Yes':
            TechSupport_Yes = 1
        else:
            TechSupport_Yes = 0

        # StreamingTV
        streamingtv = request.form['streamingtv']
        if streamingtv == 'Nostreamingtv':
            StreamingTV_NoInternetService = 1
        else:
            StreamingTV_NoInternetService = 0
        
        if streamingtv == 'Yes':
            StreamingTV_Yes = 1
        else:
            StreamingTV_Yes = 0

        # StreamingMovies
        streamingmovies = request.form['streamingmovies']
        if streamingmovies == 'Nostreamingmovies':
            StreamingTVMovies_NoInternetService = 1
        else:
            StreamingTVMovies_NoInternetService = 0
        
        if streamingmovies == 'Yes':
            StreamingTVMovies_Yes = 1
        else:
            StreamingTVMovies_Yes = 0

        # Contract
        contract = request.form['contract']
        if contract == 'one':
            Contract_One = 1
        else:
            Contract_One = 0
        
        if contract == 'Two':
            Contract_Two = 1
        else:
            Contract_Two = 0

        # PaperlessBilling_Yes
        paperlessbilling = request.form['paperlessbilling']
        if paperlessbilling == 'Yes':
            PaperlessBilling_Yes = 1
        else:
            PaperlessBilling_Yes = 0

        # PaymentMethod
        paymentmethod = request.form['paymentmethod']
        if paymentmethod == 'Creditcard':
            PaymentMethod_CreditCard = 1
        else:
            PaymentMethod_CreditCard = 0

        if paymentmethod == 'Electroniccheck':
            PaymentMethod_ElectronicCheck = 1
        else:
            PaymentMethod_ElectronicCheck = 0

        if paymentmethod == 'Mailedcheck':
            PaymentMethod_MailedCheck = 1
        else:
            PaymentMethod_MailedCheck = 0

        # Tenure Group
        tenure = request.form['tenure']
        if tenure == 'Twelve':
            tenure_Twelve = 1
        else:
            tenure_Twelve = 0
        
        if tenure == 'Twentyfour':
            tenure_Twentyfour = 1
        else:
            tenure_Twentyfour = 0

        if tenure == 'Thirtysix':
            tenure_Thirtysix = 1
        else:
            tenure_Thirtysix = 0
        
        if tenure == 'Fortyeight':
            tenure_Fortyeight = 1
        else:
            tenure_Fortyeight = 0
        
        if tenure == 'Sixty':
            tenure_Sixty = 1
        else:
            tenure_Sixty = 0
        
        totalcharges = np.log(totalcharges)
        arr = [[seniorcitizen,monthlycharges,totalcharges,gender_Male,Partner_Yes,Dependent_Yes,PhoneService,MultipleLinesNoPhoneService,MultipleLines_Yes,InternetServiceFibreOptics,InternetService_No,OnlineSecurity_NoInternetService,OnlineSecurity_Yes,OnlineBackup_NoInternetService,OnlineBackup_Yes,DeviceProtection_NoInternetService,DeviceProtection_Yes,TechSupport_NoInternetService,TechSupport_Yes,StreamingTV_NoInternetService,StreamingTV_Yes,StreamingTVMovies_NoInternetService,StreamingTVMovies_Yes,Contract_One,Contract_Two,PaperlessBilling_Yes,PaymentMethod_CreditCard,PaymentMethod_ElectronicCheck,PaymentMethod_MailedCheck,tenure_Twelve,tenure_Twentyfour,tenure_Thirtysix,tenure_Fortyeight,tenure_Sixty]]

        prediction_arr = model.predict(arr)
        # prediction_probability = model1_probability1.predict_proba(arr)[:,1]
        # prediction_probability = "{:.2f}".format(prediction_probability[0]*100)

        prediction_probability = model.predict_proba(arr)[:,-1]
        # prediction_probability = "{:.2f}".format(prediction_probability[0]*100)
        prediction_probability = prediction_probability[0]*100

        print(prediction_probability)

        if prediction_probability < 50:
            prediction_probability_new = 100 - prediction_probability
        else:
            prediction_probability_new = prediction_probability

        print(prediction_probability_new)
        lst = []
        for i in arr:
            lst.append(i)
        arr_new = lst[0]
        # final_prob = arr_new[1]
        # final_prob = round(final_prob)
        is_prediction = prediction_arr[0]
        if is_prediction == 1:
            prediction = 'churn'
        else:
            prediction = 'not churn'
        # prediction_probability = round(prediction_probability)
        print(prediction_probability)
        # return render_template('predict.html',prediction=prediction,final_prob=final_prob)
        return render_template('predict.html',prediction=prediction,prediction_probability_new=prediction_probability_new)
    else:    
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
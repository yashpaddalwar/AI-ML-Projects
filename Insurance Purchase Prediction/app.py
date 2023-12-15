from flask import Flask, render_template,request
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,LassoCV

model = pickle.load(open('Insurance Model.pickle','rb'))
# a = model.predict([[324.000000,107.0,4.0,4.0,4.5,8.87,1]])
app = Flask(__name__)
df = pd.read_csv('insurance.csv')
df['sex'] = df['sex'].replace('male',1)
df['sex'] = df['sex'].replace('female',0)
df['sex'].astype(int)
# Transforming smoker column
df['smoker'] = df['smoker'].replace('yes',1)
df['smoker'] = df['smoker'].replace('no',0)
df['smoker'].astype(int)
# Transforming region column
df['region'] = df['region'].replace('southwest',1)
df['region'] = df['region'].replace('southeast',2)
df['region'] = df['region'].replace('northwest',3)
df['region'] = df['region'].replace('northeast',4)
df['region'].astype(int)
x = df.drop(columns=['expenses'])
y = df['expenses']

scaler = StandardScaler()
arr = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(arr,y,test_size=0.15,random_state=100)

lassocv = LassoCV(cv=5,max_iter=10000000,alphas=None,normalize=True)
lassocv.fit(xtrain,ytrain)
lasso = Lasso(alpha=lassocv.alpha_)
lasso.fit(xtrain,ytrain)




@app.route('/',methods=['POST','GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        # try:
            age = request.form['age']
            is_sex = request.form['sex']
            bmi = float(request.form['bmi'])
            children = float(request.form['children'])
            is_smoker = request.form['smoker']
            is_region = request.form['region']
            if is_sex == 'Male':
                sex = 1
            else:
                sex = 0
            
            if is_smoker == 'Yes':
                smoker = 1
            else:
                smoker = 0

            if is_region == 'southwest':
                region = 1
            elif is_region == 'southeast':
                region = 2
            elif is_region == 'northwest':
                region = 3
            else:
                region = 4
            arr = [[age,sex,bmi,children,smoker,region]]
            arrfit = scaler.transform(arr)
            prediction = lasso.predict(arrfit)
            if prediction[0] < 0:
                prediction = [0]
            return render_template('predict.html',prediction=round(prediction[0],2))
        # except Exception as e:
        #     print(f"The Exception is : {e}")
        #     return "<h1 style='color:red;'>Something went wrong :(</h1>"
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask,render_template,request
import numpy as np



import pickle
with open("artifacts\lr_model.pkl",'rb') as model_file:
    model=pickle.load(model_file)
    


# import json

# with open('column_name.json','r') as json_file:
#     col_name=json.load(json_file)

# print(col_name)





app= Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])

def predict():
    data=request.form 

    
    
    final_input=float(data['cgpa'])
    

    print(data)
    print(f'final_input= {final_input}')

    result=model.predict([[final_input]]) 
    print(result)
    return 'The package you will get :' + str(np.round(result[0][0],2)) + 'lpa'
    

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8080,debug=True)
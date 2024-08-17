from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        # Retrieve input data from the form
        item_weight = float(request.form['item_weight'])
        item_fat_content = float(request.form['item_fat_content'])
        item_visibility = float(request.form['item_visibility'])
        item_type = float(request.form['item_type'])
        item_mrp = float(request.form['item_mrp'])
        outlet_establishment_year = float(request.form['outlet_establishment_year'])
        outlet_size = float(request.form['outlet_size'])
        outlet_location_type = float(request.form['outlet_location_type'])
        outlet_type = float(request.form['outlet_type'])

        # Create input array for the model
        X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                       outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

        # Load scaler and model from the specified paths
       
        scaler_path = r'C:\Users\dell\Desktop\Big Mart Sales Analysis NTCC\BigMart-Sales-Prediction-With-Deployment\models\scaler.sav'
        model_path = r'C:\Users\dell\Desktop\Big Mart Sales Analysis NTCC\BigMart-Sales-Prediction-With-Deployment\models\best_model.sav'


        

       # Load scaler and model from the specified paths
      
        sc = joblib.load(scaler_path)
        model = joblib.load(model_path)


        # Standardize the input data
        X_std = sc.transform(X)

        # Predict the output using the loaded model
        Y_pred = model.predict(X_std)

        # Render the predict.html template with the prediction result
        return render_template('predict.html', prediction="{:.4f}".format(float(Y_pred)))

    # Render the predict.html template for GET requests
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True, port=9457)
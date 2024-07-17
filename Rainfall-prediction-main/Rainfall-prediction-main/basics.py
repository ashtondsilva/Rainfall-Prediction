from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load the model
model = pickle.load(open('classifier.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('interface.html')

@app.route('/predict', methods=['POST'])
def interface():
    try:
        # Get form data
        data1 = request.form['a']
        data2 = request.form['b']
        data3 = request.form['c']
        data4 = request.form['d']
        
        # Convert input data to floats
        arr = np.array([[float(data1), float(data2), float(data3), float(data4)]])
        
        # Make prediction
        pred = model.predict(arr)[0]  # Get the first element of the prediction array
        
        # Determine the message based on the prediction value
        if pred == 0:
            message = "It's a sunny day."
        elif pred > 0 and pred < 30:
            message = "It's a rainy day."
        else:
            message = "RED ALERT: PLEASE REFRAIN FROM VENTURING OUTSIDE!!!!!"
        
        # Render the result in the interface.html template
        return render_template('interface.html', data=pred, message=message)
    except ValueError:
        # Handle the error if conversion fails
        error_message = "Please enter valid numeric values."
        return render_template('interface.html', error=error_message)

@app.route('/data', methods=['GET'])
def data():
    # This is just a placeholder; adjust as needed for your data source
    return jsonify(data=0)  # or data=1 based on actual conditions

if __name__ == "__main__":
    app.run(debug=True)



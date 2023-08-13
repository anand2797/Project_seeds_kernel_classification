from flask import Flask, request, render_template
import pandas as pd


from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

data = pd.read_csv('cleaned_seed_data.csv')

model = joblib.load('support_vector_machine_seed.pkl')

# scaling
scaler = StandardScaler()
scaler.fit(data.drop('target', axis=1))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        area = request.form['area']
        perimeter = request.form['perimeter']
        compactness = request.form['compactness']
        length = request.form['length']
        width = request.form['width']
        a_coef = request.form['a_coef']
        length_grv = request.form['lkg']

        input_df = pd.DataFrame([[area, perimeter, compactness, length, width, a_coef, length_grv]])

        scaled_data = scaler.transform(input_df)

        # prediction
        prediction = model.predict(scaled_data)

        return render_template('result.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)


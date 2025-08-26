from flask import Flask, request, render_template
from predict import predict_price

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'latitude': float(request.form['latitude']),
            'longitude': float(request.form['longitude']),
            'baths': int(request.form['baths']),
            'area_marla': float(request.form['area_marla']),
            'bedrooms': int(request.form['bedrooms']),
            'year': int(request.form['year']),
            'month': int(request.form['month']),
            'day': int(request.form['day']),
            'property_type': request.form['property_type'],
            'location': request.form['location'],
            'city': request.form['city'],
            'locality': request.form['locality'],
            'purpose': request.form['purpose']
        }

        # Validate required features
        required_features = ['latitude', 'longitude', 'baths', 'area_marla', 'bedrooms', 'year', 'month', 'day',
                            'property_type', 'location', 'city', 'locality', 'purpose']
        missing_features = [feat for feat in required_features if feat not in data or data[feat] is None]
        if missing_features:
            return render_template('index.html', error=f'Missing or invalid features: {missing_features}')

        # Make prediction
        predicted_price = predict_price(data)

        # Format price (in millions PKR)
        predicted_price_millions = round(predicted_price / 1_000_000, 2)

        # Return template with prediction and map coordinates
        return render_template('index.html',
                             prediction=predicted_price_millions,
                             lat=data['latitude'],
                             lon=data['longitude'],
                             city=data['city'],
                             locality=data['locality'])

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
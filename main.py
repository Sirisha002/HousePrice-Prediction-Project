from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    # Get unique values for dropdown options
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form input
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    # Create DataFrame with input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                               columns=['beds', 'baths', 'size', 'zip_code'])

    # Debug: print input data
    print("Input Data:")
    print(input_data)

    # Convert 'baths' column to numeric, coercing errors
    input_data['baths'] = pd.to_numeric(input_data['baths'], errors='coerce')

    # Convert other columns to appropriate types (int, float)
    input_data = input_data.astype({'beds': int, 'baths': float, 'size': float, 'zip_code': int})

    # Handle unknown categories (replace with mode value)
    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            print(f"Unknown categories in {column}: {unknown_categories}")
            # Replace unknown categories with the most frequent value (mode)
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    # Debug: print processed input data
    print("Processed Input Data:")
    print(input_data)

    # Make prediction using the trained model
    prediction = pipe.predict(input_data)[0]

    # Post-processing: Ensure the prediction is not negative (clamp to 0)
    prediction = max(0, prediction)

    # Return prediction as a string
    return str(prediction)

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True, port=5000)

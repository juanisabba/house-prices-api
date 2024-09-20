import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Load and preprocess data
df = pd.read_csv('https://www.dropbox.com/scl/fi/qzln61nb6q2ysxybnrufj/bsas_realstate_on_sale_properati_dataset_2020.csv?rlkey=le0z3sbq2kyl0xgpszmmt6bt4&st=wc5vrrqw&dl=1')
df = df[df["l2"] == "Capital Federal"]
df = df[(df["property_type"] == "Departamento")
        | (df["property_type"] == "PH")]
df.loc[df['l3'] == 'Catalinas', 'l3'] = 'Retiro'
df.loc[df['l3'] == 'Barrio Norte', 'l3'] = 'Recoleta'
df.loc[df['l3'] == 'Centro / Microcentro', 'l3'] = 'San Nicolás'
df.loc[df['l3'] == 'Congreso', 'l3'] = 'Balvanera'
df.loc[df['l3'] == 'Las Cañitas', 'l3'] = 'Palermo'
df.loc[df['l3'] == 'Once', 'l3'] = 'Balvanera'
df.loc[df['l3'] == 'Parque Centenario', 'l3'] = 'Villa Crespo'
df.loc[df['l3'] == 'Tribunales', 'l3'] = 'San Nicolás'
df = df[df['rooms'] < 7]
df = df[df['bedrooms'] < 6]
df = df[df['bathrooms'] < 6]
df.dropna(inplace=True)
df = df.drop(columns=["start_date", "end_date", "created_on", "l1",
             "l2", "currency", "title", "description", "operation_type"])
df = pd.get_dummies(df, columns=["l3", "property_type"], drop_first=True)

# Define feature matrix and target vector
X = df.drop(['price'], axis=1)
y = df['price']

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the RandomForestRegressor model on the entire dataset
forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest.fit(X_scaled, y)

# Define all possible feature columns based on the training data
all_feature_columns = X.columns.tolist()


def calculate_percentage_adjustment(years: int) -> float:
    if years == 0:
        return 0.05
    elif years == 15:
        return 0
    elif years >= 30 and years < 60:
        return -0.05
    elif years >= 60:
        return -0.1

    percentage = 0.05 - (years * 0.1 / 30)
    return percentage


app.route('/', methods=['GET'])


def home():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Initialize features with zeros for all possible feature columns
    features = {col: 0 for col in all_feature_columns}

    # Process categorical data (l3 and property_type)
    if 'l3' in data:
        l3_value = data['l3']
        l3_columns = [
            col for col in all_feature_columns if col.startswith('l3_')]
        if f'l3_{l3_value}' in l3_columns:
            features[f'l3_{l3_value}'] = 1

    if 'property_type' in data:
        property_type_value = data['property_type']
        property_type_columns = [
            col for col in all_feature_columns if col.startswith('property_type_')]
        if f'property_type_{property_type_value}' in property_type_columns:
            features[f'property_type_{property_type_value}'] = 1

    # Update features with other data from the request
    for key, value in data.items():
        if key not in ['l3', 'property_type']:
            if key in features:
                features[key] = value

    # Convert features to DataFrame and scale them
    features_df = pd.DataFrame(features, index=[0])
    features_scaled = scaler.transform(features_df)

    # Make prediction
    prediction = forest.predict(features_scaled)
    adjustment = 0
    adjustment_garage = 0.04

    if data['garage'] == 1:
        adjustment += adjustment_garage
    else:
        adjustment -= adjustment_garage

    year_adjustment = calculate_percentage_adjustment(data['year'])
    adjustment += year_adjustment

    prediction = prediction * (1 + adjustment)

    # Calculate the average price per square meter (m²) for the given neighborhood
    neighborhood_col = f'l3_{data["l3"]}'
    if neighborhood_col in df.columns:
        # Use .copy() to avoid SettingWithCopyWarning
        neighborhood_df = df[df[neighborhood_col] == 1].copy()
        # Calculate price per m² for the neighborhood
        neighborhood_df.loc[:, 'price_per_m2'] = neighborhood_df['price'] / \
            neighborhood_df['surface_total']
        average_m2_price = neighborhood_df['price_per_m2'].mean()
    else:
        return jsonify({'error': f'Neighborhood {data["l3"]} not found'}), 400

    # Calculate the average price per m² for all neighborhoods and sort them
    neighborhood_avg_m2_price_dict = {}
    neighborhood_columns = [col for col in df.columns if col.startswith('l3_')]

    # Iterate over one-hot encoded neighborhood columns
    for neighborhood in neighborhood_columns:
        neighborhood_df = df[df[neighborhood]
                             == 1].copy()  # Use .copy() here too
        # Calculate price per m² for each neighborhood
        neighborhood_df.loc[:, 'price_per_m2'] = neighborhood_df['price'] / \
            neighborhood_df['surface_total']
        avg_m2_price = neighborhood_df['price_per_m2'].mean()
        neighborhood_avg_m2_price_dict[neighborhood.replace(
            'l3_', '')] = avg_m2_price

    # Sort by price per m² (most expensive to cheapest)
    sorted_neighborhood_avg_m2_price = pd.Series(
        neighborhood_avg_m2_price_dict).sort_values(ascending=False)

    # Get the rank/position of the selected neighborhood
    try:
        neighborhood_position = sorted_neighborhood_avg_m2_price.index.get_loc(
            data['l3']) + 1  # 1-based index
    except KeyError:
        return jsonify({'error': f'Neighborhood {data["l3"]} not found in the rankings'}), 400
    # Return prediction, average price for the requested neighborhood, and its rank position
    return jsonify({
        'predicted_price': round(prediction[0]),
        'average_m2_price': round(average_m2_price),
        'neighborhood': data['l3'],
        'neighborhood_position': neighborhood_position
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print('Starting Flask API server port:', port)
    # Get the port from the environment (useful for deployment), otherwise default to 5000
    app.run(debug=True, host='0.0.0.0', port=port | 5000)

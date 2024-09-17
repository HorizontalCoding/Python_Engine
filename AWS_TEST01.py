# -*- coding: utf-8 -*-
#==========================================
# random_state 변경 버전(시드값)
#==========================================
"""Flask API for MBTI Travel Recommendations"""

import time
import threading
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
import csv

app = Flask(__name__)

# Global variables for caching
cached_model = None
model_lock = threading.Lock()  # Lock for model training

# Load data
def load_data():
    """Load data from CSV files."""
    df_travel_total = pd.read_csv('csv/df_travel_total_gwloro.csv')
    return df_travel_total

# Preprocess data
def preprocess_data(df):
    """Preprocess the data."""
    df = df[~df['TRAVEL_MISSION_CHECK'].isnull()].copy()
    df = df[~df['VISIT_AREA_TYPE_CD'].isnull()].copy()
    df['TRAVEL_MISSION_INT'] = df['TRAVEL_MISSION_CHECK'].str.split(';').str[0].astype(int)

    df_filter = df[[
        'MBTI',
        'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
        'VISIT_AREA_NM',
        'VISIT_AREA_TYPE_CD',
        'DGSTFN',
    ]]

    df_filter = df_filter.dropna()
    return df_filter

# Train model
def train_model(train_data):
    """Train the CatBoost model with time-based hyperparameters."""
    global cached_model
    with model_lock:  # Ensure that only one thread can access this block at a time
        if cached_model is not None:
            return cached_model

        categorical_features_names = [
            'MBTI',
            'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
            'VISIT_AREA_NM', 'VISIT_AREA_TYPE_CD'
        ]

        current_time = time.time()
        depth = 4 + int(current_time) % 5  # Values between 4 and 8
        n_estimators = 1000 + int(current_time * 1000000) % 501  # Values between 1000 and 1500

        train_pool = Pool(train_data.drop(['DGSTFN'], axis=1), label=train_data['DGSTFN'], cat_features=categorical_features_names)
        
        model = CatBoostRegressor(
            loss_function='RMSE',
            eval_metric='MAE',
            task_type='CPU',
            depth=depth,
            learning_rate=0.01,
            n_estimators=n_estimators,
            verbose=500,
            thread_count=2  # Utilize both vCPUs on c5.large
        )

        model.fit(
            train_pool,
            verbose=500
        )

        print(f"Model trained with depth={depth} and n_estimators={n_estimators}")

        cached_model = model
        return model

# Load MBTI data
def load_mbti_data(file_path):
    """Load MBTI data from CSV."""
    mbti_data_dict = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header
        for row in reader:
            mbti_data_dict[row[0].lower()] = row[1:]  # Convert MBTI key to lowercase
    return mbti_data_dict

# Predict results
def predict_results(model, df_filter, mbti, mbti_data_dict):
    """Generate predictions using the trained model."""
    area_names = df_filter[['VISIT_AREA_NM']].drop_duplicates()
    area_cd = df_filter[['VISIT_AREA_TYPE_CD']]
    area_info = pd.merge(area_names, area_cd, left_index=True, right_index=True)

    if mbti not in mbti_data_dict:
        return pd.DataFrame(columns=['VISIT_AREA_TYPE_CD', 'VISIT_AREA_NM', 'SCORE'])

    mbti_data = mbti_data_dict[mbti]
    
    traveler = {
        'MBTI': mbti.upper(),
        'TRAVEL_STYL_1': mbti_data[0],
        'TRAVEL_STYL_2': mbti_data[1],
        'TRAVEL_STYL_3': mbti_data[2],
        'TRAVEL_STYL_4': mbti_data[3],
        'TRAVEL_STYL_5': mbti_data[4],
        'TRAVEL_STYL_6': mbti_data[5],
        'TRAVEL_STYL_7': mbti_data[6],
        'TRAVEL_STYL_8': mbti_data[7],
    }

    results = pd.DataFrame(columns=['VISIT_AREA_TYPE_CD', 'VISIT_AREA_NM', 'SCORE'])

    for area, id in zip(area_info['VISIT_AREA_NM'], area_info['VISIT_AREA_TYPE_CD']):
        total_data = list(traveler.values())
        total_data.extend([area, id])

        score = model.predict([total_data])[0]

        new_result = pd.DataFrame([[id, area, score]], columns=['VISIT_AREA_TYPE_CD', 'VISIT_AREA_NM', 'SCORE'])
        results = pd.concat([results, new_result], ignore_index=True)

    results = results.sort_values('SCORE', ascending=False)
    return results

# Search for locations
def search_location(results_mrg, search_word):
    """Search for locations based on user input."""
    search_location = results_mrg[
        results_mrg['ROAD_NM_ADDR'].str.contains(search_word, na=False) |
        results_mrg['LOTNO_ADDR'].str.contains(search_word, na=False)
    ]

    search_location = search_location.drop_duplicates(subset=['VISIT_AREA_NM'])
    search_location = search_location.head(7)  # Limit to top 7 results
    return search_location

def clean_data_for_json(df):
    """Clean data to ensure JSON compatibility."""
    df = df.fillna('N/A')  # Replace NaN with a placeholder or empty string
    return df

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    """Endpoint for getting travel recommendations."""
    client_ip = request.remote_addr
    print(f"Client IP address: {client_ip}")

    data = request.json
    mbti = data.get('mbti', '').strip().lower()
    search_word = data.get('search_word', '').strip()

    df = load_data()
    df_filter = preprocess_data(df)

    current_time = time.time()
    random_state = int(current_time) % 1000  # Use seconds to create a random_state

    train_data, _ = train_test_split(df_filter, test_size=0.2, random_state=random_state)
    model = train_model(train_data)

    mbti_data_dict = load_mbti_data('csv/mbti_average_int.csv')
    results = predict_results(model, df_filter, mbti, mbti_data_dict)
    
    if results.empty:
        return jsonify({'error': 'No results found for the given MBTI.'})

    results_mrg = pd.merge(results, df, how='inner')

    search_location_df = search_location(results_mrg, search_word)

    # Clean data for JSON response
    search_location_df = clean_data_for_json(search_location_df)
    
    # Limit to top 3 results
    results_list = search_location_df.head(7).to_dict(orient='records')

    return jsonify(results_list)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)

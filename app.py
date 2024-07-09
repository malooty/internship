from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# RMSE Calculation Function
def calculate_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Route to Calculate Daily RMSE
@app.route('/calculate_daily_rmse', methods=['POST'])
def calculate_daily_rmse_route():
    forecast_files = request.files.getlist('forecast_files')
    actual_file = request.files.get('actual_file')
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    if not forecast_files or not actual_file:
        return jsonify({'error': 'Both forecast and actual files are required'}), 400

    try:
        actual_data = pd.read_csv(actual_file)
        
        # Convert the date columns from 'YYYYMMDD' to 'datetime'
        actual_data['date'] = pd.to_datetime(actual_data['date'], format='%Y%m%d', errors='coerce')

        if actual_data['date'].isnull().any():
            return jsonify({'error': 'Some dates in the actual file could not be converted. Please check the date format in the CSV.'}), 400

        # Convert the date columns to the desired format (dd/mm/yy)
        actual_data['date'] = actual_data['date'].dt.strftime('%d/%m/%y')

        # Filter actual data based on the provided date range
        start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date, format='%Y-%m-%d')
        actual_data['date'] = pd.to_datetime(actual_data['date'], format='%d/%m/%y')
        filtered_actual_data = actual_data[(actual_data['date'] >= start_date) & (actual_data['date'] <= end_date)]

        if filtered_actual_data.empty:
            return jsonify({'error': 'No actual data available in the given date range'}), 400

        # Initialize the result DataFrame
        result_df = pd.DataFrame()

        for file in forecast_files:
            forecast_data = pd.read_csv(file)
            
            # Convert the date columns from 'YYYYMMDD' to 'datetime'
            forecast_data['date'] = pd.to_datetime(forecast_data['date'], format='%Y%m%d', errors='coerce')

            if forecast_data['date'].isnull().any():
                return jsonify({'error': f'Some dates in the forecast file {file.filename} could not be converted. Please check the date format in the CSV.'}), 400

            # Convert the date columns to the desired format (dd/mm/yy)
            forecast_data['date'] = forecast_data['date'].dt.strftime('%d/%m/%y')

            # Filter forecast data based on the provided date range
            forecast_data['date'] = pd.to_datetime(forecast_data['date'], format='%d/%m/%y')
            filtered_forecast_data = forecast_data[(forecast_data['date'] >= start_date) & (forecast_data['date'] <= end_date)]

            if filtered_forecast_data.empty:
                return jsonify({'error': f'No forecast data available in the given date range for {file.filename}'}), 400

            # Merge the forecast and actual data on date
            merged_data = pd.merge(filtered_forecast_data, filtered_actual_data, on='date', suffixes=('_fcst', '_act'))

            if merged_data.empty:
                return jsonify({'error': f'No matching data found between forecast and actual files for {file.filename}'}), 400

            # Group by date and calculate RMSE for each group
            daily_rmse = merged_data.groupby(merged_data['date'].dt.strftime('%d/%m/%y')).apply(
                lambda x: calculate_rmse(x['price_fcst'], x['price_act'])
            )

            # Add the team's RMSE results to the result DataFrame
            team_name = file.filename.replace('.csv', '')
            team_df = daily_rmse.reset_index(name=team_name)
            if result_df.empty:
                result_df = team_df
            else:
                result_df = pd.merge(result_df, team_df, on='date', how='outer')

        # Convert the DataFrame to JSON
        result_json = result_df.to_json(orient='records')

        return jsonify(result_json), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to serve index.html
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)

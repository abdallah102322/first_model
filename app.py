"""
========================================================================================
ALGERIAN USED CAR VALUATION & DEAL CLASSIFIER WEB APPLICATION
========================================================================================
Flask production server integrating the Nested Cross-Validation CatBoost champion
pipeline (100,197 records).

Dual Operational Modes:
  1. Regression Mode: Fair Market Value Estimator & Negotiation Corridor
  2. Classification Mode: Deal Classifier (Is the Asking Price High, Low, or Fair?)
========================================================================================
"""

import os
import io
import uuid
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory, Response
from flask_cors import CORS

from car_valuation_service import valuation_service, CATALOG_PATH, BASE_DIR

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

TEMP_DIR = os.path.join(BASE_DIR, 'temp_uploads')
DIAGNOSTICS_DIR = os.path.join(BASE_DIR, 'ml_diagnostics')
REPORT_PATH = os.path.join(BASE_DIR, 'champion_model_report.json')

os.makedirs(TEMP_DIR, exist_ok=True)


@app.route('/')
def index():
    """Serves the unified dual-mode AI web dashboard."""
    return render_template('index.html')


@app.route('/api/catalog', methods=['GET'])
def get_catalog():
    """
    Returns the comprehensive catalog containing all 58 wilayas,
    brand-to-model hierarchies, fuel types, and transmissions.
    """
    try:
        catalog_data = valuation_service.catalog
        return jsonify({
            'status': 'success',
            'catalog': catalog_data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_price():
    """
    Mode 1 (Regression):
    Accepts vehicle specifications and predicts fair market valuation
    with empirical negotiation corridor and depreciation curves.
    """
    try:
        data = request.get_json(force=True) or {}
        if not data:
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400

        result = valuation_service.predict_fair_price(data)
        curves = valuation_service.generate_sensitivity_curve(data)
        result['curves'] = curves

        return jsonify({
            'status': 'success',
            'mode': 'regression',
            'data': result
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/classify_deal', methods=['POST'])
def classify_deal():
    """
    Mode 2 (Classification):
    Accepts vehicle specifications AND an asking price (Million Centimes or DZD).
    Returns classification: GREAT_DEAL, GOOD_DEAL, FAIR_PRICE, HIGH_PRICE, or VERY_HIGH,
    along with exact price gap, visual gauge score, and negotiation strategy.
    """
    try:
        data = request.get_json(force=True) or {}
        if not data:
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400

        asking_price = data.get('asking_price')
        if asking_price is None or str(asking_price).strip() == '':
            return jsonify({'status': 'error', 'message': 'Asking price is required for deal classification'}), 400

        try:
            asking_price = float(asking_price)
        except (ValueError, TypeError):
            return jsonify({'status': 'error', 'message': 'Asking price must be a valid number'}), 400

        price_unit = str(data.get('price_unit', 'million')).strip().lower()

        deal_result = valuation_service.classify_deal(data, asking_price=asking_price, price_unit=price_unit)
        curves = valuation_service.generate_sensitivity_curve(data)
        deal_result['curves'] = curves

        return jsonify({
            'status': 'success',
            'mode': 'classification',
            'data': deal_result
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/model-report', methods=['GET'])
def get_model_report():
    """
    Returns nested cross-validation benchmark results, feature importances,
    and diagnostic metadata.
    """
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            return jsonify({
                'status': 'success',
                'report': report_data
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    else:
        return jsonify({'status': 'error', 'message': 'Model report not found'}), 404


@app.route('/api/diagnostics-image/<filename>', methods=['GET'])
def get_diagnostics_image(filename):
    """Serves high-resolution diagnostic plots (residuals, learning curves, SHAP)."""
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(DIAGNOSTICS_DIR, safe_filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        return jsonify({'status': 'error', 'message': 'Image not found'}), 404


@app.route('/api/template-csv', methods=['GET'])
def template_csv():
    """Generates a CSV template compatible with batch valuation & classification."""
    sample_data = [
        {
            'brand': 'Renault', 'model': 'Clio 4', 'year': 2018,
            'mileage_km': 120000, 'fuel_type': 'Diesel', 'transmission': 'Manual',
            'wilaya_code': 16, 'asking_price': 320
        },
        {
            'brand': 'Volkswagen', 'model': 'Golf 7', 'year': 2019,
            'mileage_km': 95000, 'fuel_type': 'Diesel', 'transmission': 'Manual',
            'wilaya_code': 31, 'asking_price': 480
        },
        {
            'brand': 'Dacia', 'model': 'Duster', 'year': 2021,
            'mileage_km': 65000, 'fuel_type': 'Diesel', 'transmission': 'Manual',
            'wilaya_code': 19, 'asking_price': 410
        },
        {
            'brand': 'Kia', 'model': 'Picanto', 'year': 2020,
            'mileage_km': 50000, 'fuel_type': 'Petrol', 'transmission': 'Automatic',
            'wilaya_code': 25, 'asking_price': 275
        },
        {
            'brand': 'Geely', 'model': 'Coolray', 'year': 2024,
            'mileage_km': 12000, 'fuel_type': 'Petrol', 'transmission': 'Automatic',
            'wilaya_code': 16, 'asking_price': 430
        }
    ]
    df = pd.DataFrame(sample_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    return Response(
        csv_buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=algeria_cars_batch_template.csv"}
    )


@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """
    Processes an uploaded CSV batch file. Evaluates each record with fair market
    pricing and deal classification (if asking price is provided).
    """
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

        file = request.files['file']
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'status': 'error', 'message': 'File must be a valid .csv file'}), 400

        df = pd.read_csv(file)
        if len(df) == 0:
            return jsonify({'status': 'error', 'message': 'Uploaded CSV is empty'}), 400

        records = df.to_dict(orient='records')
        evaluated_records = valuation_service.evaluate_batch(records)
        df_out = pd.DataFrame(evaluated_records)

        file_id = str(uuid.uuid4())
        out_filename = f"evaluated_{file_id}.csv"
        out_filepath = os.path.join(TEMP_DIR, out_filename)
        df_out.to_csv(out_filepath, index=False)

        # Summary statistics
        fair_prices = df_out['fair_price_million'].dropna()
        stats = {
            'total_rows': len(df_out),
            'avg_fair_million': round(float(fair_prices.mean()), 1) if len(fair_prices) > 0 else 0,
            'min_fair_million': round(float(fair_prices.min()), 1) if len(fair_prices) > 0 else 0,
            'max_fair_million': round(float(fair_prices.max()), 1) if len(fair_prices) > 0 else 0,
            'great_deals_count': int((df_out['deal_rating_key'] == 'GREAT_DEAL').sum()) if 'deal_rating_key' in df_out else 0,
            'high_prices_count': int((df_out['deal_rating_key'].isin(['HIGH_PRICE', 'VERY_HIGH'])).sum()) if 'deal_rating_key' in df_out else 0
        }

        preview = df_out.head(10).to_dict(orient='records')

        return jsonify({
            'status': 'success',
            'file_id': file_id,
            'stats': stats,
            'preview': preview
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/download-evaluated/<file_id>', methods=['GET'])
def download_evaluated(file_id):
    """Downloads the evaluated batch CSV with fair prices and deal ratings."""
    try:
        filename = f"evaluated_{file_id}.csv"
        filepath = os.path.join(TEMP_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'status': 'error', 'message': 'File expired or not found'}), 404

        return send_file(
            filepath,
            mimetype="text/csv",
            as_attachment=True,
            download_name="algeria_cars_evaluated_market_prices.csv"
        )
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'pipeline_loaded': valuation_service.pipeline is not None,
        'brands_count': len(valuation_service.catalog.get('sorted_brands', []))
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Algerian Car Valuation & Deal Classifier Server on http://localhost:{port} ...")
    app.run(host='0.0.0.0', port=port, debug=False)

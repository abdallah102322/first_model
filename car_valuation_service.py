"""
========================================================================================
CAR VALUATION & DEAL CLASSIFICATION SERVICE
========================================================================================
Production service wrapping the champion pipeline for:
  1. Regression Mode: Fair Market Value Estimation, Negotiation Corridors, Sensitivity
  2. Classification Mode: Price Rating (Great Deal, Fair Price, Overpriced, Very High)
========================================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# Ensure feature engineer class is accessible during unpickling regardless of caller module
import train_algerian_car_price_model
if '__main__' in sys.modules:
    setattr(sys.modules['__main__'], 'AlgeriaCarFeatureEngineer', train_algerian_car_price_model.AlgeriaCarFeatureEngineer)
sys.modules['AlgeriaCarFeatureEngineer'] = train_algerian_car_price_model.AlgeriaCarFeatureEngineer

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, 'algerian_car_price_pipeline.joblib')
CATALOG_PATH = os.path.join(BASE_DIR, 'catalog.json')

class CarValuationService:
    def __init__(self, pipeline_path=PIPELINE_PATH, catalog_path=CATALOG_PATH):
        self.pipeline_path = pipeline_path
        self.catalog_path = catalog_path
        self.pipeline = None
        self.catalog = {}
        self.load()

    def load(self):
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(f"Pipeline artifact not found at: {self.pipeline_path}")
        
        self.pipeline = joblib.load(self.pipeline_path)
        
        if os.path.exists(self.catalog_path):
            with open(self.catalog_path, 'r', encoding='utf-8') as f:
                self.catalog = json.load(f)
        else:
            self.catalog = {
                'sorted_brands': ['Renault', 'Volkswagen', 'Peugeot', 'Geely', 'Kia', 'Hyundai', 'Seat', 'Dacia', 'Fiat', 'Livan'],
                'brand_models': {},
                'wilayas': {16: 'Alger', 31: 'Oran', 25: 'Constantine', 19: 'Sétif'},
                'fuel_types': ['Petrol', 'Diesel', 'LPG'],
                'transmissions': ['Manual', 'Automatic']
            }

    def predict_fair_price(self, car_dict: dict) -> dict:
        """
        Mode 1 (Regression): Predicts fair market price, negotiation range,
        and depreciation profile.
        """
        # Prepare single row matching raw feature schema
        brand = str(car_dict.get('brand', 'Renault')).strip().title() or 'Renault'
        model = str(car_dict.get('model', 'Clio 4')).strip().title() or 'Clio 4'
        
        try:
            year = int(car_dict.get('year', 2018))
        except (ValueError, TypeError):
            year = 2018

        try:
            mileage_km = float(car_dict.get('mileage_km', 120000))
        except (ValueError, TypeError):
            mileage_km = 120000.0

        # Standardize fuel type
        raw_fuel = str(car_dict.get('fuel_type', 'Diesel')).strip().lower()
        fuel_map = {
            'essence': 'Petrol', 'petrol': 'Petrol', 'essence / benzin': 'Petrol', 'gasoline': 'Petrol',
            'diesel': 'Diesel', 'mazout': 'Diesel',
            'gpl': 'LPG', 'lpg': 'LPG', 'gaz': 'LPG', 'gas': 'LPG',
            'hybride': 'Petrol', 'hybrid': 'Petrol',
            'electrique': 'Petrol', 'electric': 'Petrol'
        }
        fuel_type = fuel_map.get(raw_fuel, 'Petrol' if 'ess' in raw_fuel else ('Diesel' if 'dies' in raw_fuel else 'Petrol'))

        # Standardize transmission
        raw_trans = str(car_dict.get('transmission', 'Manual')).strip().lower()
        trans_map = {
            'manual': 'Manual', 'manuelle': 'Manual', 'manuel': 'Manual',
            'automatic': 'Automatic', 'automatique': 'Automatic', 'auto': 'Automatic'
        }
        transmission = trans_map.get(raw_trans, 'Automatic' if 'auto' in raw_trans else 'Manual')
        
        try:
            wilaya_code = int(car_dict.get('wilaya_code', 16))
        except (ValueError, TypeError):
            wilaya_code = 16

        df_input = pd.DataFrame([{
            'brand': brand,
            'model': model,
            'year': year,
            'mileage_km': mileage_km,
            'fuel_type': fuel_type,
            'transmission': transmission,
            'wilaya_code': wilaya_code
        }])

        pred_million = float(self.pipeline.predict(df_input)[0])
        pred_million = max(25.0, min(3000.0, pred_million))
        pred_dzd = pred_million * 10000.0

        # Empirical negotiation corridor: ±8% for everyday cars (200-500M), ±12% for budget/luxury
        margin_pct = 0.08 if (200.0 <= pred_million <= 500.0) else 0.12
        low_bound_million = round(pred_million * (1.0 - margin_pct), 2)
        high_bound_million = round(pred_million * (1.0 + margin_pct), 2)

        return {
            'brand': brand,
            'model': model,
            'year': year,
            'mileage_km': mileage_km,
            'fuel_type': fuel_type,
            'transmission': transmission,
            'wilaya_code': wilaya_code,
            'wilaya_name': self.catalog.get('wilayas', {}).get(str(wilaya_code), f"Wilaya {wilaya_code}"),
            'predicted_price_million_centimes': round(pred_million, 2),
            'predicted_price_dzd': round(pred_dzd, -3),
            'formatted_price_centimes': f"{round(pred_million, 2):.2f} Million Centimes",
            'formatted_price_dzd': f"{round(pred_dzd, -3):,.0f} DZD",
            'corridor': {
                'low_million_centimes': low_bound_million,
                'high_million_centimes': high_bound_million,
                'low_dzd': round(low_bound_million * 10000.0, -3),
                'high_dzd': round(high_bound_million * 10000.0, -3),
                'formatted_low_centimes': f"{low_bound_million:.2f} M",
                'formatted_high_centimes': f"{high_bound_million:.2f} M",
                'formatted_low_dzd': f"{round(low_bound_million * 10000.0, -3):,.0f} DZD",
                'formatted_high_dzd': f"{round(high_bound_million * 10000.0, -3):,.0f} DZD"
            }
        }

    def classify_deal(self, car_dict: dict, asking_price: float, price_unit: str = 'million') -> dict:
        """
        Mode 2 (Classification): Compares asking price against fair market valuation
        and classifies the deal as:
          - GREAT_DEAL (Asking <= -10% below market)
          - GOOD_DEAL  (Asking between -4% and -10% below market)
          - FAIR_PRICE (Asking within ±4% of market)
          - HIGH_PRICE (Asking between +4% and +12% above market)
          - VERY_HIGH  (Asking > +12% above market)
        """
        # Step 1: Predict fair valuation
        reg_result = self.predict_fair_price(car_dict)
        fair_million = reg_result['predicted_price_million_centimes']
        fair_dzd = reg_result['predicted_price_dzd']

        # Normalize asking price to Million Centimes & DZD
        if price_unit.lower() in ['dzd', 'dinars']:
            asking_dzd = float(asking_price)
            asking_million = asking_dzd / 10000.0
        else:
            asking_million = float(asking_price)
            asking_dzd = asking_million * 10000.0

        # Calculations
        diff_million = round(asking_million - fair_million, 2)
        diff_dzd = round(asking_dzd - fair_dzd, -3)
        diff_pct = round((diff_million / fair_million) * 100.0, 1)

        # Classification Logic
        if diff_pct <= -10.0:
            rating_key = "GREAT_DEAL"
            rating_title = "Great Deal (Affaire / Prix Très Bas)"
            rating_color = "#10b981"  # Emerald green
            gauge_score = 15          # 0-100 scale (low = cheap deal)
            verdict = "Very Attractive Price"
            explanation = (
                f"The asking price is {abs(diff_pct):.1f}% below fair market valuation. "
                f"You save approximately {abs(diff_million):.1f} Million Centimes ({abs(diff_dzd):,.0f} DZD). "
                "Ensure you inspect bodywork, chassis, and paperwork for hidden flaws."
            )
        elif -10.0 < diff_pct <= -4.0:
            rating_key = "GOOD_DEAL"
            rating_title = "Good Deal (Bonne Affaire)"
            rating_color = "#34d399"  # Light green
            gauge_score = 35
            verdict = "Competitive Price"
            explanation = (
                f"The vehicle is priced {abs(diff_pct):.1f}% below average market price. "
                f"Good negotiation margin: savings of roughly {abs(diff_million):.1f} Million Centimes."
            )
        elif -4.0 < diff_pct <= 4.0:
            rating_key = "FAIR_PRICE"
            rating_title = "Fair Market Price (Prix Équitable)"
            rating_color = "#3b82f6"  # Blue
            gauge_score = 50
            verdict = "Market Average"
            explanation = (
                f"The asking price matches prevailing Algerian market rates within a tight margin ({diff_pct:+.1f}%). "
                "Fair transaction price. Counter-offer with 3% to 5% discount."
            )
        elif 4.0 < diff_pct <= 12.0:
            rating_key = "HIGH_PRICE"
            rating_title = "High in Price (Prix Élevé / Au-Dessus)"
            rating_color = "#f59e0b"  # Amber
            gauge_score = 75
            verdict = "Slightly Overpriced"
            explanation = (
                f"The seller is asking {diff_pct:.1f}% (+{diff_million:.1f} M Centimes) above fair value. "
                "Unless the car has rare options or pristine zero-paint-touch condition, negotiate down."
            )
        else:
            rating_key = "VERY_HIGH"
            rating_title = "Very High (Ghali Bzaf / Très Élevé)"
            rating_color = "#ef4444"  # Red
            gauge_score = 95
            verdict = "Significantly Overpriced"
            explanation = (
                f"The vehicle is overpriced by {diff_pct:.1f}% (+{diff_million:.1f} Million Centimes / +{diff_dzd:,.0f} DZD). "
                "Substantially above market valuation. Recommendation: negotiate aggressively or look elsewhere."
            )

        return {
            'valuation': reg_result,
            'asking_price_million_centimes': round(asking_million, 2),
            'asking_price_dzd': round(asking_dzd, -3),
            'formatted_asking_centimes': f"{round(asking_million, 2):.2f} Million Centimes",
            'formatted_asking_dzd': f"{round(asking_dzd, -3):,.0f} DZD",
            'difference_million_centimes': diff_million,
            'difference_dzd': diff_dzd,
            'difference_pct': diff_pct,
            'is_high': diff_pct > 4.0,
            'is_low': diff_pct < -4.0,
            'is_fair': -4.0 <= diff_pct <= 4.0,
            'rating_key': rating_key,
            'rating_title': rating_title,
            'rating_color': rating_color,
            'gauge_score': gauge_score,
            'verdict': verdict,
            'explanation': explanation,
            'suggested_target_price_centimes': f"{fair_million:.2f} M",
            'suggested_target_price_dzd': f"{fair_dzd:,.0f} DZD"
        }

    def generate_sensitivity_curve(self, car_dict: dict) -> dict:
        """Generates age and mileage depreciation sensitivity curves."""
        base_year = int(car_dict.get('year', 2018))
        base_mileage = float(car_dict.get('mileage_km', 120000))
        
        # 1. Age curve (from year-5 to year+3)
        years = [y for y in range(max(1990, base_year - 5), min(2026, base_year + 4))]
        age_points = []
        for y in years:
            sim = car_dict.copy()
            sim['year'] = y
            res = self.predict_fair_price(sim)
            age_points.append({'year': y, 'price_million': res['predicted_price_million_centimes']})

        # 2. Mileage curve (from 0 to 300,000 km)
        mileages = [0, 30000, 60000, 100000, 150000, 200000, 250000, 300000]
        mileage_points = []
        for m in mileages:
            sim = car_dict.copy()
            sim['mileage_km'] = m
            res = self.predict_fair_price(sim)
            mileage_points.append({'mileage_km': m, 'price_million': res['predicted_price_million_centimes']})

        return {
            'age_curve': age_points,
            'mileage_curve': mileage_points
        }

    def evaluate_batch(self, records: list) -> list:
        """
        Processes a batch of car records. Predicts fair prices and,
        if an asking price is provided, performs deal classification.
        """
        evaluated = []
        for rec in records:
            item = dict(rec)
            try:
                fair = self.predict_fair_price(item)
                item['fair_price_million'] = fair['predicted_price_million_centimes']
                item['fair_price_dzd'] = fair['predicted_price_dzd']
                item['corridor_low_million'] = fair['corridor']['low_million_centimes']
                item['corridor_high_million'] = fair['corridor']['high_million_centimes']

                # Check if an asking price was provided
                asking = None
                for price_col in ['asking_price', 'price', 'price_million', 'asking_price_million']:
                    if price_col in item and item[price_col] not in [None, '', 'nan', np.nan]:
                        try:
                            asking = float(item[price_col])
                            break
                        except (ValueError, TypeError):
                            pass

                if asking is not None:
                    deal = self.classify_deal(item, asking_price=asking, price_unit='million' if asking < 50000 else 'dzd')
                    item['deal_rating'] = deal['rating_title']
                    item['deal_rating_key'] = deal['rating_key']
                    item['deal_color'] = deal['rating_color']
                    item['deal_verdict'] = deal['verdict']
                    item['difference_pct'] = deal['difference_pct']
                    item['difference_million'] = deal['difference_million_centimes']
                    item['is_high'] = deal['is_high']
                    item['is_low'] = deal['is_low']
                else:
                    item['deal_rating'] = 'N/A'
                    item['deal_rating_key'] = 'NONE'
                    item['deal_color'] = '#94a3b8'
                    item['deal_verdict'] = 'No asking price'
                    item['difference_pct'] = 0.0
                    item['difference_million'] = 0.0
                    item['is_high'] = False
                    item['is_low'] = False

            except Exception as e:
                item['error'] = str(e)
                item['fair_price_million'] = None
                item['deal_rating'] = 'Error'

            evaluated.append(item)
        return evaluated

# Global singleton
valuation_service = CarValuationService()

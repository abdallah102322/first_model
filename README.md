# 🚗 CoteDZ — Algerian Automotive Market Valuation & Deal Classifier

A production-grade machine learning system and full-stack web application for estimating fair market prices and classifying deals (*Great Deal* vs *Overpriced / Ghali*) across the Algerian automotive market.

Trained and evaluated on **100,197 verified vehicle listings** from Ouedkniss using a leak-free scikit-learn pipeline and 5-Fold Stratified Nested Cross-Validation with Optuna hyperparameter optimization.

---

## 🌟 Key Features

### 1. Dual Operational Modes
* **📈 Mode 1: Fair Market Value Estimator (Regression)**:
  * Predicts fair market price in **Million Centimes** and **Algerian Dinars (DZD)** ($1\text{ Million Centimes} = 10,000\text{ DZD}$).
  * Computes an empirical negotiation corridor ($\pm 8\%$ to $\pm 12\%$).
  * Generates interactive depreciation curves (residual price vs vehicle age and mileage).
* **🏷️ Mode 2: Deal Checker & Price Rater (Classification)**:
  * Compares seller's asking price against the fair market benchmark.
  * Evaluates deals across a 5-segment automotive meter:
    * 🟢 **Great Deal (Affaire en Or)**: $\le -10\%$ below market.
    * 🟢 **Good Deal (Bonne Affaire)**: $-10\%$ to $-4\%$.
    * 🔵 **Fair Price (Prix du Marché)**: Within $\pm 4\%$.
    * 🟠 **High Price (Prix Élevé)**: $+4\%$ to $+12\%$.
    * 🔴 **Overpriced (Ghali Bzaf)**: $> +12\%$ above market.
  * Provides actionable negotiation guidance tailored to the Algerian market.

### 2. High-Throughput Batch CSV Evaluation
* Upload full fleets or scraped CSV files.
* Automatically computes fair market prices and deal ratings.
* Export annotated CSV with a single click.

### 3. Machine Learning Excellence
* **Champion Model**: CatBoost Regressor with continuous target encoding and $\log(1 + y)$ target transformation.
* **Nested CV Benchmark**: **MAE 152.57 Million Centimes**, **MdAPE 18.50%**, **$R^2 = 0.894$** across 5 outer folds.

---

## 📁 Repository Structure

```
D:\database\
├── algeria_cars_200k_model_ready.csv          # Cleaned modeling corpus (100,197 listings)
├── scraped_ouedkniss_cars_200k_master.csv     # Raw master corpus (200,000 scraped listings)
├── catalog.json                               # Catalog of 58 wilayas, brands, and models
├── algerian_car_price_pipeline.joblib         # Champion ML pipeline artifact
├── train_algerian_car_price_model.py          # Data cleaning, nested CV benchmark & training script
├── car_valuation_service.py                   # Production inference & deal classification service
├── champion_model_report.json                 # Model metadata & 9-model CV benchmark scores
├── app.py                                     # Production Flask server (REST API & UI)
├── run_app.py                                 # Application launcher
├── run_web.bat                                # 1-click Windows batch launcher
├── templates/
│   └── index.html                             # Clean CoteDZ frontend (Tailwind CSS, Chart.js)
├── temp_uploads/                              # Temporary folder for CSV batch uploads
├── scrape_200k.py                             # High-throughput GraphQL scraper engine
├── clean_master_200k.py                       # Data engineering & standardization script
├── ml_diagnostics/                            # 11 High-resolution diagnostic plots & screenshots
├── generate_pdf_report.py                     # Automated PDF report compilation script
├── report_template.html                       # Base template for technical report
├── final_project_report.html                  # Standalone HTML report with embedded base64 plots
└── Algerian_Car_Valuation_Final_Project_Report.pdf  # Comprehensive 13-page academic/technical report
```

---

## 🚀 Quickstart

### 1. Launch the Web Application
Double-click `run_web.bat` or run:
```bash
python run_app.py
```
Open your browser at: **http://localhost:5000**

### 2. Retrain or Benchmark Models
To run the full leak-free Nested CV benchmark across all 9 models and generate diagnostics:
```bash
python train_algerian_car_price_model.py --sample_size 15000 --n_trials 20
```

### 3. Recompile the Final PDF Report
```bash
python generate_pdf_report.py
```

---

## 📡 REST API Documentation

### `POST /api/predict`
Accepts vehicle specifications and returns fair market value, negotiation corridors, and depreciation curves.
```json
{
  "brand": "Renault",
  "model": "Clio 4",
  "year": 2018,
  "mileage_km": 120000,
  "fuel_type": "Diesel",
  "transmission": "Manual",
  "wilaya_code": 16
}
```

### `POST /api/classify_deal`
Accepts vehicle specifications **and asking price**; returns deal tier, price gap, and negotiation advice.
```json
{
  "brand": "Volkswagen",
  "model": "Golf 7",
  "year": 2019,
  "mileage_km": 95000,
  "fuel_type": "Diesel",
  "transmission": "Manual",
  "wilaya_code": 31,
  "asking_price": 520,
  "price_unit": "million"
}
```

### `GET /api/catalog`
Returns available brands, models, and all 58 Algerian wilayas.

---

## 📄 Final PDF Report
The complete 13-page academic and technical project report is available at:
`Algerian_Car_Valuation_Final_Project_Report.pdf`

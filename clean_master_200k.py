import os
import re
import time
import json
import numpy as np
import pandas as pd

RAW_FILE = os.path.join(os.path.dirname(__file__), 'scraped_ouedkniss_cars_200k_master.csv')
CLEAN_ALL_FILE = os.path.join(os.path.dirname(__file__), 'algeria_cars_200k_cleaned_all.csv')
MODEL_READY_FILE = os.path.join(os.path.dirname(__file__), 'algeria_cars_200k_model_ready.csv')

def clean_200k_pipeline():
    start_time = time.time()
    print("=" * 70, flush=True)
    print("       🧹 CLEANING 210,000 VEHICLE MASTER CORPUS", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"Input file not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    initial_count = len(df)
    print(f"[*] Loaded raw master dataset: {initial_count:,} records", flush=True)

    # -----------------------------------------------------------------
    # 1. REMOVE NON-CAR & MOTORCYCLE LISTINGS
    # -----------------------------------------------------------------
    moto_brands = ['yamaha', 'sym', 'kymco', 'vespa', 'kawasaki', 'ducati', 'bmw motorrad']
    mask_brand = df['brand'].astype(str).str.lower().isin(moto_brands)

    ad_patterns = r'\b(tmax|sh150|sh300|xadv|x-adv|cbr|scooter|remorque|tracteur|dépôt-vente|depot vente|achat voiture|achat auto|voiture accidentée)\b'
    mask_title = df['title'].astype(str).str.lower().str.contains(ad_patterns, na=False, regex=True)

    mask_drop = mask_brand | mask_title
    df = df[~mask_drop].copy()
    print(f"[+] Step 1: Filtered {mask_drop.sum():,} non-car / motorcycle / ad records. ({len(df):,} remaining)", flush=True)

    # -----------------------------------------------------------------
    # 2. RESOLVE DELEGATED WILAYAS TO OFFICIAL WILAYA CODES (1-58)
    # -----------------------------------------------------------------
    delegated_wilayas = {
        523: 3,   # Laghouat
        524: 5,   # Batna
        525: 7,   # Biskra
        526: 12,  # Tébessa
        527: 13,  # Tlemcen
        528: 14,  # Tiaret
        529: 17,  # Djelfa
        530: 17,  # Djelfa
        531: 26,  # Médéa
        532: 28,  # M'Sila
        533: 32   # El Bayadh
    }
    
    # Map by code
    for d_code, target_wilaya in delegated_wilayas.items():
        df.loc[df['wilaya_code'] == d_code, 'wilaya_code'] = target_wilaya

    # Fallback any residual wilaya outside 1-58 to Alger (16)
    df.loc[(df['wilaya_code'] < 1) | (df['wilaya_code'] > 58), 'wilaya_code'] = 16
    df['wilaya_code'] = df['wilaya_code'].astype(int)
    print(f"[+] Step 2: Resolved all delegated districts to official wilayas (1-58).", flush=True)

    # -----------------------------------------------------------------
    # 3. REMAP ORPHAN MODEL NAMES LISTED AS BRANDS
    # -----------------------------------------------------------------
    orphan_models = {
        'picanto': ('Kia', 'Picanto'),
        'symbol': ('Renault', 'Symbol'),
        'clio': ('Renault', 'Clio'),
        'golf': ('Volkswagen', 'Golf'),
        'polo': ('Volkswagen', 'Polo'),
        'ibiza': ('Seat', 'Ibiza'),
        '208': ('Peugeot', '208'),
        '207': ('Peugeot', '207'),
        '308': ('Peugeot', '308'),
        '206': ('Peugeot', '206'),
        'accent': ('Hyundai', 'Accent'),
        'tucson': ('Hyundai', 'Tucson'),
        'duster': ('Dacia', 'Duster'),
        'stepway': ('Dacia', 'Stepway'),
        'sandero': ('Dacia', 'Sandero'),
        'logan': ('Dacia', 'Logan'),
        'yaris': ('Toyota', 'Yaris'),
        'hilux': ('Toyota', 'Hilux'),
        'corolla': ('Toyota', 'Corolla'),
        'spark': ('Chevrolet', 'Spark'),
        'sail': ('Chevrolet', 'Sail'),
        'maruti': ('Suzuki', 'Maruti'),
        'alto': ('Suzuki', 'Alto'),
        'swift': ('Suzuki', 'Swift'),
        'tiguan': ('Volkswagen', 'Tiguan'),
        'caddy': ('Volkswagen', 'Caddy'),
        'passat': ('Volkswagen', 'Passat'),
        'berlingo': ('Citroen', 'Berlingo'),
        'megane': ('Renault', 'Megane'),
        'kangoo': ('Renault', 'Kangoo'),
        'i10': ('Hyundai', 'i10'),
        'i20': ('Hyundai', 'i20'),
        'i30': ('Hyundai', 'i30'),
        'sportage': ('Kia', 'Sportage'),
        'leon': ('Seat', 'Leon'),
        'octavia': ('Skoda', 'Octavia'),
        'fabia': ('Skoda', 'Fabia')
    }

    remapped_count = 0
    for m_lower, (proper_brand, proper_model) in orphan_models.items():
        mask_orphan = df['brand'].astype(str).str.lower() == m_lower
        if mask_orphan.sum() > 0:
            remapped_count += mask_orphan.sum()
            df.loc[mask_orphan, 'brand'] = proper_brand
            # If model was missing or nan, set it to proper_model
            mask_need_model = mask_orphan & (df['model'].isna() | (df['model'].astype(str).str.lower() == 'nan'))
            df.loc[mask_need_model, 'model'] = proper_model
    print(f"[+] Step 3: Remapped {remapped_count:,} orphan model titles to correct Brand & Model.", flush=True)

    # -----------------------------------------------------------------
    # 4. BRAND CANONICALIZATION & UNIFICATION
    # -----------------------------------------------------------------
    brand_mapping = {
        'mercedes-benz': 'Mercedes', 'mercedes': 'Mercedes',
        'volkswagen': 'Volkswagen', 'vw': 'Volkswagen',
        'gac': 'GAC', 'roewe': 'Roewe', 'jetta': 'Jetta',
        'land rover': 'Land Rover', 'rover': 'Land Rover',
        'alfa romeo': 'Alfa Romeo', 'alfa': 'Alfa Romeo',
        'great wall': 'Great Wall', 'dfsk': 'DFSK', 'mg': 'MG',
        'ds': 'DS', 'jac': 'JAC', 'byd': 'BYD', 'bmw': 'BMW',
        'chery': 'Chery', 'geely': 'Geely', 'livan': 'Livan',
        'jetour': 'Jetour', 'baic': 'Baic', 'changan': 'Changan',
        'hyundai': 'Hyundai', 'kia': 'Kia', 'toyota': 'Toyota',
        'nissan': 'Nissan', 'renault': 'Renault', 'peugeot': 'Peugeot',
        'fiat': 'Fiat', 'dacia': 'Dacia', 'seat': 'Seat', 'skoda': 'Skoda',
        'audi': 'Audi', 'ford': 'Ford', 'chevrolet': 'Chevrolet',
        'citroen': 'Citroen', 'opel': 'Opel', 'suzuki': 'Suzuki',
        'mitsubishi': 'Mitsubishi', 'honda': 'Honda', 'mazda': 'Mazda',
        'jeep': 'Jeep', 'porsche': 'Porsche', 'cupra': 'Cupra',
        'subaru': 'Subaru', 'isuzu': 'Isuzu', 'haval': 'Haval', 'kaiyi': 'Kaiyi',
        'bestune': 'Bestune', 'lifan': 'Lifan', 'chana': 'Chana', 'foton': 'Foton'
    }

    def clean_brand(b):
        if not b or str(b).lower() in ['nan', 'none', '', 'null']:
            return 'Renault'
        b_str = str(b).strip()
        return brand_mapping.get(b_str.lower(), b_str.title())

    df['brand'] = df['brand'].apply(clean_brand)
    print(f"[+] Step 4: Canonicalized brands ({df['brand'].nunique():,} unique brands).", flush=True)

    # -----------------------------------------------------------------
    # 5. MODEL NORMALIZATION
    # -----------------------------------------------------------------
    def clean_model(m):
        if not m or str(m).lower() in ['nan', 'none', '', 'null']:
            return 'nan'
        s = str(m).strip()
        s = re.sub(r'[\/\\\'"`,;]', '', s).strip()
        if not s or s.lower() == 'nan':
            return 'nan'
        return s.title() if len(s) > 2 else s.upper()

    df['model'] = df['model'].apply(clean_model)
    df['model_missing'] = (df['model'] == 'nan').astype(int)
    print(f"[+] Step 5: Normalized models ({df['model_missing'].sum():,} missing models flagged).", flush=True)

    # -----------------------------------------------------------------
    # 6. MILEAGE VALIDATION & OUTLIER CAPPING
    # -----------------------------------------------------------------
    def clean_mileage(row):
        km = row['mileage_km']
        miss = row['mileage_missing']
        if pd.isna(km) or miss == 1:
            return np.nan, 1
        try:
            val = float(km)
            if val < 0:
                val = 0.0
            elif val > 650000:
                val = 650000.0
            return val, 0
        except:
            return np.nan, 1

    res_km = df.apply(clean_mileage, axis=1)
    df['mileage_km'] = [r[0] for r in res_km]
    df['mileage_missing'] = [r[1] for r in res_km]
    print(f"[+] Step 6: Validated mileages ({df['mileage_missing'].sum():,} missing flags).", flush=True)

    # -----------------------------------------------------------------
    # 7. FUEL & TRANSMISSION NORMALIZATION
    # -----------------------------------------------------------------
    def clean_fuel(f):
        if not f or str(f).lower() in ['nan', 'none', '', 'null']:
            return 'nan', 1
        t = str(f).lower()
        if 'ess' in t or 'petrol' in t:
            return 'Petrol', 0
        elif 'dies' in t or 'maz' in t or 'gasoil' in t:
            return 'Diesel', 0
        elif 'gpl' in t or 'lpg' in t or 'gaz' in t:
            return 'LPG', 0
        return 'Petrol', 0

    res_fuel = df['fuel_type'].apply(clean_fuel)
    df['fuel_type'] = [r[0] for r in res_fuel]
    df['fuel_missing'] = [r[1] for r in res_fuel]

    def clean_transmission(t):
        if not t or str(t).lower() in ['nan', 'none', '', 'null']:
            return 'nan', 1
        s = str(t).lower()
        if 'man' in s:
            return 'Manual', 0
        elif 'aut' in s:
            return 'Automatic', 0
        return 'nan', 1

    res_trans = df['transmission'].apply(clean_transmission)
    df['transmission'] = [r[0] for r in res_trans]
    df['transmission_missing'] = [r[1] for r in res_trans]
    print(f"[+] Step 7: Normalized fuel types and transmissions.", flush=True)

    # -----------------------------------------------------------------
    # 8. DEDUPLICATION
    # -----------------------------------------------------------------
    # Primary: by unique record_id
    df = df.drop_duplicates(subset=['record_id']).copy()

    # Secondary: by exact vehicle attribute duplicate postings
    df_clean_all = df.drop_duplicates(
        subset=['brand', 'model', 'year', 'wilaya_code', 'city', 'mileage_km', 'price_dzd']
    ).copy()
    duplicates_removed = len(df) - len(df_clean_all)
    print(f"[+] Step 8: Removed {duplicates_removed:,} duplicate vehicle postings. ({len(df_clean_all):,} unique listings)", flush=True)

    # Save complete clean catalog
    df_clean_all.to_csv(CLEAN_ALL_FILE, index=False, encoding='utf-8')
    size_all = os.path.getsize(CLEAN_ALL_FILE) / (1024 * 1024)
    print(f"[✔] SAVED COMPLETE CLEAN CATALOG: {CLEAN_ALL_FILE} ({len(df_clean_all):,} rows, {size_all:.1f} MB)", flush=True)

    # -----------------------------------------------------------------
    # 9. EXTRACT MODEL-READY TRAINING DATASET (CLEAN VALID PRICES)
    # -----------------------------------------------------------------
    valid_price_mask = (
        df_clean_all['price_million_centimes'].notna() &
        (df_clean_all['price_million_centimes'] >= 25.0) &
        (df_clean_all['price_million_centimes'] <= 3000.0)
    )
    df_model_ready = df_clean_all[valid_price_mask].copy()
    df_model_ready['price_dzd'] = (df_model_ready['price_million_centimes'] * 10000.0).astype(int)

    # 15 standard model columns
    target_columns = [
        'record_id', 'year', 'brand', 'model', 'wilaya_code', 'city',
        'mileage_km', 'mileage_missing', 'fuel_type', 'fuel_missing',
        'transmission', 'transmission_missing', 'model_missing',
        'price_dzd', 'price_million_centimes'
    ]
    df_model_ready = df_model_ready[target_columns].copy()
    df_model_ready.to_csv(MODEL_READY_FILE, index=False, encoding='utf-8')
    size_ready = os.path.getsize(MODEL_READY_FILE) / (1024 * 1024)

    total_time = round(time.time() - start_time, 1)
    print(f"[✔] SAVED MODEL-READY DATASET: {MODEL_READY_FILE} ({len(df_model_ready):,} rows, {size_ready:.1f} MB)", flush=True)
    print(f"[*] Price Range : {df_model_ready['price_million_centimes'].min():.1f}M to {df_model_ready['price_million_centimes'].max():.1f}M Centimes", flush=True)
    print(f"[*] Median Price: {df_model_ready['price_million_centimes'].median():.1f}M Centimes ({df_model_ready['price_dzd'].median():,.0f} DZD)", flush=True)
    print(f"[*] Mean Price  : {df_model_ready['price_million_centimes'].mean():.1f}M Centimes ({df_model_ready['price_dzd'].mean():,.0f} DZD)", flush=True)
    print("=" * 70, flush=True)
    print(f"[+] Total Cleaning Duration: {total_time} seconds!", flush=True)
    print("=" * 70, flush=True)

if __name__ == '__main__':
    clean_200k_pipeline()

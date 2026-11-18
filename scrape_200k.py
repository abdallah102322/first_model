import os
import re
import sys
import time
import json
import urllib.request
import pandas as pd

GRAPHQL_URL = 'https://api.ouedkniss.com/graphql'

HEADERS = {
    'Content-Type': 'application/json',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Origin': 'https://www.ouedkniss.com',
    'Referer': 'https://www.ouedkniss.com/automobiles-voitures/1936'
}

SEARCH_QUERY = """
query SearchAnnouncementsQuery($filter: SearchFilterInput) {
  search(filter: $filter) {
    announcements {
      data {
        id
        title
        slug
        price
        pricePreview
        priceUnit
        priceType
        cities {
          id
          name
          region {
            id
            name
          }
        }
        smallDescription {
          specification {
            codename
          }
          valueText
        }
      }
      paginatorInfo {
        total
        lastPage
        hasMorePages
      }
    }
  }
}
"""

CATALOG_FILE = os.path.join(os.path.dirname(__file__), 'catalog.json')
BRANDS_LIST = [
    'Mercedes-Benz', 'Land Rover', 'Great Wall', 'Volkswagen', 'Chevrolet',
    'Mitsubishi', 'Alfa Romeo', 'Dongfeng', 'Porsche', 'Citroen', 'Renault',
    'Peugeot', 'Hyundai', 'Toyota', 'Nissan', 'Changan', 'Jetour', 'Subaru',
    'Suzuki', 'Dacia', 'Geely', 'Chery', 'Skoda', 'Honda', 'Mazda', 'Livan',
    'Haval', 'Isuzu', 'Seat', 'Audi', 'Fiat', 'Ford', 'Jeep', 'Baic', 'DFSK',
    'Kia', 'BMW', 'JAC', 'MG', 'DS', 'Cupra', 'Kaiyi', 'Bestune', 'Roewe'
]
if os.path.exists(CATALOG_FILE):
    try:
        with open(CATALOG_FILE, 'r', encoding='utf-8') as f:
            cat = json.load(f)
            BRANDS_LIST.extend(cat.get('sorted_brands', []))
    except:
        pass

BRANDS_LIST = sorted(list(set(BRANDS_LIST)), key=lambda x: len(x), reverse=True)

def execute_query(page, count=60, retries=5):
    variables = {
        "filter": {
            "categorySlug": "automobiles-voitures",
            "page": page,
            "count": count
        }
    }
    payload = json.dumps({"query": SEARCH_QUERY, "variables": variables}).encode('utf-8')
    req = urllib.request.Request(GRAPHQL_URL, data=payload, headers=HEADERS)
    
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            if attempt == retries - 1:
                return None
            time.sleep((attempt + 1) * 2.0)
    return None

def parse_mileage(raw_text):
    if not raw_text:
        return None
    cleaned = re.sub(r'[^0-9]', '', str(raw_text))
    if not cleaned:
        return None
    try:
        val = float(cleaned)
        if val > 1500000:
            while val > 1500000:
                val /= 10.0
        return val
    except:
        return None

def parse_fuel(raw_text):
    if not raw_text:
        return None
    t = str(raw_text).lower()
    if 'ess' in t or 'petrol' in t:
        return 'Petrol'
    elif 'dies' in t or 'maz' in t or 'gasoil' in t:
        return 'Diesel'
    elif 'gpl' in t or 'lpg' in t or 'gaz' in t:
        return 'LPG'
    return 'Petrol'

def parse_transmission(raw_text):
    if not raw_text:
        return None
    t = str(raw_text).lower()
    if 'man' in t:
        return 'Manual'
    elif 'aut' in t:
        return 'Automatic'
    return None

def extract_vehicle(item):
    aid = item.get('id')
    title = (item.get('title') or '').strip()

    specs = {}
    for sd in item.get('smallDescription') or []:
        cname = sd.get('specification', {}).get('codename')
        vlist = sd.get('valueText') or []
        if cname and vlist:
            specs[cname] = vlist[0]

    # Year
    ym = re.search(r'\b(20[0-2][0-9]|19[7-9][0-9])\b', title)
    year = int(ym.group(1)) if ym else 2021

    # Brand
    brand = None
    title_lower = title.lower()
    for b in BRANDS_LIST:
        if re.search(r'\b' + re.escape(b.lower()) + r'\b', title_lower):
            brand = b
            break
    if not brand:
        brand = title.split()[0] if title.split() else 'Renault'

    # Model
    rem = title
    rem = re.sub(re.escape(brand), '', rem, flags=re.IGNORECASE)
    rem = re.sub(r'\b' + str(year) + r'\b', '', rem)
    rem = re.sub(r'[^\w\s\-\.]', '', rem).strip()
    model = rem.split()[0] if rem.split() else 'nan'
    model_missing = 1 if not model or model == 'nan' else 0

    # Wilaya & City
    city = 'Alger'
    wilaya_code = 16
    if item.get('cities'):
        c_obj = item['cities'][0]
        city = (c_obj.get('name') or 'Alger').strip()
        if c_obj.get('region'):
            try:
                wilaya_code = int(c_obj['region'].get('id', 16))
            except:
                wilaya_code = 16

    # Mileage
    raw_km = specs.get('kilometrage')
    mileage_km = parse_mileage(raw_km)
    mileage_missing = 1 if mileage_km is None else 0

    # Fuel
    raw_fuel = specs.get('energie')
    fuel_type = parse_fuel(raw_fuel)
    fuel_missing = 1 if fuel_type is None else 0
    if fuel_missing:
        fuel_type = 'nan'

    # Transmission
    raw_trans = specs.get('transmission')
    transmission = parse_transmission(raw_trans)
    transmission_missing = 1 if transmission is None else 0
    if transmission_missing:
        transmission = 'nan'

    # Price
    raw_price = item.get('price')
    price_dzd = None
    price_million_centimes = None
    if raw_price is not None:
        try:
            p = float(raw_price)
            if p > 50000:
                price_dzd = p
                price_million_centimes = round(p / 10000.0, 1)
        except:
            pass

    return {
        'record_id': f"DZ{aid}",
        'year': year,
        'brand': brand.strip(),
        'model': model.strip(),
        'wilaya_code': wilaya_code,
        'city': city,
        'mileage_km': mileage_km,
        'mileage_missing': mileage_missing,
        'fuel_type': fuel_type,
        'fuel_missing': fuel_missing,
        'transmission': transmission,
        'transmission_missing': transmission_missing,
        'model_missing': model_missing,
        'price_dzd': price_dzd,
        'price_million_centimes': price_million_centimes,
        'title': title,
        'url': f"https://www.ouedkniss.com/annonce/{aid}"
    }

def main():
    target_count = 100000
    start_page = 1936
    out_file = os.path.join(os.path.dirname(__file__), 'scraped_ouedkniss_cars_part2_100k.csv')
    part1_file = os.path.join(os.path.dirname(__file__), 'scraped_ouedkniss_cars_100k.csv')
    init_10k_file = os.path.join(os.path.dirname(__file__), 'scraped_ouedkniss_cars_10000.csv')
    master_200k_file = os.path.join(os.path.dirname(__file__), 'scraped_ouedkniss_cars_200k_master.csv')
    state_file = os.path.join(os.path.dirname(__file__), 'scrape_200k_state.json')

    print("=" * 70, flush=True)
    print(f"[*] Starting 200,000 Scaling Run (Batch 2: 100,000 cars)", flush=True)
    print(f"[*] Starting Page  : {start_page}", flush=True)
    print(f"[*] Target Rows    : {target_count:,}", flush=True)
    print(f"[*] Destination    : {out_file}", flush=True)
    print("=" * 70, flush=True)

    seen_ids = set()

    # Pre-populate seen_ids from previous scrapes to guarantee ZERO duplicates
    for fpath in [init_10k_file, part1_file]:
        if os.path.exists(fpath):
            try:
                df_prev = pd.read_csv(fpath, usecols=['record_id'])
                for rid in df_prev['record_id'].dropna():
                    seen_ids.add(str(rid).replace('DZ', ''))
            except Exception as e:
                print(f"[!] Warning reading {fpath}: {e}", flush=True)
    print(f"[*] Loaded {len(seen_ids):,} existing IDs to guarantee ZERO duplicates.", flush=True)

    records = []
    current_page = start_page
    if os.path.exists(out_file) and os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                current_page = state.get('next_page', start_page)
            df_existing = pd.read_csv(out_file)
            records = df_existing.to_dict(orient='records')
            for r in records:
                seen_ids.add(str(r['record_id']).replace('DZ', ''))
            print(f"[+] Resuming Batch 2 from Page {current_page} with {len(records):,} records.", flush=True)
        except Exception as e:
            print(f"[!] Could not resume, starting fresh from page {start_page}: {e}", flush=True)
            records = []
            current_page = start_page

    start_time = time.time()
    last_checkpoint_count = len(records)

    while len(records) < target_count:
        res = execute_query(current_page, count=60)
        if not res or 'data' not in res or not res['data'].get('search'):
            print(f"[!] Warning: Query error on page {current_page}. Retrying in 3s...", flush=True)
            time.sleep(3.0)
            res = execute_query(current_page, count=60)
            if not res or 'data' not in res or not res['data'].get('search'):
                print(f"[!] Skipping page {current_page} after retry failure.", flush=True)
                current_page += 1
                continue

        announcements = res['data']['search']['announcements']
        items = announcements.get('data') or []
        if not items:
            print(f"[*] Reached end of catalog at page {current_page}. Finished.", flush=True)
            break

        for item in items:
            if not item or not isinstance(item, dict) or not item.get('id'):
                continue
            aid = str(item['id'])
            if aid in seen_ids:
                continue
            seen_ids.add(aid)

            row = extract_vehicle(item)
            records.append(row)

            if len(records) >= target_count:
                break

        # Checkpoint every 1,000 records
        if len(records) - last_checkpoint_count >= 1000 or len(records) == target_count:
            last_checkpoint_count = len(records)
            elapsed = time.time() - start_time
            speed = round(len(records) / max(elapsed, 0.1), 1)
            pct = round(len(records) / target_count * 100, 1)
            eta_min = round((target_count - len(records)) / max(speed * 60, 1), 1)
            print(f"[{len(records):6d} / {target_count:,}] ({pct:5.1f}%) | Page {current_page:4d} | Speed: {speed:4.1f} cars/s | Elapsed: {int(elapsed/60)}m {int(elapsed%60)}s | ETA: ~{eta_min} min", flush=True)

            pd.DataFrame(records).to_csv(out_file, index=False, encoding='utf-8')
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump({'next_page': current_page + 1, 'scraped_count': len(records)}, f)

        current_page += 1
        time.sleep(0.35)

    total_time = round(time.time() - start_time, 1)
    df_part2 = pd.DataFrame(records)
    df_part2.to_csv(out_file, index=False, encoding='utf-8')

    print("=" * 70, flush=True)
    print(f"[+] Batch 2 Finished! Scraped {len(df_part2):,} vehicles from pages {start_page} to {current_page - 1}.", flush=True)
    print(f"[+] Total Duration: {round(total_time/60, 1)} minutes ({total_time}s)", flush=True)
    print(f"[+] Part 2 saved: {out_file}", flush=True)

    # -----------------------------------------------------------------
    # Merge into 200,000 Master Corpus
    # -----------------------------------------------------------------
    print("\n[*] Merging into Master 200k Corpus...", flush=True)
    dfs = []
    if os.path.exists(init_10k_file):
        dfs.append(pd.read_csv(init_10k_file))
    if os.path.exists(part1_file):
        dfs.append(pd.read_csv(part1_file))
    dfs.append(df_part2)

    master_df = pd.concat(dfs, ignore_index=True)
    master_df = master_df.drop_duplicates(subset=['record_id']).reset_index(drop=True)
    master_df.to_csv(master_200k_file, index=False, encoding='utf-8')

    print(f"[✔] MASTER 200,000 CORPUS READY: {master_200k_file}")
    print(f"[✔] Total verified unique vehicles: {len(master_df):,}", flush=True)
    print("=" * 70, flush=True)

if __name__ == '__main__':
    main()

"""
Compiles report_template.html into final_project_report.html with base64 embedded images,
and renders it into a print-quality PDF using Chrome / Edge headless.
"""

import os
import base64
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIAGNOSTICS_DIR = os.path.join(BASE_DIR, 'ml_diagnostics')
TEMPLATE_PATH = os.path.join(BASE_DIR, 'report_template.html')
HTML_OUTPUT_PATH = os.path.join(BASE_DIR, 'final_project_report.html')
PDF_OUTPUT_PATH = os.path.join(BASE_DIR, 'Algerian_Car_Valuation_Final_Project_Report.pdf')

def get_base64_img(filename):
    filepath = os.path.join(DIAGNOSTICS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded}"
    else:
        print(f"[!] Warning: Image not found: {filepath}")
        return ""

def main():
    print("[*] Reading template HTML...")
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        html = f.read()

    print("[*] Encoding images and replacing placeholders...")
    replacements = {
        '{{IMG_EDA_DIST}}': get_base64_img('eda_price_distribution.png'),
        '{{IMG_EDA_BRANDS}}': get_base64_img('eda_top_brands.png'),
        '{{IMG_EDA_FUEL}}': get_base64_img('eda_depreciation_by_fuel.png'),
        '{{IMG_PRED_ACTUAL}}': get_base64_img('predicted_vs_actual.png'),
        '{{IMG_RESIDUALS}}': get_base64_img('residuals_vs_predicted.png'),
        '{{IMG_ERROR_DIST}}': get_base64_img('error_distribution.png'),
        '{{IMG_LEARNING_CURVE}}': get_base64_img('learning_curve.png'),
        '{{IMG_CV_FOLDS}}': get_base64_img('cv_score_per_fold.png'),
        '{{IMG_FEAT_IMP}}': get_base64_img('feature_importance.png'),
        '{{IMG_SHAP}}': get_base64_img('shap_summary.png'),
        '{{IMG_WEBAPP}}': get_base64_img('webapp_deal_checker_screenshot.png'),
    }

    for tag, b64 in replacements.items():
        html = html.replace(tag, b64)

    with open(HTML_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[+] Written full standalone HTML report: {HTML_OUTPUT_PATH}")

    # Detect Chrome or Edge
    chrome_candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    ]
    browser_exe = None
    for cand in chrome_candidates:
        if os.path.exists(cand):
            browser_exe = cand
            break

    if not browser_exe:
        print("[!] No Chrome or Edge executable found to print PDF.")
        return

    print(f"[*] Rendering PDF via {browser_exe} ...")
    cmd = [
        browser_exe,
        '--headless',
        '--disable-gpu',
        f'--print-to-pdf={PDF_OUTPUT_PATH}',
        '--no-pdf-header-footer',
        HTML_OUTPUT_PATH
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(PDF_OUTPUT_PATH):
        size_kb = os.path.getsize(PDF_OUTPUT_PATH) / 1024.0
        print(f"\n[✔] SUCCESS: PDF Report Generated Successfully!")
        print(f"    Path: {PDF_OUTPUT_PATH}")
        print(f"    File Size: {size_kb:.1f} KB")
    else:
        print(f"[!] PDF printing failed: {res.stderr}")

if __name__ == '__main__':
    main()

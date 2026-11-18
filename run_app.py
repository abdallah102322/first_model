import os
import sys
import time
import socket
import webbrowser
import threading
from app import app

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def open_browser(port):
    time.sleep(1.2)
    url = f"http://localhost:{port}"
    print(f"\n[+] Opening browser at: {url}")
    webbrowser.open(url)

if __name__ == '__main__':
    port = 5000
    if is_port_open(port):
        port = 5001

    print("=" * 65)
    print("      🚗 SayaraPrice AI - Used Cars Valuation Model")
    print(f"      Running on: http://localhost:{port}")
    print("=" * 65)

    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    app.run(host='0.0.0.0', port=port, debug=False)

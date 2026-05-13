import http.server
import socketserver
import threading
import subprocess
import sys
import os

PORT = 7860

def run_agi():
    print("--- Starting Veylon AGI Backend (No-Input Mode) ---")
    # Setting env var to tell the script to skip any input() calls if they exist
    os.environ['PYTHONUNBUFFERED'] = '1'
    subprocess.run([sys.executable, 'veylon_agi_v5.py'], input=b'exit\n')

class HealthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Veylon AGI is Running')

if __name__ == '__main__':
    threading.Thread(target=run_agi, daemon=True).start()
    print(f"--- Serving on port {PORT} ---")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('0.0.0.0', PORT), HealthHandler) as httpd:
        httpd.serve_forever()
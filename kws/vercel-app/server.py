"""
Local dev server that mimics Vercel's serverless environment.
Run: python server.py
Then open http://localhost:8080
"""
import os
import sys
import json
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Add parent to path so api module works
sys.path.insert(0, os.path.dirname(__file__))

from api.predict import _decode_audio, _extract_mfcc, _get_session, LABELS
import numpy as np


class KWSHandler(SimpleHTTPRequestHandler):
    """Serve static files from public/ and handle /api/predict."""

    def __init__(self, *args, **kwargs):
        self.directory = os.path.join(os.path.dirname(__file__), 'public')
        super().__init__(*args, directory=self.directory, **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path.startswith('/api/predict'):
            self._handle_predict()
        else:
            self.send_error(404)

    def _handle_predict(self):
        try:
            content_type = self.headers.get('Content-Type', '')
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)

            # Parse multipart
            audio_bytes = self._parse_multipart(body, content_type)
            if audio_bytes is None:
                self._json_response(400, {'error': 'No audio file found'})
                return

            # Decode audio
            audio = _decode_audio(audio_bytes)

            # Extract features
            mfcc = _extract_mfcc(audio)

            # Run inference
            session = _get_session()
            input_tensor = mfcc[np.newaxis, np.newaxis, :, :]
            input_name = session.get_inputs()[0].name

            t0 = time.time()
            outputs = session.run(None, {input_name: input_tensor})
            inference_ms = round((time.time() - t0) * 1000, 1)

            logits = outputs[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            pred_idx = int(np.argmax(probs))
            result = {
                'keyword': LABELS[pred_idx],
                'confidence': round(float(probs[pred_idx]), 4),
                'probabilities': {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))},
                'inference_ms': inference_ms,
            }

            self._json_response(200, result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json_response(500, {'error': str(e)})

    def _json_response(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _parse_multipart(self, body, content_type):
        for part in content_type.split(';'):
            part = part.strip()
            if part.startswith('boundary='):
                boundary = part[9:].strip('"')
                break
        else:
            return None

        boundary_bytes = boundary.encode()
        parts = body.split(b'--' + boundary_bytes)

        for part in parts:
            if b'filename=' in part:
                header_end = part.find(b'\r\n\r\n')
                if header_end == -1:
                    header_end = part.find(b'\n\n')
                    if header_end == -1:
                        continue
                    content = part[header_end + 2:]
                else:
                    content = part[header_end + 4:]

                if content.endswith(b'\r\n'):
                    content = content[:-2]
                if content.endswith(b'--\r\n'):
                    content = content[:-4]
                if content.endswith(b'--'):
                    content = content[:-2]
                if content.endswith(b'\r\n'):
                    content = content[:-2]

                return content
        return None

    def log_message(self, format, *args):
        if '/api/' in str(args[0]) if args else False:
            print(f"[API] {args[0]}")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), KWSHandler)
    print(f"\n  Emergency Keyword Spotter")
    print(f"  Local server: http://localhost:{port}")
    print(f"  Model: {os.path.abspath(os.path.join(os.path.dirname(__file__), 'model', 'kws_model.onnx'))}")
    print(f"  Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()

"""
Comparison Landing Page - Choose between Traditional and VLM demos
"""

import sys
from pathlib import Path
from flask import Flask, render_template

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = Flask(__name__)

@app.route('/')
def index():
    """Render the comparison landing page."""
    return render_template('compare.html')

if __name__ == '__main__':
    print("🍳 Starting Cookify Demo Comparison Server")
    print("📊 Compare Traditional vs VLM-Enhanced Processing")
    print("🌐 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)


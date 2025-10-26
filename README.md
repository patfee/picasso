# Crane Curve Viewer (Dash + Flask)

Loads `data/height.csv` and `data/outreach.csv` (matrices where **columns are Main-jib angles** and **rows are Folding-jib angles**).  
Interpolates to in-between angles and plots points plus the outer envelope.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py

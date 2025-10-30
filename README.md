# Crane Curve Viewer (Dash + Flask)

Loads `data/height.csv` and `data/outreach.csv` (matrices where **columns are Main-jib angles** and **rows are Folding-jib angles**).  
Interpolates to in-between angles and plots points plus the outer envelope.

app/ app.py # creates Dash and registers pages 
shared/ 
    data.py # load_matrix_csv, build_interpolators, helpers 
    geom.py # boundary/alpha shape utils 
pages/ 
    envelope.py # layout + callbacks for envelope 
    harbour.py # layout + callbacks for harbour 
    tts_data.py # layout + callbacks for TTS downloads 
    test.py

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py

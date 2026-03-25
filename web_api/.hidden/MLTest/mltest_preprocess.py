import joblib
from pathlib import Path

base_dir = Path(__file__).parent
preprocess_input = joblib.load(base_dir / 'preprocessing.pkl')

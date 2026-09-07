import joblib
from pathlib import Path

base_dir = Path(__file__).parent
lin_reg = joblib.load(base_dir / 'model.pkl')

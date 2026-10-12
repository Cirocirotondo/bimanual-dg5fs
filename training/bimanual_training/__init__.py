"""Reinforcement-learning training for the bimanual DG-5F-S Twin (Isaac Lab + Newton)."""
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = TRAINING_DIR.parent                      # bimanual-dg5fs
ASSETS_DIR = ROOT_DIR / "assets"                     # Limb URDFs (docs/adr/0001)
USD_DIR = TRAINING_DIR / "assets" / "usd"
REFERENCES_DIR = TRAINING_DIR / "references"
LOGS_DIR = TRAINING_DIR / "logs"

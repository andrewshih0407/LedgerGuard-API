# LedgerGuard — ML Backend

Ensemble anomaly detection engine (Isolation Forest + Autoencoder + LightGBM).
Trained models are included in models_saved/.

## Run the API server
pip install -r requirements-server.txt
python -m uvicorn src.ledgerguard.api.server:app --reload --port 8000

## Run inference on a CSV
python scripts/infer.py --csv sample_data/demo_transactions.csv --model-dir models_saved/demo

## Retrain from scratch
python scripts/train.py --csv sample_data/demo_transactions.csv --save-dir models_saved/demo

## Performance (creditcard benchmark dataset)
F1: 0.8925  |  Precision: 95.4%  |  Recall: 83.8%  |  AUC-ROC: 0.980

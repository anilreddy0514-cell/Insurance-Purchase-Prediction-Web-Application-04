# CoverSignal Insurance Purchase Predictor

A local demonstration app that trains a Logistic Regression model and serves interactive insurance purchase estimates through FastAPI.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_data.py
python train_model.py
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The interactive API docs are at `/docs`, and `/api/health` reports whether the model loaded.

The data generator creates `data/insurance_purchases.csv` with 5,000 reproducible synthetic rows. The training script evaluates a held-out split and saves the preprocessing-plus-classifier pipeline as `model.pkl`. The interface accepts a customer profile and displays the model's estimated purchase probability and its 50% classification.

## Input ranges

Age: 18-100; annual income: $1,000-$2,000,000; annual premium: $100-$50,000; vehicle age: 0-50 years. Region must be Central, Coastal, Mountain, or Metro. The API independently validates all fields.

## Important

The sample data is generated for software demonstration only. It does not represent real policyholder behavior, and predictions must not be used for underwriting, pricing, or customer decisions. Pickled scikit-learn models should only be loaded from trusted sources.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the architecture, data-generation method, model pipeline, API contract, and frontend request flow.

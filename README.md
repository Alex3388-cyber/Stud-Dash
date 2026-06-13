# Student Performance Prediction Dashboard

A modular Streamlit dashboard starter for exploring student data, training machine learning models, and making performance predictions.

## Project Structure

```text
StudentPP_Dashboard/
|-- app.py                         # Main Streamlit entry point
|-- requirements.txt               # Python dependencies
|-- .gitignore                     # Files and folders Git should ignore
|-- assets/                        # Static UI assets such as CSS, images, icons, and branding
|-- database/                      # SQLite schema, connection helpers, and local database files
|-- datasets/                      # Raw and processed datasets used by the dashboard and ML pipeline
|-- models/                        # Model training, prediction helpers, and saved model artifacts
|-- pages/                         # Streamlit multipage dashboard screens
|-- preprocessing/                 # Data loading, validation, cleaning, and feature engineering
|-- utilities/                     # Shared constants and helper functions
`-- visualizations/                # Plotly charts and dashboard visualization helpers
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Keep raw source files in `datasets/raw/` and cleaned modeling data in `datasets/processed/`.
- Saved model files belong in `models/artifacts/`.
- SQLite database files belong in `database/` and are ignored by Git by default.
- Add new dashboard pages to `pages/` using Streamlit's multipage naming convention.

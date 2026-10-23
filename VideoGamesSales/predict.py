"""Command-line prediction for the Video Game Sales project.

Usage:
    python predict.py --platform PS2 --genre Action --publisher "Electronic Arts" --year 2005
"""
import argparse
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "Models/video_game_sales_model.joblib"


def predict_game(platform, genre, publisher, year, model_path=MODEL_PATH):
    b = joblib.load(model_path)
    if platform not in b["known_platforms"]:
        raise ValueError(f"Unknown platform '{platform}'. Choose from: {b['known_platforms']}")
    if genre not in b["known_genres"]:
        raise ValueError(f"Unknown genre '{genre}'. Choose from: {b['known_genres']}")

    pub = publisher if publisher in b["top_publishers"] else "Other"
    row = pd.DataFrame([{"Platform": platform, "Year": year, "Genre": genre, "Publisher_Grouped": pub}])
    row = pd.get_dummies(row, columns=["Platform", "Genre", "Publisher_Grouped"])
    row = row.reindex(columns=b["feature_columns"], fill_value=0)

    cls = int(b["classifier"].predict(row)[0])
    proba = b["classifier"].predict_proba(row)[0]
    sales = max(0.01, float(np.expm1(b["regressor"].predict(row)[0])))
    return {
        "predicted_category": b["class_names"][cls],
        "probabilities": {n: round(float(p), 3) for n, p in zip(b["class_names"], proba)},
        "predicted_global_sales_millions": round(sales, 2),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict video game sales category and global sales")
    ap.add_argument("--platform", required=True)
    ap.add_argument("--genre", required=True)
    ap.add_argument("--publisher", required=True)
    ap.add_argument("--year", type=int, required=True)
    a = ap.parse_args()
    r = predict_game(a.platform, a.genre, a.publisher, a.year)
    print(f"Category : {r['predicted_category']}")
    print(f"Probabilities: {r['probabilities']}")
    print(f"Predicted global sales: ~{r['predicted_global_sales_millions']} million units")

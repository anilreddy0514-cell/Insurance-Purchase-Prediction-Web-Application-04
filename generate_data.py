"""Create a reproducible synthetic insurance-purchase training dataset."""

import csv
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "insurance_purchases.csv"
ROW_COUNT = 5_000
REGION_EFFECTS = {"Central": 0.12, "Coastal": 0.2, "Mountain": -0.12, "Metro": -0.05}


def generate_data(path: Path = DATA_PATH, row_count: int = ROW_COUNT) -> None:
    rng = random.Random(20261013)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as dataset_file:
        writer = csv.DictWriter(
            dataset_file,
            fieldnames=[
                "age",
                "annual_income",
                "annual_premium",
                "vehicle_age_years",
                "previously_insured",
                "vehicle_damage",
                "region",
                "purchased",
            ],
        )
        writer.writeheader()

        for _ in range(row_count):
            age = rng.randint(18, 79)
            income = round(min(max(rng.gauss(68_000, 26_000), 18_000), 180_000) / 500) * 500
            premium = round(min(max(rng.gauss(950, 320), 250), 2_500) / 10) * 10
            vehicle_age = min(round(rng.expovariate(1 / 6)), 22)
            previously_insured = int(rng.random() < 0.42)
            damage_chance = min(0.62, 0.26 + vehicle_age * 0.012)
            vehicle_damage = int(rng.random() < damage_chance)
            region = rng.choice(tuple(REGION_EFFECTS))

            log_odds = (
                -1.25
                + 0.015 * (age - 40)
                + 0.000009 * (income - 68_000)
                - 0.00055 * (premium - 950)
                + 0.06 * (vehicle_age - 6)
                - 0.95 * previously_insured
                + 0.9 * vehicle_damage
                + REGION_EFFECTS[region]
            )
            purchase_probability = 1 / (1 + math.exp(-log_odds))

            writer.writerow(
                {
                    "age": age,
                    "annual_income": income,
                    "annual_premium": premium,
                    "vehicle_age_years": vehicle_age,
                    "previously_insured": previously_insured,
                    "vehicle_damage": vehicle_damage,
                    "region": region,
                    "purchased": int(rng.random() < purchase_probability),
                }
            )

    print(f"Generated {row_count:,} synthetic records at {path}")


if __name__ == "__main__":
    generate_data()
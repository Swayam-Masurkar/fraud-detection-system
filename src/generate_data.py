"""
Generates a synthetic multimodal insurance-claims dataset:
- structured numeric/categorical fields
- free-text narratives with per-row variation (no repeated templates,
  so the text signal is realistic instead of a leakage lookup table)
- synthetic claim "photos" written to disk, with visual complexity that
  loosely (not deterministically) correlates with claim severity

Run:
    python src/generate_data.py --n 5000 --seed 42
"""
import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd

INCIDENTS = ["storm", "fire", "flood", "hail", "collision", "theft"]
LOCATIONS = ["urban", "suburban", "rural", "coastal", "mountain"]

INCIDENT_PHRASES = {
    "storm": "Storm damaged the roof and water entered the house",
    "fire": "A fire started in the kitchen from an electrical fault",
    "flood": "The basement flooded during heavy rainfall",
    "hail": "Hail damaged the hood and windscreen of the vehicle",
    "collision": "The vehicle was struck by another car at an intersection",
    "theft": "Personal property was reported stolen from the premises",
}

RED_FLAG_PHRASES = [
    "requesting full cash payout immediately",
    "no photos available yet but total loss is claimed",
    "similar claim was filed last year for the same item",
    "contractor invoice has not been provided",
    "witness could not be reached for a statement",
    "policy was purchased only a few weeks before the incident",
    "requesting settlement without an adjuster visit",
    "documentation is inconsistent with the reported date",
]

NEUTRAL_PHRASES = [
    "photos and a contractor estimate are attached",
    "the adjuster has already visited the site",
    "a police report was filed the same day",
    "repair receipts are included with this claim",
    "this is the first claim filed under this policy",
    "witnesses confirmed the sequence of events",
    "damage was moderate and repair is already underway",
    "all requested documents have been submitted",
]


def make_narrative(rng: random.Random, incident: str, suspicious: bool) -> str:
    base = INCIDENT_PHRASES[incident]
    amount_mention = rng.choice(
        [
            "",
            f" Estimated repair cost is around ${rng.randint(500, 20000):,}.",
            "",
        ]
    )
    pool = RED_FLAG_PHRASES if suspicious else NEUTRAL_PHRASES
    n_clauses = rng.choice([1, 1, 2])
    clauses = rng.sample(pool, k=n_clauses)
    tail = " ".join(c[0].upper() + c[1:] + "." for c in clauses)
    return f"{base}.{amount_mention} {tail}"


def make_synthetic_image(path: str, rng: np.random.Generator, complexity: int):
    """Writes a small procedurally generated 'claim photo' to disk.
    `complexity` (roughly 0-10) loosely drives how much visual damage/edges
    the image shows, standing in for real claim photography."""
    img = np.full((128, 128, 3), 235, dtype=np.uint8)
    noise = rng.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    n_shapes = max(1, int(rng.poisson(complexity + 1)))
    for _ in range(n_shapes):
        pt1 = (int(rng.integers(0, 128)), int(rng.integers(0, 128)))
        pt2 = (int(rng.integers(0, 128)), int(rng.integers(0, 128)))
        color = tuple(int(c) for c in rng.integers(30, 150, size=3))
        thickness = int(rng.integers(1, 4))
        if rng.random() < 0.5:
            cv2.line(img, pt1, pt2, color, thickness)
        else:
            cv2.rectangle(img, pt1, pt2, color, thickness)

    cv2.imwrite(path, img)


def generate(n: int, seed: int, out_dir: str) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    img_dir = os.path.join(out_dir, "image_folder")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    for i in range(n):
        incident = rng.choice(INCIDENTS)
        location = rng.choice(LOCATIONS)
        claim_amount = float(np.round(np_rng.lognormal(mean=8.2, sigma=0.9), 2))
        claim_amount = min(claim_amount, 60000.0)
        num_prev_claims = int(np_rng.poisson(0.4))
        days_since_policy = int(np_rng.integers(10, 3000))
        disaster_nearby = int(np_rng.random() < 0.15)
        num_payments = int(np_rng.integers(1, 6))
        payment_irregularity = int(np_rng.random() < 0.12)

        # Latent fraud score: noisy, not a deterministic rule
        early_policy_big_claim = (days_since_policy < 90) and (claim_amount > 8000)
        score = (
            1.8 * early_policy_big_claim
            + 1.4 * payment_irregularity
            + 0.5 * min(num_prev_claims, 3)
            - 0.9 * disaster_nearby
            + 0.6 * (claim_amount > 15000)
            + np_rng.normal(0, 1.1)  # noise so it's not perfectly separable
        )
        fraud_prob = 1 / (1 + np.exp(-(score - 3.4)))
        is_fraud = int(np_rng.random() < fraud_prob)

        # text is *loosely* correlated with the label, with label noise,
        # so TF-IDF carries signal without being a perfect lookup
        suspicious_wording = (np_rng.random() < (0.75 if is_fraud else 0.15))
        narrative = make_narrative(rng, incident, suspicious_wording)

        # image complexity loosely tied to claimed severity, not to fraud directly
        complexity = 2 + claim_amount / 8000 + np_rng.normal(0, 1.0)
        img_path = os.path.join(img_dir, f"img_{i}.png")
        make_synthetic_image(img_path, np_rng, complexity=max(0, complexity))

        rows.append(
            dict(
                claim_amount=claim_amount,
                num_prev_claims=num_prev_claims,
                days_since_policy=days_since_policy,
                location=location,
                disaster_nearby=disaster_nearby,
                num_payments=num_payments,
                payment_irregularity=payment_irregularity,
                narrative=narrative,
                is_fraud=is_fraud,
                image_path=img_path,
            )
        )

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data")
    args = parser.parse_args()

    df = generate(args.n, args.seed, args.out)
    csv_path = os.path.join(args.out, "insurance_claims.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")
    print(f"Fraud rate: {df['is_fraud'].mean():.3%}")
    print(f"Unique narratives: {df['narrative'].nunique()} / {len(df)}")

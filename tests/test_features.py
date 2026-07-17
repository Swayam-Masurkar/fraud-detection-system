import pandas as pd
import pytest

from src.features import LABEL, build_preprocessor, ImageFeatureExtractor


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        [
            dict(
                claim_amount=5000.0,
                num_prev_claims=1,
                days_since_policy=200,
                location="urban",
                disaster_nearby=0,
                num_payments=2,
                payment_irregularity=0,
                narrative="Storm damaged the roof. Photos are attached.",
                image_path="data/image_folder/img_0.png",
                is_fraud=0,
            ),
            dict(
                claim_amount=15000.0,
                num_prev_claims=3,
                days_since_policy=20,
                location="rural",
                disaster_nearby=0,
                num_payments=1,
                payment_irregularity=1,
                narrative="Total loss claimed, requesting immediate cash payout.",
                image_path="data/image_folder/img_1.png",
                is_fraud=1,
            ),
        ]
    )


def test_preprocessor_output_shape(sample_df):
    X = sample_df.drop(columns=[LABEL])
    y = sample_df[LABEL]
    pre = build_preprocessor()
    Xt = pre.fit_transform(X, y)
    assert Xt.shape[0] == 2
    assert Xt.shape[1] > 0


def test_image_feature_extractor_handles_missing_file():
    df = pd.DataFrame({"image_path": ["does/not/exist.png"]})
    extractor = ImageFeatureExtractor()
    feats = extractor.transform(df)
    assert feats.shape == (1, ImageFeatureExtractor.N_FEATURES)
    assert (feats == 0).all()


def test_preprocessor_handles_unseen_location(sample_df):
    X = sample_df.drop(columns=[LABEL])
    y = sample_df[LABEL]
    pre = build_preprocessor()
    pre.fit(X, y)
    new_row = X.iloc[[0]].copy()
    new_row["location"] = "unseen_location"
    # should not raise, thanks to handle_unknown="ignore"
    pre.transform(new_row)

import numpy as np
import pandas as pd

from src.pipeline import fuse


def test_zscore_is_centered_and_scaled():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    z = fuse.zscore(x)
    assert np.isclose(z.mean(), 0.0, atol=1e-7)
    assert np.isclose(z.std(), 1.0, atol=1e-6)


def test_fuse_main_writes_expected_fusion(tmp_path, monkeypatch):
    split = "testing"
    scene = "unit_scene"

    root = tmp_path / split
    (root / "frames").mkdir(parents=True)
    (root / "audio").mkdir(parents=True)
    (root / "verbal").mkdir(parents=True)

    visual_df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "smoothed_score": [0.2, 1.2, 0.8],
        }
    )
    audio_df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "audio_score": [0.1, 0.6, 0.1],
        }
    )
    verbal_df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "verbal_score": [0.0, 1.0, 0.0],
        }
    )

    visual_df.to_csv(root / "frames" / f"{scene}_frames.csv", index=False)
    audio_df.to_csv(root / "audio" / f"{scene}_audio.csv", index=False)
    verbal_df.to_csv(root / "verbal" / f"{scene}_verbal.csv", index=False)

    monkeypatch.setattr(fuse, "SCORES_ROOT", tmp_path)
    monkeypatch.setattr(fuse, "SCENE", scene)
    monkeypatch.setattr(fuse, "DATASET_SPLIT", split)
    monkeypatch.setenv("VISUAL_ZSCORE", "1")

    fuse.main()

    out = pd.read_csv(root / "fused" / f"{scene}_fused.csv")
    expected = np.maximum.reduce(
        [out["visual_z"].values, out["audio_z"].values, out["verbal_z"].values]
    )
    assert np.allclose(out["fused_score"].values, expected)

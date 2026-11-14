"""A stream model trained with the loader's transform, evaluated on a voxel view (RESUM_FLEX)."""

import json

import numpy as np
import pytest

from legend_optical_map_ml.config import OpticalDataConfig, StreamTransformConfig
from legend_optical_map_ml.features import decode_absolute_phi, stream_features
from legend_optical_map_ml.photon_store import build
from legend_optical_map_ml.pipeline import ensure_prepared_optical_data
from tests.test_optical_pipeline import dataset
from tests.test_photon_store import chunk, store_config

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")

PHI = {
    "vertex": {"representation": "absolute", "coordinates": "cylindrical"},
    "momentum": {"representation": "components", "coordinates": "cylindrical"},
}
CYLINDRICAL = {"rotational_symmetry": 6, "angle_weighting": "radius"}
TRANSFORM = {"cylindrical": CYLINDRICAL, "phi": PHI}
# θ stays in the batches (cartesian, 3 columns = dim_theta): RESUM_FLEX needs it present for a
# stream model, which never uses it. phi = the stream's transformed columns.
VIEW = {"theta": {"cylindrical": CYLINDRICAL}, "phi": PHI}


def test_stream_model_with_transform_predicts_the_same_on_a_voxel_view(tmp_path):
    from core.surrogates.inference import open_for_prediction
    from core.surrogates.stream_training import run_stream, stream_logits
    from schemas.surrogates import StreamRunConfig

    # Photon store (homogeneous photons) and a voxel campaign with the matching view.
    for i in range(12):
        chunk(tmp_path / "box" / "results", i)
    store = store_config(tmp_path)
    build(store, workers=1)
    voxels = dataset(tmp_path / "voxels")
    voxels = OpticalDataConfig.model_validate({**voxels.model_dump(), "views": {"cyl": VIEW}})
    ensure_prepared_optical_data(voxels)
    view = voxels.view_directory("cyl")

    config = StreamRunConfig.model_validate({
        "model": {"kind": "mlp", "architecture": {"hidden_dims": [16, 16]}},
        "training": {"batch_size": 64, "n_passes": 1, "eval_every": 2, "learning_rate": 3e-3,
                     "device": "cpu", "validation_events": None},
        "event_store": str(store.store),
        "event_source": "legend_optical_map_ml.photon_loader:PhotonStream",
        "event_source_options": {"transform": TRANSFORM, "block_photons": 7,
                                 "buffer_photons": 20},
        "data_provider": "legend_optical_map_ml.provider:OpticalDataset",
        "data_directory": str(view),
        "evaluation": [{"split": "validation", "partition": "validation", "fidelity": "lf"}],
        "history_design_points": 3,
        "dim_theta": 3,
        "dim_phi": 7,
        "output_directory": str(tmp_path / "run"),
    })
    run_stream(config)
    rows = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert {r["fidelity"] for r in rows} >= {"stream", "lf"}

    # Predict on the view: RESUM_FLEX maps the view's phi columns into the stream units.
    model, provider, _ = open_for_prediction(tmp_path / "run" / "checkpoints" / "best", view)
    batch = provider.load("validation", "lf")[0]
    logits = np.asarray(model.predict(batch).logits, dtype=float).reshape(batch.phi.shape[:2])
    # The same photons as stream records: raw positions and momenta, transformed as the
    # loader does, label column irrelevant.
    norm = provider.normalization()["phi"]
    raw = batch.phi * np.asarray(norm["scale"]) + np.asarray(norm["offset"])  # [B, N, 7]
    valid = np.arange(raw.shape[1]) < np.asarray(batch.events_per_design_point)[:, None]
    centres = np.repeat(provider.coordinates("validation", "lf"), raw.shape[1], axis=0)
    positions, momenta = decode_absolute_phi(
        raw.reshape(-1, raw.shape[2]), provider.view.config, near=centres
    )
    photons = np.column_stack([positions, momenta, np.zeros(len(positions))]).astype(np.float32)
    features = stream_features(photons, StreamTransformConfig.model_validate(TRANSFORM))[0]
    records = np.column_stack([features, np.zeros((len(features), 1), np.float32)])
    direct = stream_logits(model, torch.as_tensor(records)).double().numpy()
    assert valid.sum() > 10 and np.std(direct[valid.ravel()]) > 1e-3  # a real comparison
    np.testing.assert_allclose(logits[valid], direct.reshape(logits.shape)[valid], atol=1e-4)

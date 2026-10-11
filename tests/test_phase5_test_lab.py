"""
Unit and integration tests for Phase 5 Test Lab & Model Evaluation Subsystems.
Verifies:
1. Sample picker catalog and selection (/api/test_lab/samples, /api/test_lab/select_sample)
2. File upload validation (rejects invalid extension and oversized files)
3. Temporary upload auto-cleanup mechanism
4. Test Lab query execution on loaded media with latency breakdown
5. Model Evaluation report loaded dynamically from experiment_02_summary.json (nothing hard-coded)
6. Mode switching between 'walk' and 'test_lab'
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import io
import pytest
from fastapi.testclient import TestClient

from ui.backend.app_server import app
from ui.backend.test_lab_manager import get_test_lab_manager, UPLOAD_DIR, MAX_FILE_SIZE_BYTES


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_samples_catalog(client):
    """Verifies that sample picker returns all 4 required test assets."""
    res = client.get("/api/test_lab/samples")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"

    samples = data["samples"]
    filenames = [s["filename"] for s in samples]

    assert "test_scene.jpg" in filenames
    assert "door_test.png" in filenames
    assert "ocr_test.png" in filenames
    assert "test_video.mp4" in filenames

    # Verify all samples exist on disk
    for s in samples:
        assert s["available"] is True
        assert s["size_bytes"] > 0


def test_sample_selection(client):
    """Verifies selecting samples sets active media and creates detection overlays."""
    # Select image sample
    res = client.post("/api/test_lab/select_sample", json={"filename": "door_test.png"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    media = data["media"]
    assert media["name"] == "door_test.png"
    assert media["type"] == "image"
    assert len(media["detections"]) > 0

    # Verify door was detected
    det_objects = [d["object"].lower() for d in media["detections"]]
    assert "door" in det_objects

    # Select video sample
    res_vid = client.post("/api/test_lab/select_sample", json={"filename": "test_video.mp4"})
    assert res_vid.status_code == 200
    data_vid = res_vid.json()
    assert data_vid["status"] == "success"
    assert data_vid["media"]["type"] == "video"


def test_upload_validation_and_cleanup(client):
    """Verifies upload validates file type/size and cleans temporary files."""
    mgr = get_test_lab_manager()

    # 1. Reject invalid file extension (.txt)
    txt_file = io.BytesIO(b"Hello world invalid content")
    res_invalid = client.post(
        "/api/test_lab/upload",
        files={"file": ("malicious_payload.exe", txt_file, "application/octet-stream")}
    )
    assert res_invalid.status_code == 400
    assert "Unsupported file type" in res_invalid.json()["error"]

    # 2. Reject oversized file (> 50MB)
    # Simulate oversized byte check
    oversized_bytes = b"0" * (MAX_FILE_SIZE_BYTES + 1024)
    with pytest.raises(ValueError, match="exceeds maximum limit"):
        mgr.handle_upload(oversized_bytes, "huge_video.mp4")

    # 3. Successful valid image upload and temp-file auto cleanup
    img_content = b"\xFF\xD8\xFF\xE0\x00\x10JFIF" + b"\x00" * 200  # Minimal mock JPEG header
    res_valid1 = client.post(
        "/api/test_lab/upload",
        files={"file": ("temp_upload_1.jpg", io.BytesIO(img_content), "image/jpeg")}
    )
    assert res_valid1.status_code == 200
    assert os.path.exists(os.path.join(UPLOAD_DIR, "temp_upload_1.jpg"))

    # Upload second file: must automatically clean up temp_upload_1.jpg
    res_valid2 = client.post(
        "/api/test_lab/upload",
        files={"file": ("temp_upload_2.jpg", io.BytesIO(img_content), "image/jpeg")}
    )
    assert res_valid2.status_code == 200
    assert os.path.exists(os.path.join(UPLOAD_DIR, "temp_upload_2.jpg"))
    assert not os.path.exists(os.path.join(UPLOAD_DIR, "temp_upload_1.jpg"))

    # Cleanup afterwards
    mgr.cleanup_temp_uploads()
    assert len(os.listdir(UPLOAD_DIR)) == 0


def test_query_loaded_media_with_pipeline_trace(client):
    """Verifies executing a query on loaded media returns measured stage latencies."""
    # Ensure door_test.png is active
    client.post("/api/test_lab/select_sample", json={"filename": "door_test.png"})

    res = client.post("/api/test_lab/query", json={"query": "where is the door"})
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    assert "door" in data["response"].lower()
    assert data["time_to_answer_ms"] > 0

    # Verify stage latencies
    latencies = data["stage_latencies"]
    assert "nlp_ms" in latencies
    assert "yolo_ms" in latencies
    assert "spatial_ms" in latencies
    assert latencies["yolo_ms"] > 0


def test_dynamic_evaluation_summary(client):
    """Verifies /api/evaluation dynamically loads experiment_02_summary.json without hardcoding."""
    res = client.get("/api/evaluation")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"

    report = data["data"]
    assert "experiment" in report
    assert "Experiment 02" in report["experiment"]

    # Verify overall metrics
    overall = report["overall_results"]
    assert overall["precision"] == 0.614
    assert overall["recall"] == 0.4853
    assert overall["mAP50"] == 0.5147
    assert overall["mAP50_95"] == 0.3629

    # Verify door detection metrics
    door = report["door_results"]
    assert door["precision"] == 0.862
    assert door["recall"] == 0.971
    assert door["mAP50"] == 0.98
    assert door["mAP50_95"] == 0.851

    # Verify comparison against Experiment 01
    comp = report["comparison_with_experiment_01"]
    assert comp["experiment_01_mAP50"] == 0.504
    assert comp["experiment_02_mAP50"] == 0.5147

    # Verify CNN architectures portfolio dynamically discovered
    cnn_archs = report.get("cnn_architectures_implemented", [])
    assert len(cnn_archs) >= 5
    assert "VGG" in cnn_archs
    assert "RESNET" in cnn_archs


def test_mode_switching(client):
    """Verifies /api/set_mode properly switches between 'walk' and 'test_lab'."""
    res_lab = client.post("/api/set_mode", json={"mode": "test_lab"})
    assert res_lab.status_code == 200
    assert res_lab.json()["mode"] == "test_lab"

    res_walk = client.post("/api/set_mode", json={"mode": "walk"})
    assert res_walk.status_code == 200
    assert res_walk.json()["mode"] == "walk"

    # Reject invalid mode
    res_bad = client.post("/api/set_mode", json={"mode": "invalid_mode"})
    assert res_bad.status_code == 400


if __name__ == "__main__":
    pytest.main(["-v", __file__])

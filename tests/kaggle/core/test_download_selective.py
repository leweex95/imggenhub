import os
import shutil
import tempfile
from pathlib import Path
import pytest
from imggenhub.kaggle.core import download_selective

def create_fake_image_files(dest_path, count=3):
    dest_path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (dest_path / f"img_{i}.png").write_bytes(b"fakeimg")

def create_fake_non_image_files(dest_path):
    (dest_path / "cli_command.txt").write_text("command")
    (dest_path / "stable-diffusion-batch-generator.log").write_text("log")
    (dest_path / "README.md").write_text("not allowed")
    (dest_path / "artifact.bin").write_bytes(b"bin")
    (dest_path / "stable-diffusion.cpp").mkdir(exist_ok=True)
    (dest_path / "stable-diffusion.cpp" / "CMakeCache.txt").write_text("cmake")

def test_list_local_image_files(tmp_path):
    create_fake_image_files(tmp_path, 2)
    files = download_selective._list_local_image_files(tmp_path)
    assert files == {"img_0.png", "img_1.png"}

def test_handle_remove_readonly(tmp_path):
    file_path = tmp_path / "readonly.txt"
    file_path.write_text("readonly")
    os.chmod(file_path, 0o444)
    # Should not raise
    download_selective._handle_remove_readonly(os.remove, str(file_path), None)
    assert not file_path.exists()

def test_run_success_images(monkeypatch, tmp_path):
    # Simulate images appearing and process exiting
    called = {}
    def fake_popen(*args, **kwargs):
        class FakeProc:
            def __init__(self):
                self._poll = None
                self._terminated = False
            def poll(self):
                return 0 if self._terminated else None
            def terminate(self):
                self._terminated = True
            def wait(self, timeout=None):
                self._terminated = True
            def kill(self):
                self._terminated = True
        called['popen'] = True
        return FakeProc()
    monkeypatch.setattr(download_selective, 'subprocess', download_selective.subprocess)
    monkeypatch.setattr(download_selective.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(download_selective, '_get_kaggle_command', lambda: ['echo'])
    # Simulate images appearing
    def fake_list_local_image_files(dest_path):
        if not hasattr(fake_list_local_image_files, 'called'):
            fake_list_local_image_files.called = 0
        fake_list_local_image_files.called += 1
        if fake_list_local_image_files.called < 3:
            return set()
        return {"img_0.png", "img_1.png"}
    monkeypatch.setattr(download_selective, '_list_local_image_files', fake_list_local_image_files)
    assert download_selective.run(dest=str(tmp_path))

def test_run_timeout(monkeypatch, tmp_path):
    # Simulate no images ever appearing, process times out
    class FakeProc:
        def __init__(self):
            self._terminated = False
        def poll(self):
            return None
        def terminate(self):
            self._terminated = True
        def wait(self, timeout=None):
            self._terminated = True
        def kill(self):
            self._terminated = True
    monkeypatch.setattr(download_selective, 'subprocess', download_selective.subprocess)
    monkeypatch.setattr(download_selective.subprocess, 'Popen', lambda *a, **k: FakeProc())
    monkeypatch.setattr(download_selective, '_get_kaggle_command', lambda: ['echo'])
    monkeypatch.setattr(download_selective, '_list_local_image_files', lambda d: set())
    assert not download_selective.run(dest=str(tmp_path))

def test_run_removes_non_images(monkeypatch, tmp_path):
    # Simulate images and non-image files, ensure non-images are removed
    class FakeProc:
        def __init__(self):
            self._terminated = False
        def poll(self):
            return None
        def terminate(self):
            self._terminated = True
        def wait(self, timeout=None):
            self._terminated = True
        def kill(self):
            self._terminated = True
    monkeypatch.setattr(download_selective, 'subprocess', download_selective.subprocess)
    monkeypatch.setattr(download_selective.subprocess, 'Popen', lambda *a, **k: FakeProc())
    monkeypatch.setattr(download_selective, '_get_kaggle_command', lambda: ['echo'])
    # Simulate images and non-images
    def fake_list_local_image_files(dest_path):
        return {"img_0.png", "img_1.png"}
    monkeypatch.setattr(download_selective, '_list_local_image_files', fake_list_local_image_files)
    create_fake_image_files(tmp_path, 2)
    create_fake_non_image_files(tmp_path)
    assert download_selective.run(dest=str(tmp_path))
    # Only allowed non-image files remain
    allowed = {"cli_command.txt", "stable-diffusion-batch-generator.log", "img_0.png", "img_1.png"}
    found = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    assert found <= allowed

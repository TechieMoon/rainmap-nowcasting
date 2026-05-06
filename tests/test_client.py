import pytest


def test_client_window_smoke() -> None:
    pytest.importorskip("PySide6")
    from rainmap_nowcasting.client import QApplication, MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.windowTitle() == "RainMap Nowcasting"
    window.close()
    app.quit()

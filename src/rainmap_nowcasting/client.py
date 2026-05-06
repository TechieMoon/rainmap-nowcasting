from __future__ import annotations

import sys
from pathlib import Path

from . import DEFAULT_REPO_ID
from .hf import ensure_model_files, has_model_files
from .inference import predict_frames

try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError:  # pragma: no cover - tested by import fallback only.
    QApplication = None


class DownloadWorker(QThread):
    progress = Signal(str)
    failed = Signal(str)
    done = Signal(str)

    def run(self) -> None:
        try:
            self.progress.emit("Downloading model files...")
            files = ensure_model_files(repo_id=DEFAULT_REPO_ID)
            self.done.emit(str(files.model_dir))
        except Exception as exc:
            self.failed.emit(str(exc))


class PredictionWorker(QThread):
    progress = Signal(str)
    failed = Signal(str)
    done = Signal(dict)

    def __init__(self, input_dir: str, output_dir: str) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self) -> None:
        try:
            self.progress.emit("Preparing model...")
            metadata = predict_frames(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                repo_id=DEFAULT_REPO_ID,
            )
            self.done.emit(metadata)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RainMap Nowcasting")
        self.resize(900, 680)
        self.worker: QThread | None = None
        self.frame_paths: list[str] = []

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        form = QFormLayout()
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit(str(Path("outputs") / "client_predictions"))
        form.addRow("Input images", self._with_button(self.input_edit, "Browse", self.browse_input))
        form.addRow("Output folder", self._with_button(self.output_edit, "Browse", self.browse_output))
        layout.addLayout(form)

        actions = QHBoxLayout()
        self.download_button = QPushButton("Download model")
        self.download_button.clicked.connect(self.download_model)
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)
        actions.addWidget(self.download_button)
        actions.addWidget(self.predict_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.status_label = QLabel()
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        self.preview_selector = QComboBox()
        self.preview_selector.currentIndexChanged.connect(self.update_preview)
        layout.addWidget(self.preview_selector)

        self.preview = QLabel()
        self.preview.setMinimumHeight(300)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background: #111; color: #eee; border: 1px solid #444;")
        layout.addWidget(self.preview, stretch=1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        layout.addWidget(self.log)

        self.refresh_status()

    def _with_button(self, edit: QLineEdit, label: str, callback) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton(label)
        button.clicked.connect(callback)
        layout.addWidget(edit)
        layout.addWidget(button)
        return widget

    def append_log(self, message: str) -> None:
        self.log.append(message)

    def refresh_status(self) -> None:
        if has_model_files(DEFAULT_REPO_ID):
            self.status_label.setText("Model status: cached locally")
        else:
            self.status_label.setText("Model status: will download on first use")

    def browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select input image folder")
        if path:
            self.input_edit.setText(path)

    def browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.output_edit.setText(path)

    def set_busy(self, busy: bool) -> None:
        self.download_button.setEnabled(not busy)
        self.predict_button.setEnabled(not busy)

    def download_model(self) -> None:
        self.set_busy(True)
        self.worker = DownloadWorker()
        self.worker.progress.connect(self.append_log)
        self.worker.failed.connect(self.on_failed)
        self.worker.done.connect(self.on_downloaded)
        self.worker.start()

    def predict(self) -> None:
        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        if not input_dir:
            QMessageBox.warning(self, "Input required", "Choose an input image folder.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Output required", "Choose an output folder.")
            return
        self.set_busy(True)
        self.preview_selector.clear()
        self.preview.clear()
        self.worker = PredictionWorker(input_dir=input_dir, output_dir=output_dir)
        self.worker.progress.connect(self.append_log)
        self.worker.failed.connect(self.on_failed)
        self.worker.done.connect(self.on_predicted)
        self.worker.start()

    def on_downloaded(self, model_dir: str) -> None:
        self.set_busy(False)
        self.refresh_status()
        self.append_log(f"Model ready: {model_dir}")

    def on_predicted(self, metadata: dict) -> None:
        self.set_busy(False)
        self.refresh_status()
        self.frame_paths = [str(path) for path in metadata.get("frame_paths", [])]
        self.preview_selector.clear()
        for path in self.frame_paths:
            self.preview_selector.addItem(Path(path).name, path)
        if self.frame_paths:
            self.preview_selector.setCurrentIndex(0)
            self.update_preview(0)
        self.append_log(f"Saved prediction GIF: {metadata.get('gif_path')}")

    def on_failed(self, message: str) -> None:
        self.set_busy(False)
        self.append_log(f"Error: {message}")
        QMessageBox.critical(self, "RainMap Nowcasting", message)

    def update_preview(self, index: int) -> None:
        if index < 0:
            return
        path = self.preview_selector.itemData(index)
        if not path:
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        scaled = pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.update_preview(self.preview_selector.currentIndex())


def main() -> None:
    if QApplication is None:
        raise SystemExit("PySide6 is not installed. Run: python -m pip install -e .[gui]")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()

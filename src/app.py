"""Simple PySide6 GUI for live face detection, enrolment & recognition."""

from __future__ import annotations

import sys
from typing import List

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from . import config

# Local imports (absolute for packaged usage)
from .engine import FaceEngine
from .face_db import FaceDatabase


class MainWindow(QMainWindow):
    def __init__(
        self,
        cam_index: int = config.APP_CAM_INDEX,
        enrol_frames: int = config.APP_ENROL_FRAMES,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Face Recognition Demo")

        # ---- UI widgets -------------------------------------------------
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.info_label = QLabel("No face")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.enrol_button = QPushButton("Enroll new user")
        self.enrol_button.clicked.connect(self._start_enrol)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.enrol_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # ---- Face engine & DB -----------------------------------------
        self.engine = FaceEngine()
        self.db = FaceDatabase()

        # ---- Video -----------------------------------------------------
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.APP_FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.APP_FRAME_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

        # Timer to grab frames (30 fps ≈ 33 ms)
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_frame)
        self.timer.start(config.APP_TIMER_INTERVAL_MS)

        # ---- Enrolment state ------------------------------------------
        self.enrol_frames_target = enrol_frames
        self._enrolling = False
        self._pending_feats: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _start_enrol(self):
        if self._enrolling:
            return  # already enrolling

        dialog = EnrolDialog(self)
        if dialog.exec():
            name, person_type = dialog.get_details()
            if not name:
                return
            self._current_enrol_name = name
            self._current_person_type = person_type
            self._enrolling = True
            self._pending_feats.clear()
            self.info_label.setText(f"Enrolling {name}…")

    def _process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        faces_feats = self.engine.detect_and_extract(frame, top_k=1)
        name_to_show = "No face"
        if faces_feats:
            face, feat = faces_feats[0]
            # Draw bbox
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if self._enrolling:
                # collect embeddings
                self._pending_feats.append(feat)
                remaining = self.enrol_frames_target - len(self._pending_feats)
                name_to_show = f"Enrolling {self._current_enrol_name} ({remaining})"
                if remaining <= 0:
                    # Average embedding and save
                    avg_emb = np.mean(self._pending_feats, axis=0)
                    self.db.add_person(avg_emb, self._current_enrol_name, self._current_person_type)
                    self._enrolling = False
                    name_to_show = f"Enrolled {self._current_enrol_name}!"
            else:
                # Recognition
                match_name = self.db.search(feat)
                if match_name is not None:
                    name_to_show = f"Hello, {match_name}!"
                    cv2.putText(
                        frame,
                        match_name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    name_to_show = "Unknown"

        self.info_label.setText(name_to_show)
        # Convert BGR → RGB → Qt
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    # ------------------------------------------------------------------
    # Qt events
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        self.db.close()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class EnrolDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New person")
        layout = QVBoxLayout(self)

        # Name input
        self.name_label = QLabel("Enter name:")
        self.name_input = QLineEdit()
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)

        # Person type
        self.person_type_group = QButtonGroup()
        self.cadet_radio = QRadioButton("Cadet")
        self.employee_radio = QRadioButton("Employee")
        self.cadet_radio.setChecked(True)
        self.person_type_group.addButton(self.cadet_radio)
        self.person_type_group.addButton(self.employee_radio)
        person_type_layout = QHBoxLayout()
        person_type_layout.addWidget(self.cadet_radio)
        person_type_layout.addWidget(self.employee_radio)
        layout.addLayout(person_type_layout)

        # Buttons
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

    def get_details(self):
        name = self.name_input.text()
        person_type = "Cadet" if self.cadet_radio.isChecked() else "Employee"
        return name, person_type


# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()

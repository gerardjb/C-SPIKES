from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import numpy as np
from c_spikes.cascade2p.cascade import download_model as cascade_download_model
from c_spikes.gui.data import DataManager, EpochRef, scan_dataset_dir
from c_spikes.gui.inference import (
    InferenceSettings,
    RunContext,
    build_run_context,
    ensure_run_dirs,
    run_inference_for_epoch_safe,
)
from c_spikes.gui.models import (
    detect_biophys_model_kind,
    format_model_dir_label,
    list_biophys_model_dirs,
    list_cascade_available_models,
    list_cascade_local_models,
    list_ens2_model_dirs,
)
from c_spikes.gui.plotting import plot_epoch


REPO_ROOT = Path(__file__).resolve().parents[3]
CASCADE_ROOT = REPO_ROOT / "Pretrained_models" / "CASCADE"
ENS2_ROOT = REPO_ROOT / "Pretrained_models" / "ENS2"
BIOPHYS_ROOT = REPO_ROOT / "Pretrained_models" / "BiophysML"
PGAS_PARAMS_ROOT = REPO_ROOT / "parameter_files"
PGAS_GPARAM_ROOT = REPO_ROOT / "src" / "c_spikes" / "pgas"


def _edges_for_epoch(
    edges_map: Optional[Dict[str, np.ndarray]],
    epoch: EpochRef,
) -> Optional[np.ndarray]:
    if edges_map is None:
        return None
    key = epoch.file_path.stem
    edges = edges_map.get(key)
    if edges is None:
        return None
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 2 or edges.shape[1] != 2:
        return None
    if epoch.epoch_index >= edges.shape[0]:
        return None
    return edges[epoch.epoch_index : epoch.epoch_index + 1]


class InferenceWorker(QtCore.QThread):
    result_ready = QtCore.Signal(str, dict, dict)
    status = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(
        self,
        *,
        epochs: List[EpochRef],
        data_manager: DataManager,
        settings: InferenceSettings,
        context: RunContext,
        edges_map: Optional[Dict[str, np.ndarray]],
        edges_enabled: bool,
    ) -> None:
        super().__init__()
        self._epochs = epochs
        self._data_manager = data_manager
        self._settings = settings
        self._context = context
        self._edges_map = edges_map
        self._edges_enabled = edges_enabled

    def run(self) -> None:
        for epoch in self._epochs:
            self.status.emit(f"Loading {epoch.display}...")
            try:
                time, dff, spikes = self._data_manager.load_epoch(epoch)
            except Exception as exc:
                self.result_ready.emit(epoch.epoch_id, {}, {"load": str(exc)})
                continue
            edges = None
            if self._edges_enabled:
                edges = _edges_for_epoch(self._edges_map, epoch)
            method_results, method_errors = run_inference_for_epoch_safe(
                epoch_id=epoch.epoch_id,
                time=time,
                dff=dff,
                spike_times=spikes,
                settings=self._settings,
                context=self._context,
                edges=edges,
            )
            self.result_ready.emit(epoch.epoch_id, method_results, method_errors)
        self.finished.emit()


class DownloadWorker(QtCore.QThread):
    status = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)

    def __init__(self, model_name: str, model_root: Path) -> None:
        super().__init__()
        self._model_name = model_name
        self._model_root = model_root

    def run(self) -> None:
        try:
            cascade_download_model(self._model_name, model_folder=str(self._model_root))
        except Exception as exc:
            self.finished.emit(False, str(exc))
            return
        if self._model_name == "update_models":
            self.finished.emit(True, f"Updated available_models.yaml in {self._model_root}")
        else:
            self.finished.emit(True, f"Downloaded {self._model_name} into {self._model_root}")


class ConfigEditorDialog(QtWidgets.QDialog):
    def __init__(self, path: Path, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._path = Path(path)
        self.setWindowTitle(f"Edit PGAS Config: {self._path.name}")
        self.resize(700, 500)

        layout = QtWidgets.QVBoxLayout(self)
        self._editor = QtWidgets.QPlainTextEdit(self)
        self._editor.setPlainText(self._path.read_text(encoding="utf-8"))
        layout.addWidget(self._editor)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        save_btn = QtWidgets.QPushButton("Save", self)
        cancel_btn = QtWidgets.QPushButton("Cancel", self)
        save_btn.clicked.connect(self._on_save)
        cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(save_btn)
        button_row.addWidget(cancel_btn)
        layout.addLayout(button_row)

    def text(self) -> str:
        return self._editor.toPlainText()

    def _on_save(self) -> None:
        text = self.text()
        if self._path.suffix.lower() == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid JSON", str(exc))
                return
        self.accept()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("C-SPIKES GUI")
        self.resize(1200, 800)

        self._data_manager = DataManager()
        self._epoch_refs: List[EpochRef] = []
        self._results_by_epoch: Dict[str, Dict[str, object]] = {}
        self._errors_by_epoch: Dict[str, Dict[str, str]] = {}
        self._current_epoch_index: Optional[int] = None
        self._run_context: Optional[RunContext] = None
        self._worker: Optional[InferenceWorker] = None
        self._download_worker: Optional[DownloadWorker] = None
        self._device_locked = False
        self._edges_path: Optional[Path] = None
        self._edges_map: Optional[Dict[str, np.ndarray]] = None
        self._edges_enabled = False
        self._xlim_cids: List[Tuple[object, int]] = []
        self._syncing_x = False

        self._build_ui()
        self._refresh_model_lists()
        self._refresh_pgas_lists()
        self._update_gpu_options()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        main_layout = QtWidgets.QHBoxLayout(central)

        left_panel = QtWidgets.QWidget(central)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)

        scroll = QtWidgets.QScrollArea(central)
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_panel)
        main_layout.addWidget(scroll, 0)

        right_panel = QtWidgets.QWidget(central)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        self._figure = Figure(figsize=(6, 5), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        right_layout.addWidget(self._toolbar, 0)
        right_layout.addWidget(self._canvas, 1)
        main_layout.addWidget(right_panel, 1)

        self.setCentralWidget(central)

        left_layout.addWidget(self._build_dataset_group())
        left_layout.addWidget(self._build_epoch_group())
        left_layout.addWidget(self._build_batch_group())
        left_layout.addWidget(self._build_method_group())
        left_layout.addWidget(self._build_model_group())
        left_layout.addWidget(self._build_biophys_group())
        left_layout.addWidget(self._build_pgas_group())
        left_layout.addWidget(self._build_device_group())
        left_layout.addWidget(self._build_status_group())

    def _build_dataset_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Dataset")
        layout = QtWidgets.QVBoxLayout(box)

        row = QtWidgets.QHBoxLayout()
        self._data_dir_edit = QtWidgets.QLineEdit(box)
        self._data_dir_edit.setReadOnly(True)
        browse_btn = QtWidgets.QPushButton("Browse", box)
        browse_btn.clicked.connect(self._choose_data_dir)
        row.addWidget(self._data_dir_edit)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(QtWidgets.QLabel("Run tag", box))
        self._run_tag_edit = QtWidgets.QLineEdit(box)
        self._run_tag_edit.setPlaceholderText("auto")
        run_row.addWidget(self._run_tag_edit)
        layout.addLayout(run_row)

        self._use_cache_check = QtWidgets.QCheckBox("Use cache", box)
        self._use_cache_check.setChecked(True)
        layout.addWidget(self._use_cache_check)

        edges_row = QtWidgets.QHBoxLayout()
        self._edges_check = QtWidgets.QCheckBox("Use edges", box)
        self._edges_check.toggled.connect(self._on_edges_toggled)
        self._edges_path_edit = QtWidgets.QLineEdit(box)
        self._edges_path_edit.setReadOnly(True)
        edges_browse = QtWidgets.QPushButton("Edges...", box)
        edges_browse.clicked.connect(self._choose_edges_file)
        edges_row.addWidget(self._edges_check)
        edges_row.addWidget(self._edges_path_edit)
        edges_row.addWidget(edges_browse)
        layout.addLayout(edges_row)

        self._data_info_label = QtWidgets.QLabel("No dataset loaded", box)
        layout.addWidget(self._data_info_label)

        return box

    def _build_epoch_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Epoch")
        layout = QtWidgets.QVBoxLayout(box)

        self._epoch_combo = QtWidgets.QComboBox(box)
        self._epoch_combo.currentIndexChanged.connect(self._on_epoch_selected)
        layout.addWidget(self._epoch_combo)

        nav_row = QtWidgets.QHBoxLayout()
        self._prev_btn = QtWidgets.QPushButton("Prev", box)
        self._next_btn = QtWidgets.QPushButton("Next", box)
        self._prev_btn.clicked.connect(lambda: self._step_epoch(-1))
        self._next_btn.clicked.connect(lambda: self._step_epoch(1))
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._next_btn)
        layout.addLayout(nav_row)

        return box

    def _build_batch_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Batch Selection")
        layout = QtWidgets.QVBoxLayout(box)

        self._epoch_list = QtWidgets.QListWidget(box)
        self._epoch_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self._epoch_list)

        btn_row = QtWidgets.QHBoxLayout()
        select_all = QtWidgets.QPushButton("Select All", box)
        clear = QtWidgets.QPushButton("Clear", box)
        select_all.clicked.connect(self._epoch_list.selectAll)
        clear.clicked.connect(self._epoch_list.clearSelection)
        btn_row.addWidget(select_all)
        btn_row.addWidget(clear)
        layout.addLayout(btn_row)

        return box

    def _build_method_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Methods")
        layout = QtWidgets.QVBoxLayout(box)

        self._pgas_check = QtWidgets.QCheckBox("biophys_smc", box)
        self._biophys_check = QtWidgets.QCheckBox("BiophysML", box)
        self._cascade_check = QtWidgets.QCheckBox("Cascade", box)
        self._ens2_check = QtWidgets.QCheckBox("ENS2", box)
        self._pgas_check.setChecked(True)
        self._biophys_check.setChecked(False)
        self._cascade_check.setChecked(True)
        self._ens2_check.setChecked(True)
        layout.addWidget(self._pgas_check)
        layout.addWidget(self._biophys_check)
        layout.addWidget(self._cascade_check)
        layout.addWidget(self._ens2_check)

        self._run_btn = QtWidgets.QPushButton("Run Inference", box)
        self._run_btn.clicked.connect(self._run_inference)
        layout.addWidget(self._run_btn)

        return box

    def _build_model_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Models")
        layout = QtWidgets.QVBoxLayout(box)

        layout.addWidget(QtWidgets.QLabel("Cascade model (official)", box))
        self._cascade_official_combo = QtWidgets.QComboBox(box)
        layout.addWidget(self._cascade_official_combo)

        cascade_btn_row = QtWidgets.QHBoxLayout()
        self._cascade_refresh_btn = QtWidgets.QPushButton("Refresh List", box)
        self._cascade_download_btn = QtWidgets.QPushButton("Download", box)
        self._cascade_refresh_btn.clicked.connect(self._update_cascade_available_models)
        self._cascade_download_btn.clicked.connect(self._download_selected_cascade)
        cascade_btn_row.addWidget(self._cascade_refresh_btn)
        cascade_btn_row.addWidget(self._cascade_download_btn)
        layout.addLayout(cascade_btn_row)

        layout.addWidget(QtWidgets.QLabel("ENS2 model (official)", box))
        self._ens2_official_combo = QtWidgets.QComboBox(box)
        layout.addWidget(self._ens2_official_combo)

        neuron_row = QtWidgets.QHBoxLayout()
        neuron_row.addWidget(QtWidgets.QLabel("Neuron type", box))
        self._neuron_combo = QtWidgets.QComboBox(box)
        self._neuron_combo.addItems(["Exc", "Inh"])
        neuron_row.addWidget(self._neuron_combo)
        layout.addLayout(neuron_row)

        return box

    def _build_biophys_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("BiophysML Models")
        layout = QtWidgets.QVBoxLayout(box)

        layout.addWidget(QtWidgets.QLabel("BiophysML model", box))
        self._biophys_combo = QtWidgets.QComboBox(box)
        layout.addWidget(self._biophys_combo)

        refresh_row = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Refresh List", box)
        refresh_btn.clicked.connect(self._refresh_biophys_models)
        refresh_row.addWidget(refresh_btn)
        layout.addLayout(refresh_row)

        return box

    def _build_pgas_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("PGAS Config")
        layout = QtWidgets.QVBoxLayout(box)

        layout.addWidget(QtWidgets.QLabel("Constants file", box))
        self._pgas_constants_combo = QtWidgets.QComboBox(box)
        layout.addWidget(self._pgas_constants_combo)

        layout.addWidget(QtWidgets.QLabel("GParam file", box))
        self._pgas_gparam_combo = QtWidgets.QComboBox(box)
        layout.addWidget(self._pgas_gparam_combo)

        edit_btn = QtWidgets.QPushButton("Edit Constants", box)
        edit_btn.clicked.connect(self._edit_pgas_constants)
        layout.addWidget(edit_btn)

        return box

    def _build_device_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Device")
        layout = QtWidgets.QVBoxLayout(box)

        self._cpu_radio = QtWidgets.QRadioButton("CPU", box)
        self._gpu_radio = QtWidgets.QRadioButton("GPU", box)
        self._cpu_radio.setChecked(True)
        layout.addWidget(self._cpu_radio)
        layout.addWidget(self._gpu_radio)

        return box

    def _build_status_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Status")
        layout = QtWidgets.QVBoxLayout(box)
        self._status_log = QtWidgets.QPlainTextEdit(box)
        self._status_log.setReadOnly(True)
        layout.addWidget(self._status_log)
        return box

    def _log(self, message: str) -> None:
        self._status_log.appendPlainText(message)

    def _choose_data_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if directory:
            self._data_dir_edit.setText(directory)
            self._load_dataset_dir(Path(directory))

    def _load_dataset_dir(self, directory: Path) -> None:
        epochs, errors = scan_dataset_dir(directory)
        self._epoch_refs = epochs
        self._results_by_epoch.clear()
        self._errors_by_epoch.clear()
        try:
            default_context = build_run_context(directory, "")
            self._run_tag_edit.setPlaceholderText(f"auto ({default_context.run_tag})")
        except Exception:
            self._run_tag_edit.setPlaceholderText("auto")
        self._epoch_combo.blockSignals(True)
        self._epoch_combo.clear()
        self._epoch_list.clear()
        for epoch in epochs:
            self._epoch_combo.addItem(epoch.display, epoch)
            item = QtWidgets.QListWidgetItem(epoch.display)
            item.setData(Qt.UserRole, epoch)
            self._epoch_list.addItem(item)
        self._epoch_combo.blockSignals(False)
        if errors:
            for err in errors:
                self._log(f"Dataset scan error: {err}")
        if epochs:
            self._data_info_label.setText(f"Loaded {len(epochs)} epochs from {directory}")
            self._epoch_combo.setCurrentIndex(0)
        else:
            self._data_info_label.setText("No epochs found")
            self._figure.clear()
            self._canvas.draw()

    def _on_epoch_selected(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._epoch_refs):
            return
        self._current_epoch_index = idx
        epoch = self._epoch_refs[idx]
        try:
            time, dff, spikes = self._data_manager.load_epoch(epoch)
        except Exception as exc:
            self._log(f"Failed to load epoch: {exc}")
            return
        methods = self._results_by_epoch.get(epoch.epoch_id, {})
        plot_epoch(
            self._figure,
            time=time,
            dff=dff,
            methods=methods,
            spike_times=spikes,
            title=epoch.display,
        )
        self._connect_x_sync()
        self._canvas.draw()

    def _step_epoch(self, delta: int) -> None:
        if not self._epoch_refs:
            return
        if self._current_epoch_index is None:
            self._current_epoch_index = 0
        new_idx = (self._current_epoch_index + delta) % len(self._epoch_refs)
        self._epoch_combo.setCurrentIndex(new_idx)

    def _update_gpu_options(self) -> None:
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
            gpu_available = False
        else:
            gpu_available = Path("/dev/nvidia0").exists() or Path("/dev/nvidiactl").exists()

        if gpu_available:
            self._gpu_radio.setEnabled(True)
            self._gpu_radio.setChecked(True)
        else:
            self._gpu_radio.setEnabled(False)
            self._cpu_radio.setChecked(True)

    def _apply_device_preference(self) -> None:
        if self._device_locked:
            return
        if self._cpu_radio.isChecked():
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            self._log("Device preference set to CPU (CUDA disabled).")
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            self._log("Device preference set to GPU (if available).")
        self._device_locked = True
        self._cpu_radio.setEnabled(False)
        self._gpu_radio.setEnabled(False)

    def _refresh_model_lists(self) -> None:
        self._refresh_cascade_models()
        self._refresh_ens2_models()
        self._refresh_biophys_models()

    def _refresh_cascade_models(self) -> None:
        available = list_cascade_available_models(CASCADE_ROOT)
        local = list_cascade_local_models(CASCADE_ROOT)
        models = available if available else local
        self._cascade_official_combo.clear()
        if models:
            for name in models:
                self._cascade_official_combo.addItem(name)

    def _update_cascade_available_models(self) -> None:
        if self._download_worker and self._download_worker.isRunning():
            self._log("Another download is in progress.")
            return
        self._download_worker = DownloadWorker("update_models", CASCADE_ROOT)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.start()
        self._log("Refreshing Cascade model list...")

    def _refresh_ens2_models(self) -> None:
        self._ens2_official_combo.clear()
        for model_dir in list_ens2_model_dirs(ENS2_ROOT):
            self._ens2_official_combo.addItem(format_model_dir_label(ENS2_ROOT, model_dir), model_dir)
    def _refresh_biophys_models(self) -> None:
        self._biophys_combo.clear()
        for model_dir in list_biophys_model_dirs(BIOPHYS_ROOT):
            self._biophys_combo.addItem(format_model_dir_label(BIOPHYS_ROOT, model_dir), model_dir)

    def _refresh_pgas_lists(self) -> None:
        self._pgas_constants_combo.clear()
        if PGAS_PARAMS_ROOT.exists():
            for path in sorted(PGAS_PARAMS_ROOT.glob("*.json")):
                self._pgas_constants_combo.addItem(path.name, path)
        self._pgas_gparam_combo.clear()
        if PGAS_GPARAM_ROOT.exists():
            for path in sorted(PGAS_GPARAM_ROOT.glob("*.dat")):
                self._pgas_gparam_combo.addItem(path.name, path)

    def _edit_pgas_constants(self) -> None:
        path = self._current_pgas_constants_path()
        if path is None:
            self._log("No PGAS constants file selected.")
            return
        dialog = ConfigEditorDialog(path, self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        if not self._ensure_run_context():
            return
        assert self._run_context is not None
        ensure_run_dirs(self._run_context)
        temp_path = self._run_context.pgas_temp_root / path.name
        temp_path.write_text(dialog.text(), encoding="utf-8")
        label = f"{path.name} (edited)"
        self._pgas_constants_combo.addItem(label, temp_path)
        self._pgas_constants_combo.setCurrentIndex(self._pgas_constants_combo.count() - 1)
        self._log(f"Saved edited constants to {temp_path}")

    def _download_selected_cascade(self) -> None:
        model_name = self._cascade_official_combo.currentText().strip()
        if not model_name:
            self._log("Select a Cascade model to download.")
            return
        if self._download_worker and self._download_worker.isRunning():
            self._log("Download already in progress.")
            return
        self._download_worker = DownloadWorker(model_name, CASCADE_ROOT)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.start()
        self._log(f"Downloading Cascade model {model_name}...")

    def _on_download_finished(self, ok: bool, message: str) -> None:
        self._log(message)
        if ok:
            self._refresh_cascade_models()

    def _current_pgas_constants_path(self) -> Optional[Path]:
        data = self._pgas_constants_combo.currentData()
        if isinstance(data, Path):
            return data
        if self._pgas_constants_combo.currentText():
            return PGAS_PARAMS_ROOT / self._pgas_constants_combo.currentText()
        return None

    def _current_pgas_gparam_path(self) -> Optional[Path]:
        data = self._pgas_gparam_combo.currentData()
        if isinstance(data, Path):
            return data
        if self._pgas_gparam_combo.currentText():
            return PGAS_GPARAM_ROOT / self._pgas_gparam_combo.currentText()
        return None

    def _ensure_run_context(self) -> bool:
        data_dir = self._data_dir_edit.text().strip()
        if not data_dir:
            self._log("Select a dataset directory first.")
            return False
        run_tag = self._run_tag_edit.text().strip()
        context = build_run_context(Path(data_dir), run_tag)
        self._run_context = context
        if not run_tag:
            self._run_tag_edit.setText(context.run_tag)
        return True

    def _collect_epoch_selection(self) -> List[EpochRef]:
        selected = [item.data(Qt.UserRole) for item in self._epoch_list.selectedItems()]
        if selected:
            return [epoch for epoch in selected if isinstance(epoch, EpochRef)]
        if self._current_epoch_index is None and self._epoch_refs:
            self._current_epoch_index = 0
        if self._current_epoch_index is None:
            return []
        return [self._epoch_refs[self._current_epoch_index]]

    def _build_settings(self) -> Optional[InferenceSettings]:
        run_cascade = self._cascade_check.isChecked()
        run_ens2 = self._ens2_check.isChecked()
        run_biophys = self._biophys_check.isChecked()
        run_pgas = self._pgas_check.isChecked()
        if not (run_cascade or run_ens2 or run_biophys or run_pgas):
            self._log("Select at least one method to run.")
            return None

        if run_cascade:
            cascade_name = self._cascade_official_combo.currentText().strip()
            if not cascade_name:
                self._log("Select a Cascade model.")
                return None
            cascade_root = CASCADE_ROOT
            model_dir = cascade_root / cascade_name
            if not (model_dir / "config.yaml").exists():
                self._log(
                    f"Cascade model not found locally: {model_dir}. Download or select a different model."
                )
                return None
        else:
            cascade_name = ""
            cascade_root = CASCADE_ROOT

        if run_ens2:
            ens2_dir = self._ens2_official_combo.currentData()
            if not isinstance(ens2_dir, Path):
                self._log("Select an ENS2 model directory.")
                return None
            if not (ens2_dir / "exc_ens2_pub.pt").exists() and not (ens2_dir / "inh_ens2_pub.pt").exists():
                self._log(f"ENS2 weights not found in {ens2_dir}.")
                return None
        else:
            ens2_dir = ENS2_ROOT

        if run_biophys:
            biophys_dir = self._biophys_combo.currentData()
            if not isinstance(biophys_dir, Path):
                self._log("Select a BiophysML model directory.")
                return None
            biophys_kind = detect_biophys_model_kind(biophys_dir)
            if biophys_kind is None:
                self._log(f"BiophysML model not recognized in {biophys_dir}.")
                return None
        else:
            biophys_dir = BIOPHYS_ROOT
            biophys_kind = "ens2"

        if run_pgas:
            constants_path = self._current_pgas_constants_path()
            gparam_path = self._current_pgas_gparam_path()
            if constants_path is None or not constants_path.exists():
                self._log("Select a valid PGAS constants file.")
                return None
            if gparam_path is None or not gparam_path.exists():
                self._log("Select a valid PGAS gparam file.")
                return None
        else:
            constants_path = PGAS_PARAMS_ROOT / "constants_GCaMP8_soma.json"
            gparam_path = PGAS_GPARAM_ROOT / "20230525_gold.dat"

        settings = InferenceSettings(
            run_cascade=run_cascade,
            run_ens2=run_ens2,
            run_pgas=run_pgas,
            run_biophys=run_biophys,
            neuron_type=self._neuron_combo.currentText(),
            use_cache=self._use_cache_check.isChecked(),
            cascade_model_folder=cascade_root,
            cascade_model_name=cascade_name,
            ens2_pretrained_dir=ens2_dir,
            biophys_pretrained_dir=biophys_dir,
            biophys_kind=biophys_kind,
            pgas_constants_file=constants_path,
            pgas_gparam_file=gparam_path,
        )
        return settings

    def _run_inference(self) -> None:
        if self._worker and self._worker.isRunning():
            self._log("Inference already running.")
            return
        if not self._ensure_run_context():
            return
        settings = self._build_settings()
        if settings is None:
            return
        epochs = self._collect_epoch_selection()
        if not epochs:
            self._log("No epochs selected.")
            return

        self._apply_device_preference()
        assert self._run_context is not None
        ensure_run_dirs(self._run_context)
        self._write_manifest(settings, epochs)

        self._worker = InferenceWorker(
            epochs=epochs,
            data_manager=self._data_manager,
            settings=settings,
            context=self._run_context,
            edges_map=self._edges_map,
            edges_enabled=self._edges_enabled,
        )
        self._worker.status.connect(self._log)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.finished.connect(self._on_run_finished)
        self._run_btn.setEnabled(False)
        self._worker.start()
        self._log(f"Running inference on {len(epochs)} epoch(s)...")

    def _write_manifest(self, settings: InferenceSettings, epochs: List[EpochRef]) -> None:
        if self._run_context is None:
            return
        payload = {
            "run_tag": self._run_context.run_tag,
            "data_dir": str(self._run_context.data_dir),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "methods": {
                "cascade": settings.run_cascade,
                "ens2": settings.run_ens2,
                "biophys_ml": settings.run_biophys,
                "pgas": settings.run_pgas,
            },
            "models": {
                "cascade_model_folder": str(settings.cascade_model_folder),
                "cascade_model_name": settings.cascade_model_name,
                "ens2_pretrained_dir": str(settings.ens2_pretrained_dir),
                "biophys_pretrained_dir": str(settings.biophys_pretrained_dir),
                "biophys_kind": settings.biophys_kind,
                "neuron_type": settings.neuron_type,
            },
            "pgas": {
                "constants_file": str(settings.pgas_constants_file),
                "gparam_file": str(settings.pgas_gparam_file),
            },
            "edges": {
                "enabled": bool(self._edges_enabled),
                "path": str(self._edges_path) if self._edges_path else None,
            },
            "epochs": [epoch.epoch_id for epoch in epochs],
        }
        manifest_path = self._run_context.run_root / "manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _on_result_ready(self, epoch_id: str, results: dict, errors: dict) -> None:
        if results:
            self._results_by_epoch.setdefault(epoch_id, {}).update(results)
        if errors:
            self._errors_by_epoch.setdefault(epoch_id, {}).update(errors)
            for method, message in errors.items():
                self._log(f"{epoch_id} {method} error: {message}")
        if self._current_epoch_index is not None:
            current = self._epoch_refs[self._current_epoch_index]
            if current.epoch_id == epoch_id:
                self._on_epoch_selected(self._current_epoch_index)

    def _on_run_finished(self) -> None:
        self._run_btn.setEnabled(True)
        self._log("Inference finished.")

    def _connect_x_sync(self) -> None:
        for ax, cid in self._xlim_cids:
            try:
                ax.callbacks.disconnect(cid)
            except Exception:
                pass
        self._xlim_cids = []
        axes = list(self._figure.axes)
        if not axes:
            return

        def _on_xlim_changed(ax):
            if self._syncing_x:
                return
            self._syncing_x = True
            xlim = ax.get_xlim()
            for other in axes:
                if other is ax:
                    continue
                other.set_xlim(xlim)
            self._syncing_x = False
            self._canvas.draw_idle()

        for ax in axes:
            cid = ax.callbacks.connect("xlim_changed", _on_xlim_changed)
            self._xlim_cids.append((ax, cid))

    def _on_edges_toggled(self, checked: bool) -> None:
        self._edges_enabled = bool(checked)
        if self._edges_enabled and self._edges_map is None:
            self._log("Edges enabled but no edges file loaded.")

    def _choose_edges_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select edges .npy file",
            self._data_dir_edit.text().strip() or str(REPO_ROOT),
            "NumPy files (*.npy)",
        )
        if not path:
            return
        self._load_edges_file(Path(path))

    def _load_edges_file(self, path: Path, *, log: bool = True) -> None:
        path = Path(path)
        try:
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile):
                raise ValueError("Expected a .npy dict file, got .npz.")
            edges_map = None
            if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object:
                edges_map = data.item()
            elif isinstance(data, dict):
                edges_map = data
            if not isinstance(edges_map, dict):
                raise ValueError("Edges file does not contain a dict mapping dataset -> edges.")
            cleaned: Dict[str, np.ndarray] = {}
            for key, value in edges_map.items():
                cleaned[str(key)] = np.asarray(value, dtype=float)
            self._edges_map = cleaned
            self._edges_path = path
            self._edges_path_edit.setText(str(path))
            if hasattr(self, "_edges_check") and not self._edges_check.isChecked():
                self._edges_check.setChecked(True)
            if log:
                self._log(f"Loaded edges file: {path}")
        except Exception as exc:
            self._edges_map = None
            self._edges_path = None
            self._edges_path_edit.setText("")
            self._log(f"Failed to load edges file: {exc}")


def main() -> None:
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

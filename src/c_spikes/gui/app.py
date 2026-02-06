from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import numpy as np
from c_spikes.tensorflow_env import preload_tensorflow_quietly

preload_tensorflow_quietly()

from c_spikes.cascade2p.cascade import download_model as cascade_download_model
from c_spikes.biophys_ml.pipeline import (
    default_cascade_train_config,
    default_ens2_train_config,
    default_synthetic_config,
    generate_synthetic_bundles,
    train_models_for_bundles,
)
from c_spikes.gui.data import DataManager, EpochRef, scan_dataset_dir
from c_spikes.gui.inference import (
    BiophysModelSpec,
    InferenceSettings,
    ModelSpec,
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
from c_spikes.gui.plotting import METHOD_ORDER, plot_epoch


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
    row = np.asarray(edges[epoch.epoch_index], dtype=float)
    if row.shape != (2,) or not np.all(np.isfinite(row)):
        return None
    return row.reshape(1, 2)


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
        try:
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
                try:
                    method_results, method_errors = run_inference_for_epoch_safe(
                        epoch_id=epoch.epoch_id,
                        time=time,
                        dff=dff,
                        spike_times=spikes,
                        settings=self._settings,
                        context=self._context,
                        edges=edges,
                    )
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                    method_results, method_errors = {}, {"setup": message}
                self.result_ready.emit(epoch.epoch_id, method_results, method_errors)
        finally:
            self.finished.emit()


class BioMlPgasWorker(QtCore.QThread):
    status = QtCore.Signal(str)
    finished = QtCore.Signal(dict, dict)

    def __init__(
        self,
        *,
        epochs: List[EpochRef],
        windows_by_epoch: Dict[str, List[float]],
        data_manager: DataManager,
        settings: InferenceSettings,
        context: RunContext,
    ) -> None:
        super().__init__()
        self._epochs = epochs
        self._windows_by_epoch = windows_by_epoch
        self._data_manager = data_manager
        self._settings = settings
        self._context = context

    def run(self) -> None:
        param_paths: Dict[str, str] = {}
        errors: Dict[str, str] = {}
        for epoch in self._epochs:
            self.status.emit(f"[biophys_ml] PGAS for {epoch.epoch_id}...")
            try:
                time, dff, spikes = self._data_manager.load_epoch(epoch)
            except Exception as exc:
                errors[epoch.epoch_id] = f"load failed: {exc}"
                continue
            edges = None
            row = self._windows_by_epoch.get(epoch.epoch_id)
            if row is not None and len(row) == 2:
                arr = np.asarray(row, dtype=float)
                if np.all(np.isfinite(arr)):
                    edges = arr.reshape(1, 2)
            results, run_errors = run_inference_for_epoch_safe(
                epoch_id=epoch.epoch_id,
                time=time,
                dff=dff,
                spike_times=spikes,
                settings=self._settings,
                context=self._context,
                edges=edges,
            )
            if "pgas" in run_errors:
                errors[epoch.epoch_id] = run_errors["pgas"]
                continue
            try:
                matches = sorted(
                    self._context.pgas_output_root.glob(f"param_samples_{epoch.epoch_id}*.dat"),
                    key=lambda p: p.stat().st_mtime,
                )
            except OSError:
                matches = []
            if matches:
                param_paths[epoch.epoch_id] = str(matches[-1])
                self.status.emit(f"[biophys_ml] param_samples: {matches[-1].name}")
            else:
                errors[epoch.epoch_id] = "param_samples file not found after PGAS run."
        self.finished.emit(param_paths, errors)


class BioMlSyntheticWorker(QtCore.QThread):
    status = QtCore.Signal(str)
    finished = QtCore.Signal(list, str)

    def __init__(
        self,
        *,
        param_samples: List[Path],
        run_root: Path,
        run_tag: str,
        synthetic_config: Dict[str, object],
    ) -> None:
        super().__init__()
        self._param_samples = param_samples
        self._run_root = run_root
        self._run_tag = run_tag
        self._synthetic_config = synthetic_config

    def run(self) -> None:
        try:
            self.status.emit("[biophys_ml] Generating synthetic datasets...")
            bundles = generate_synthetic_bundles(
                param_samples_paths=self._param_samples,
                run_root=self._run_root,
                run_tag=self._run_tag,
                synthetic_config=self._synthetic_config,
            )
            self.finished.emit(bundles, "")
        except Exception as exc:
            self.finished.emit([], str(exc))


class BioMlTrainWorker(QtCore.QThread):
    status = QtCore.Signal(str)
    finished = QtCore.Signal(list, str)

    def __init__(
        self,
        *,
        bundles: List[Dict[str, object]],
        run_root: Path,
        model_family: str,
        model_root: Path,
        train_config: Dict[str, object],
    ) -> None:
        super().__init__()
        self._bundles = bundles
        self._run_root = run_root
        self._model_family = model_family
        self._model_root = model_root
        self._train_config = train_config

    def run(self) -> None:
        try:
            self.status.emit(f"[biophys_ml] Training {self._model_family} models...")
            records = train_models_for_bundles(
                bundles=self._bundles,
                run_root=self._run_root,
                model_family=self._model_family,
                model_root=self._model_root,
                ens2_train_config=self._train_config if self._model_family == "ens2" else None,
                cascade_train_config=self._train_config if self._model_family == "cascade" else None,
            )
            self.finished.emit(records, "")
        except Exception as exc:
            self.finished.emit([], str(exc))


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
        self._display_selection_by_epoch: Dict[str, set[str]] = {}
        self._display_available_by_epoch: Dict[str, set[str]] = {}
        self._axis_labels_by_epoch: Dict[str, Dict[str, str]] = {}
        self._updating_display_list = False
        self._current_epoch_index: Optional[int] = None
        self._run_context: Optional[RunContext] = None
        self._bio_run_context: Optional[RunContext] = None
        self._worker: Optional[InferenceWorker] = None
        self._download_worker: Optional[DownloadWorker] = None
        self._device_locked = False
        self._last_batch_rows: List[int] = []
        self._edges_path: Optional[Path] = None
        self._edges_map: Optional[Dict[str, np.ndarray]] = None
        self._edges_enabled = False
        self._xlim_cids: List[Tuple[object, int]] = []
        self._syncing_x = False
        self._edge_epoch_index: Optional[int] = None
        self._edge_time: Optional[np.ndarray] = None
        self._edge_dff: Optional[np.ndarray] = None
        self._edge_spikes: Optional[np.ndarray] = None
        self._edge_click_cid: Optional[int] = None
        self._bio_epoch_index: Optional[int] = None
        self._bio_time: Optional[np.ndarray] = None
        self._bio_dff: Optional[np.ndarray] = None
        self._bio_spikes: Optional[np.ndarray] = None
        self._bio_click_cid: Optional[int] = None
        self._bio_windows_by_epoch: Dict[str, List[float]] = {}
        self._bio_param_samples_by_epoch: Dict[str, str] = {}
        self._bio_bundles: List[Dict[str, object]] = []
        self._bio_pgas_worker: Optional[BioMlPgasWorker] = None
        self._bio_synth_worker: Optional[BioMlSyntheticWorker] = None
        self._bio_train_worker: Optional[BioMlTrainWorker] = None
        self._bio_ens2_cfg_text: str = json.dumps(default_ens2_train_config(), indent=2)
        self._bio_cascade_cfg_text: str = json.dumps(default_cascade_train_config(), indent=2)

        self._build_ui()
        self._refresh_model_lists()
        self._refresh_pgas_lists()
        self._update_gpu_options()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        outer_layout = QtWidgets.QVBoxLayout(central)

        tabs = QtWidgets.QTabWidget(central)
        outer_layout.addWidget(tabs)
        self.setCentralWidget(central)

        spike_tab = QtWidgets.QWidget(tabs)
        spike_layout = QtWidgets.QHBoxLayout(spike_tab)

        left_panel = QtWidgets.QWidget(spike_tab)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)

        scroll = QtWidgets.QScrollArea(spike_tab)
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_panel)
        spike_layout.addWidget(scroll, 0)

        right_panel = QtWidgets.QWidget(spike_tab)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        self._figure = Figure(figsize=(6, 5), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        right_layout.addWidget(self._toolbar, 0)
        right_layout.addWidget(self._canvas, 1)
        spike_layout.addWidget(right_panel, 1)

        left_layout.addWidget(self._build_dataset_group())
        left_layout.addWidget(self._build_epoch_group())
        left_layout.addWidget(self._build_batch_group())
        left_layout.addWidget(self._build_display_group())
        left_layout.addWidget(self._build_method_group())
        left_layout.addWidget(self._build_model_group())
        left_layout.addWidget(self._build_biophys_group())
        left_layout.addWidget(self._build_pgas_group())
        left_layout.addWidget(self._build_device_group())

        tabs.addTab(spike_tab, "Spike Inference")

        edge_tab = QtWidgets.QWidget(tabs)
        edge_layout = QtWidgets.QHBoxLayout(edge_tab)

        edge_left = QtWidgets.QWidget(edge_tab)
        edge_left_layout = QtWidgets.QVBoxLayout(edge_left)
        edge_left_layout.setAlignment(Qt.AlignTop)

        edge_scroll = QtWidgets.QScrollArea(edge_tab)
        edge_scroll.setWidgetResizable(True)
        edge_scroll.setWidget(edge_left)
        edge_layout.addWidget(edge_scroll, 0)

        edge_right = QtWidgets.QWidget(edge_tab)
        edge_right_layout = QtWidgets.QVBoxLayout(edge_right)
        edge_right_layout.setContentsMargins(0, 0, 0, 0)
        edge_right_layout.setSpacing(2)

        self._edge_figure = Figure(figsize=(6, 5), dpi=100)
        self._edge_canvas = FigureCanvas(self._edge_figure)
        self._edge_toolbar = NavigationToolbar(self._edge_canvas, self)
        edge_right_layout.addWidget(self._edge_toolbar, 0)
        edge_right_layout.addWidget(self._edge_canvas, 1)
        edge_layout.addWidget(edge_right, 1)

        edge_left_layout.addWidget(self._build_edge_dataset_group())
        edge_left_layout.addWidget(self._build_edge_epoch_group())
        edge_left_layout.addWidget(self._build_edge_width_group())

        tabs.addTab(edge_tab, "Edge Selection")

        bio_tab = QtWidgets.QWidget(tabs)
        bio_layout = QtWidgets.QHBoxLayout(bio_tab)

        bio_left = QtWidgets.QWidget(bio_tab)
        bio_left_layout = QtWidgets.QVBoxLayout(bio_left)
        bio_left_layout.setAlignment(Qt.AlignTop)

        bio_scroll = QtWidgets.QScrollArea(bio_tab)
        bio_scroll.setWidgetResizable(True)
        bio_scroll.setWidget(bio_left)
        bio_layout.addWidget(bio_scroll, 0)

        bio_right = QtWidgets.QWidget(bio_tab)
        bio_right_layout = QtWidgets.QVBoxLayout(bio_right)
        bio_right_layout.setContentsMargins(0, 0, 0, 0)
        bio_right_layout.setSpacing(2)

        self._bio_figure = Figure(figsize=(6, 5), dpi=100)
        self._bio_canvas = FigureCanvas(self._bio_figure)
        self._bio_toolbar = NavigationToolbar(self._bio_canvas, self)
        bio_right_layout.addWidget(self._bio_toolbar, 0)
        bio_right_layout.addWidget(self._bio_canvas, 1)
        bio_layout.addWidget(bio_right, 1)

        bio_left_layout.addWidget(self._build_bio_dataset_group())
        bio_left_layout.addWidget(self._build_bio_epoch_group())
        bio_left_layout.addWidget(self._build_bio_selection_group())
        bio_left_layout.addWidget(self._build_bio_pgas_group())
        bio_left_layout.addWidget(self._build_bio_synthetic_group())
        bio_left_layout.addWidget(self._build_bio_train_group())
        tabs.addTab(bio_tab, "Biophys ML")

        outer_layout.addWidget(self._build_global_status_group())

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
        self._run_root_label = QtWidgets.QLabel("Run root: (set dataset/run tag)", box)
        layout.addWidget(self._run_root_label)

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

    def _build_edge_dataset_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Dataset")
        layout = QtWidgets.QVBoxLayout(box)

        row = QtWidgets.QHBoxLayout()
        self._edge_data_dir_edit = QtWidgets.QLineEdit(box)
        self._edge_data_dir_edit.setReadOnly(True)
        browse_btn = QtWidgets.QPushButton("Browse", box)
        browse_btn.clicked.connect(self._choose_edge_data_dir)
        row.addWidget(self._edge_data_dir_edit)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        self._edge_data_info_label = QtWidgets.QLabel("No dataset loaded", box)
        layout.addWidget(self._edge_data_info_label)

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

    def _build_edge_epoch_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Epoch")
        layout = QtWidgets.QVBoxLayout(box)

        self._edge_epoch_combo = QtWidgets.QComboBox(box)
        self._edge_epoch_combo.currentIndexChanged.connect(self._on_edge_epoch_selected)
        layout.addWidget(self._edge_epoch_combo)

        nav_row = QtWidgets.QHBoxLayout()
        self._edge_prev_btn = QtWidgets.QPushButton("Prev", box)
        self._edge_next_btn = QtWidgets.QPushButton("Next", box)
        clear_btn = QtWidgets.QPushButton("Clear Selection", box)
        self._edge_prev_btn.clicked.connect(lambda: self._step_edge_epoch(-1, clear_current=False))
        self._edge_next_btn.clicked.connect(lambda: self._step_edge_epoch(1, clear_current=False))
        clear_btn.clicked.connect(self._clear_edge_selection_current)
        nav_row.addWidget(self._edge_prev_btn)
        nav_row.addWidget(self._edge_next_btn)
        nav_row.addWidget(clear_btn)
        layout.addLayout(nav_row)

        return box

    def _build_edge_width_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Epoch Width")
        layout = QtWidgets.QHBoxLayout(box)
        layout.addWidget(QtWidgets.QLabel("Width (s)", box))
        self._edge_width_spin = QtWidgets.QDoubleSpinBox(box)
        self._edge_width_spin.setRange(0.1, 10_000.0)
        self._edge_width_spin.setDecimals(3)
        self._edge_width_spin.setSingleStep(0.1)
        self._edge_width_spin.setValue(5.0)
        layout.addWidget(self._edge_width_spin)
        return box

    def _build_edge_status_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Status")
        layout = QtWidgets.QVBoxLayout(box)
        self._edge_status_log = QtWidgets.QPlainTextEdit(box)
        self._edge_status_log.setReadOnly(True)
        layout.addWidget(self._edge_status_log)
        return box

    def _build_global_status_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Status")
        layout = QtWidgets.QVBoxLayout(box)
        self._status_log = QtWidgets.QPlainTextEdit(box)
        self._status_log.setReadOnly(True)
        self._status_log.setMaximumHeight(110)
        layout.addWidget(self._status_log)
        return box

    def _build_bio_dataset_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Dataset")
        layout = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        self._bio_data_dir_edit = QtWidgets.QLineEdit(box)
        self._bio_data_dir_edit.setReadOnly(True)
        browse_btn = QtWidgets.QPushButton("Browse", box)
        browse_btn.clicked.connect(self._choose_data_dir)
        row.addWidget(self._bio_data_dir_edit)
        row.addWidget(browse_btn)
        layout.addLayout(row)
        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(QtWidgets.QLabel("Run tag", box))
        self._bio_run_tag_edit = QtWidgets.QLineEdit(box)
        self._bio_run_tag_edit.setPlaceholderText("auto")
        run_row.addWidget(self._bio_run_tag_edit)
        layout.addLayout(run_row)
        self._bio_run_root_label = QtWidgets.QLabel("Run root: (set dataset/run tag)", box)
        layout.addWidget(self._bio_run_root_label)
        self._bio_use_cache_check = QtWidgets.QCheckBox("Use cache", box)
        self._bio_use_cache_check.setChecked(True)
        layout.addWidget(self._bio_use_cache_check)
        self._bio_data_info_label = QtWidgets.QLabel("No dataset loaded", box)
        layout.addWidget(self._bio_data_info_label)
        return box

    def _build_bio_epoch_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Cell Parameters")
        layout = QtWidgets.QVBoxLayout(box)

        self._bio_epoch_combo = QtWidgets.QComboBox(box)
        self._bio_epoch_combo.currentIndexChanged.connect(self._on_bio_epoch_selected)
        layout.addWidget(self._bio_epoch_combo)

        nav_row = QtWidgets.QHBoxLayout()
        self._bio_prev_btn = QtWidgets.QPushButton("Prev", box)
        self._bio_next_btn = QtWidgets.QPushButton("Next", box)
        self._bio_prev_btn.clicked.connect(lambda: self._step_bio_epoch(-1))
        self._bio_next_btn.clicked.connect(lambda: self._step_bio_epoch(1))
        nav_row.addWidget(self._bio_prev_btn)
        nav_row.addWidget(self._bio_next_btn)
        layout.addLayout(nav_row)

        width_row = QtWidgets.QHBoxLayout()
        width_row.addWidget(QtWidgets.QLabel("Window width (s)", box))
        self._bio_width_spin = QtWidgets.QDoubleSpinBox(box)
        self._bio_width_spin.setRange(0.1, 10_000.0)
        self._bio_width_spin.setDecimals(3)
        self._bio_width_spin.setSingleStep(0.1)
        self._bio_width_spin.setValue(5.0)
        width_row.addWidget(self._bio_width_spin)
        layout.addLayout(width_row)

        return box

    def _build_bio_selection_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Selected Windows")
        layout = QtWidgets.QVBoxLayout(box)

        button_row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add/Update Current", box)
        remove_btn = QtWidgets.QPushButton("Remove Current", box)
        add_btn.clicked.connect(self._bio_add_current_epoch_window)
        remove_btn.clicked.connect(self._bio_remove_current_epoch_window)
        button_row.addWidget(add_btn)
        button_row.addWidget(remove_btn)
        layout.addLayout(button_row)

        self._bio_selected_list = QtWidgets.QListWidget(box)
        self._bio_selected_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self._bio_selected_list)

        run_row = QtWidgets.QHBoxLayout()
        self._bio_calc_params_btn = QtWidgets.QPushButton("Calculate Cell Parameters", box)
        self._bio_calc_params_btn.clicked.connect(self._bio_calculate_cell_parameters)
        run_row.addWidget(self._bio_calc_params_btn)
        layout.addLayout(run_row)

        return box

    def _build_bio_synthetic_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Synthetic Datasets")
        layout = QtWidgets.QVBoxLayout(box)

        layout.addWidget(QtWidgets.QLabel("Synthetic config (JSON)", box))
        button_row = QtWidgets.QHBoxLayout()
        edit_btn = QtWidgets.QPushButton("Edit config", box)
        load_btn = QtWidgets.QPushButton("Load last synthetic config for this run", box)
        edit_btn.clicked.connect(self._bio_edit_synthetic_config)
        load_btn.clicked.connect(self._bio_load_last_synthetic_config)
        button_row.addWidget(edit_btn)
        button_row.addWidget(load_btn)
        layout.addLayout(button_row)
        self._bio_synth_editor = QtWidgets.QPlainTextEdit(box)
        self._bio_synth_editor.setPlainText(json.dumps(default_synthetic_config(), indent=2))
        layout.addWidget(self._bio_synth_editor)

        self._bio_generate_synth_btn = QtWidgets.QPushButton("Generate Synthetic Datasets", box)
        self._bio_generate_synth_btn.clicked.connect(self._bio_generate_synthetic)
        layout.addWidget(self._bio_generate_synth_btn)

        self._bio_bundle_list = QtWidgets.QListWidget(box)
        layout.addWidget(self._bio_bundle_list)
        return box

    def _build_bio_pgas_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("PGAS Config")
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(QtWidgets.QLabel("Constants file", box))
        self._bio_pgas_constants_combo = QtWidgets.QComboBox(box)
        layout.addWidget(self._bio_pgas_constants_combo)
        layout.addWidget(QtWidgets.QLabel("GParam file", box))
        self._bio_pgas_gparam_combo = QtWidgets.QComboBox(box)
        layout.addWidget(self._bio_pgas_gparam_combo)
        edit_btn = QtWidgets.QPushButton("Edit config", box)
        edit_btn.clicked.connect(self._bio_edit_pgas_constants)
        layout.addWidget(edit_btn)
        return box

    def _build_bio_train_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Train Custom Model")
        layout = QtWidgets.QVBoxLayout(box)

        family_row = QtWidgets.QHBoxLayout()
        self._bio_train_ens2_radio = QtWidgets.QRadioButton("ENS2", box)
        self._bio_train_cascade_radio = QtWidgets.QRadioButton("CASCADE", box)
        self._bio_train_ens2_radio.setChecked(True)
        self._bio_train_ens2_radio.toggled.connect(self._bio_on_train_family_changed)
        self._bio_train_cascade_radio.toggled.connect(self._bio_on_train_family_changed)
        family_row.addWidget(self._bio_train_ens2_radio)
        family_row.addWidget(self._bio_train_cascade_radio)
        layout.addLayout(family_row)

        layout.addWidget(QtWidgets.QLabel("Training config (JSON)", box))
        self._bio_train_editor = QtWidgets.QPlainTextEdit(box)
        self._bio_train_editor.setPlainText(self._bio_ens2_cfg_text)
        layout.addWidget(self._bio_train_editor)

        self._bio_train_btn = QtWidgets.QPushButton("Train Models", box)
        self._bio_train_btn.clicked.connect(self._bio_train_models)
        layout.addWidget(self._bio_train_btn)
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

    def _build_display_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Displayed Results")
        layout = QtWidgets.QVBoxLayout(box)

        self._display_list = QtWidgets.QListWidget(box)
        self._display_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._display_list.itemSelectionChanged.connect(self._on_display_selection_changed)
        layout.addWidget(self._display_list)

        btn_row = QtWidgets.QHBoxLayout()
        select_all = QtWidgets.QPushButton("Select All", box)
        clear = QtWidgets.QPushButton("Clear", box)
        select_all.clicked.connect(self._select_all_display_results)
        clear.clicked.connect(self._clear_display_results)
        btn_row.addWidget(select_all)
        btn_row.addWidget(clear)
        layout.addLayout(btn_row)

        return box

    def _build_method_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Methods")
        layout = QtWidgets.QVBoxLayout(box)

        self._pgas_check = QtWidgets.QCheckBox("BiophysSMC", box)
        self._biophys_check = QtWidgets.QCheckBox("BiophysML", box)
        self._cascade_check = QtWidgets.QCheckBox("Cascade", box)
        self._ens2_check = QtWidgets.QCheckBox("ENS2", box)
        self._pgas_check.setChecked(False)
        self._biophys_check.setChecked(True)
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

        layout.addWidget(QtWidgets.QLabel("Cascade model (installed)", box))
        self._cascade_model_list = QtWidgets.QListWidget(box)
        self._cascade_model_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._cascade_model_list.setMaximumHeight(100)
        layout.addWidget(self._cascade_model_list)

        layout.addWidget(QtWidgets.QLabel("Cascade download (repo)", box))
        self._cascade_download_combo = QtWidgets.QComboBox(box)
        layout.addWidget(self._cascade_download_combo)

        cascade_btn_row = QtWidgets.QHBoxLayout()
        self._cascade_refresh_btn = QtWidgets.QPushButton("Refresh Repo List", box)
        self._cascade_download_btn = QtWidgets.QPushButton("Download", box)
        self._cascade_refresh_btn.clicked.connect(self._update_cascade_available_models)
        self._cascade_download_btn.clicked.connect(self._download_selected_cascade)
        cascade_btn_row.addWidget(self._cascade_refresh_btn)
        cascade_btn_row.addWidget(self._cascade_download_btn)
        layout.addLayout(cascade_btn_row)

        layout.addWidget(QtWidgets.QLabel("ENS2 model (official)", box))
        self._ens2_official_list = QtWidgets.QListWidget(box)
        self._ens2_official_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._ens2_official_list.setMaximumHeight(100)
        layout.addWidget(self._ens2_official_list)

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
        self._biophys_list = QtWidgets.QListWidget(box)
        self._biophys_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._biophys_list.setMaximumHeight(100)
        layout.addWidget(self._biophys_list)

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

        edit_btn = QtWidgets.QPushButton("Edit config", box)
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
            self._load_dataset_dir(Path(directory))

    def _choose_edge_data_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if directory:
            self._load_dataset_dir(Path(directory))

    def _load_dataset_dir(self, directory: Path) -> None:
        directory = Path(directory)
        epochs, errors = scan_dataset_dir(directory)
        self._epoch_refs = epochs
        self._results_by_epoch.clear()
        self._errors_by_epoch.clear()
        self._display_selection_by_epoch.clear()
        self._display_available_by_epoch.clear()
        self._axis_labels_by_epoch.clear()
        if hasattr(self, "_display_list"):
            self._display_list.clear()
        self._current_epoch_index = None
        self._run_context = None
        self._bio_run_context = None
        self._edge_epoch_index = None
        self._edge_time = None
        self._edge_dff = None
        self._edge_spikes = None
        self._bio_epoch_index = None
        self._bio_time = None
        self._bio_dff = None
        self._bio_spikes = None
        self._bio_windows_by_epoch = {}
        self._bio_param_samples_by_epoch = {}
        self._bio_bundles = []

        self._data_dir_edit.setText(str(directory))
        if hasattr(self, "_edge_data_dir_edit"):
            self._edge_data_dir_edit.setText(str(directory))
        if hasattr(self, "_bio_data_dir_edit"):
            self._bio_data_dir_edit.setText(str(directory))
        try:
            default_context = build_run_context(directory, "", run_parent="spike_inference")
            self._run_tag_edit.setPlaceholderText(f"auto ({default_context.run_tag})")
            self._run_root_label.setText(f"Run root: {default_context.run_root}")
            bio_default_context = build_run_context(directory, "", run_parent="biophys_ml")
            if hasattr(self, "_bio_run_tag_edit"):
                self._bio_run_tag_edit.setPlaceholderText(f"auto ({bio_default_context.run_tag})")
                self._bio_run_root_label.setText(f"Run root: {bio_default_context.run_root}")
        except Exception:
            self._run_tag_edit.setPlaceholderText("auto")
            self._run_root_label.setText("Run root: (unavailable)")
            if hasattr(self, "_bio_run_tag_edit"):
                self._bio_run_tag_edit.setPlaceholderText("auto")
                self._bio_run_root_label.setText("Run root: (unavailable)")

        self._reset_edges_state()
        self._auto_load_edges_for_dir(directory)
        self._epoch_combo.blockSignals(True)
        self._epoch_combo.clear()
        self._epoch_list.clear()
        for epoch in epochs:
            self._epoch_combo.addItem(epoch.display, epoch)
            item = QtWidgets.QListWidgetItem(epoch.display)
            item.setData(Qt.UserRole, epoch)
            self._epoch_list.addItem(item)
        self._epoch_combo.blockSignals(False)
        if hasattr(self, "_edge_epoch_combo"):
            self._edge_epoch_combo.blockSignals(True)
            self._edge_epoch_combo.clear()
            for epoch in epochs:
                self._edge_epoch_combo.addItem(epoch.display, epoch)
            self._edge_epoch_combo.blockSignals(False)
        if hasattr(self, "_bio_epoch_combo"):
            self._bio_epoch_combo.blockSignals(True)
            self._bio_epoch_combo.clear()
            self._bio_selected_list.clear()
            self._bio_bundle_list.clear()
            for epoch in epochs:
                self._bio_epoch_combo.addItem(epoch.display, epoch)
            self._bio_epoch_combo.blockSignals(False)
            self._bio_load_windows_for_dir(directory)
            self._bio_refresh_selection_list()
        if errors:
            for err in errors:
                self._log(f"Dataset scan error: {err}")
        if epochs:
            self._data_info_label.setText(f"Loaded {len(epochs)} epochs from {directory}")
            self._epoch_combo.setCurrentIndex(0)
            self._on_epoch_selected(0)
            if hasattr(self, "_edge_data_info_label"):
                self._edge_data_info_label.setText(f"Loaded {len(epochs)} epochs from {directory}")
            if hasattr(self, "_edge_epoch_combo"):
                self._edge_epoch_combo.setCurrentIndex(0)
                self._on_edge_epoch_selected(0)
            if hasattr(self, "_bio_data_info_label"):
                self._bio_data_info_label.setText(f"Loaded {len(epochs)} epochs from {directory}")
            if hasattr(self, "_bio_epoch_combo"):
                self._bio_epoch_combo.setCurrentIndex(0)
                self._on_bio_epoch_selected(0)
        else:
            self._data_info_label.setText("No epochs found")
            self._figure.clear()
            self._canvas.draw()
            if hasattr(self, "_edge_data_info_label"):
                self._edge_data_info_label.setText("No epochs found")
            if hasattr(self, "_edge_figure"):
                self._edge_figure.clear()
                self._edge_canvas.draw()
            if hasattr(self, "_bio_data_info_label"):
                self._bio_data_info_label.setText("No epochs found")
            if hasattr(self, "_bio_figure"):
                self._bio_figure.clear()
                self._bio_canvas.draw()

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
        self._refresh_display_list_for_epoch(epoch.epoch_id, methods)
        methods = self._filter_methods_for_display(epoch.epoch_id, methods)
        method_labels = self._filter_method_labels_for_display(epoch.epoch_id, methods)
        plot_epoch(
            self._figure,
            time=time,
            dff=dff,
            methods=methods,
            method_labels=method_labels,
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
        new_idx = self._current_epoch_index + delta
        if new_idx >= len(self._epoch_refs):
            self._notify_epoch_boundary(reached_end=True, context="Spike Inference")
            return
        if new_idx < 0:
            self._notify_epoch_boundary(reached_end=False, context="Spike Inference")
            return
        self._epoch_combo.setCurrentIndex(new_idx)

    def _on_edge_epoch_selected(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._epoch_refs):
            return
        self._edge_epoch_index = idx
        epoch = self._epoch_refs[idx]
        try:
            time, dff, spikes = self._data_manager.load_epoch(epoch)
        except Exception as exc:
            self._log(f"Failed to load epoch: {exc}")
            return
        self._edge_time = time
        self._edge_dff = dff
        self._edge_spikes = spikes
        self._render_edge_plot()

    def _step_edge_epoch(self, delta: int, *, clear_current: bool) -> None:
        if not self._epoch_refs:
            return
        if self._edge_epoch_index is None:
            self._edge_epoch_index = 0
        if clear_current:
            self._clear_edge_selection_current()
        new_idx = self._edge_epoch_index + delta
        if new_idx >= len(self._epoch_refs):
            self._notify_epoch_boundary(reached_end=True, context="Edge Selection")
            return
        if new_idx < 0:
            self._notify_epoch_boundary(reached_end=False, context="Edge Selection")
            return
        self._edge_epoch_combo.setCurrentIndex(new_idx)

    def _clear_edge_selection_current(self) -> None:
        if self._edge_epoch_index is None or not self._epoch_refs:
            return
        epoch = self._epoch_refs[self._edge_epoch_index]
        self._set_edges_for_epoch(epoch, None)
        self._render_edge_plot()

    def _on_bio_epoch_selected(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._epoch_refs):
            return
        self._bio_epoch_index = idx
        epoch = self._epoch_refs[idx]
        try:
            time, dff, spikes = self._data_manager.load_epoch(epoch)
        except Exception as exc:
            self._log(f"Failed to load epoch: {exc}")
            return
        self._bio_time = time
        self._bio_dff = dff
        self._bio_spikes = spikes
        self._render_bio_plot()

    def _step_bio_epoch(self, delta: int) -> None:
        if not self._epoch_refs:
            return
        if self._bio_epoch_index is None:
            self._bio_epoch_index = 0
        new_idx = self._bio_epoch_index + delta
        if new_idx >= len(self._epoch_refs):
            self._notify_epoch_boundary(reached_end=True, context="Biophys ML")
            return
        if new_idx < 0:
            self._notify_epoch_boundary(reached_end=False, context="Biophys ML")
            return
        self._bio_epoch_combo.setCurrentIndex(new_idx)

    def _bio_current_epoch(self) -> Optional[EpochRef]:
        if self._bio_epoch_index is None:
            return None
        if self._bio_epoch_index < 0 or self._bio_epoch_index >= len(self._epoch_refs):
            return None
        return self._epoch_refs[self._bio_epoch_index]

    def _bio_default_window(self) -> Optional[List[float]]:
        if self._bio_time is None or self._bio_time.size == 0:
            return None
        width = float(self._bio_width_spin.value())
        start = float(self._bio_time[0])
        end = float(min(self._bio_time[-1], start + width))
        return [start, end]

    def _bio_add_current_epoch_window(self) -> None:
        epoch = self._bio_current_epoch()
        if epoch is None:
            return
        row = self._bio_windows_by_epoch.get(epoch.epoch_id)
        if row is None:
            row = self._bio_default_window()
            if row is None:
                return
        self._bio_windows_by_epoch[epoch.epoch_id] = [float(row[0]), float(row[1])]
        self._bio_save_windows()
        self._bio_refresh_selection_list()
        self._render_bio_plot()

    def _bio_remove_current_epoch_window(self) -> None:
        epoch = self._bio_current_epoch()
        if epoch is None:
            return
        if epoch.epoch_id in self._bio_windows_by_epoch:
            del self._bio_windows_by_epoch[epoch.epoch_id]
            self._bio_save_windows()
            self._bio_refresh_selection_list()
            self._render_bio_plot()

    def _bio_refresh_selection_list(self) -> None:
        if not hasattr(self, "_bio_selected_list"):
            return
        self._bio_selected_list.clear()
        by_id = {e.epoch_id: e for e in self._epoch_refs}
        for epoch_id, row in self._bio_windows_by_epoch.items():
            epoch = by_id.get(epoch_id)
            if epoch is None:
                continue
            item = QtWidgets.QListWidgetItem(
                f"{epoch.display} [{row[0]:.3f}, {row[1]:.3f}]"
            )
            item.setData(Qt.UserRole, epoch)
            self._bio_selected_list.addItem(item)

    def _bio_selected_epochs(self) -> List[EpochRef]:
        selected: List[EpochRef] = []
        by_id = {e.epoch_id: e for e in self._epoch_refs}
        for epoch_id in self._bio_windows_by_epoch.keys():
            epoch = by_id.get(epoch_id)
            if epoch is not None:
                selected.append(epoch)
        return selected

    def _bio_windows_path(self) -> Optional[Path]:
        data_dir = self._data_dir_edit.text().strip()
        if not data_dir:
            return None
        edges_dir = Path(data_dir) / "edges"
        edges_dir.mkdir(parents=True, exist_ok=True)
        return edges_dir / "biophys_ml_windows.json"

    def _bio_load_windows_for_dir(self, directory: Path) -> None:
        path = Path(directory) / "edges" / "biophys_ml_windows.json"
        if not path.exists():
            self._bio_windows_by_epoch = {}
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            out: Dict[str, List[float]] = {}
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        arr = np.asarray(value, dtype=float)
                        if np.all(np.isfinite(arr)) and arr[1] > arr[0]:
                            out[str(key)] = [float(arr[0]), float(arr[1])]
            self._bio_windows_by_epoch = out
        except Exception as exc:
            self._bio_windows_by_epoch = {}
            self._log(f"Failed to load biophys window file: {exc}")

    def _bio_save_windows(self) -> None:
        path = self._bio_windows_path()
        if path is None:
            return
        path.write_text(json.dumps(self._bio_windows_by_epoch, indent=2) + "\n", encoding="utf-8")

    def _bio_on_plot_clicked(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        epoch = self._bio_current_epoch()
        if epoch is None or self._bio_time is None or self._bio_time.size == 0:
            return
        width = float(self._bio_width_spin.value())
        start_idx = int(np.argmin(np.abs(self._bio_time - float(event.xdata))))
        start = float(self._bio_time[start_idx])
        target_end = min(float(self._bio_time[-1]), start + width)
        end_idx = int(np.argmin(np.abs(self._bio_time - target_end)))
        end = float(self._bio_time[end_idx])
        if end <= start:
            end = float(self._bio_time[min(start_idx + 1, self._bio_time.size - 1)])
        self._bio_windows_by_epoch[epoch.epoch_id] = [start, end]
        self._bio_save_windows()
        self._bio_refresh_selection_list()
        self._render_bio_plot()

    def _render_bio_plot(self) -> None:
        if self._bio_time is None or self._bio_dff is None:
            return
        self._bio_figure.clear()
        ax = self._bio_figure.add_subplot(1, 1, 1)
        ax.plot(self._bio_time, self._bio_dff, color="black", linewidth=1.0)
        self._plot_edge_spikes(ax, self._bio_time, self._bio_spikes)
        epoch = self._bio_current_epoch()
        if epoch is not None:
            row = self._bio_windows_by_epoch.get(epoch.epoch_id)
            if row is not None and len(row) == 2:
                start, end = float(row[0]), float(row[1])
                if end > start:
                    ax.axvline(start, color="#009E73", linewidth=1.2)
                    ax.axvline(end, color="#009E73", linewidth=1.2)
                    ax.axvspan(start, end, color="#009E73", alpha=0.15)
        ax.set_title(epoch.display if epoch is not None else "Biophys ML")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("dF/F")
        ax.grid(True, alpha=0.2)
        self._bio_figure.tight_layout()
        if self._bio_click_cid is not None:
            self._bio_canvas.mpl_disconnect(self._bio_click_cid)
        self._bio_click_cid = self._bio_canvas.mpl_connect("button_press_event", self._bio_on_plot_clicked)
        self._bio_canvas.draw()

    def _build_bio_pgas_settings(self) -> Optional[InferenceSettings]:
        constants_path = self._bio_current_pgas_constants_path()
        gparam_path = self._bio_current_pgas_gparam_path()
        if constants_path is None or not constants_path.exists():
            self._log("Select a valid PGAS constants file.")
            return None
        if gparam_path is None or not gparam_path.exists():
            self._log("Select a valid PGAS gparam file.")
            return None
        return InferenceSettings(
            run_cascade=False,
            run_ens2=False,
            run_pgas=True,
            run_biophys=False,
            neuron_type=self._neuron_combo.currentText(),
            use_cache=self._bio_use_cache_check.isChecked(),
            cascade_model_folder=CASCADE_ROOT,
            cascade_model_names=[],
            ens2_models=[],
            biophys_models=[],
            pgas_constants_file=constants_path,
            pgas_gparam_file=gparam_path,
        )

    def _bio_calculate_cell_parameters(self) -> None:
        if self._bio_pgas_worker and self._bio_pgas_worker.isRunning():
            self._log("Biophys PGAS job already running.")
            return
        epochs = self._bio_selected_epochs()
        if not epochs:
            self._log("Add at least one epoch/window in Biophys ML.")
            return
        if not self._ensure_bio_run_context():
            return
        settings = self._build_bio_pgas_settings()
        if settings is None:
            return
        assert self._bio_run_context is not None
        ensure_run_dirs(self._bio_run_context)
        self._apply_device_preference()
        self._bio_calc_params_btn.setEnabled(False)
        self._bio_pgas_worker = BioMlPgasWorker(
            epochs=epochs,
            windows_by_epoch=self._bio_windows_by_epoch,
            data_manager=self._data_manager,
            settings=settings,
            context=self._bio_run_context,
        )
        self._bio_pgas_worker.status.connect(self._log)
        self._bio_pgas_worker.finished.connect(self._bio_on_pgas_finished)
        self._bio_pgas_worker.start()

    def _bio_on_pgas_finished(self, param_paths: dict, errors: dict) -> None:
        self._bio_calc_params_btn.setEnabled(True)
        if param_paths:
            self._bio_param_samples_by_epoch.update({str(k): str(v) for k, v in param_paths.items()})
            self._log(f"[biophys_ml] Collected {len(param_paths)} param_samples file(s).")
            self._bio_save_param_index()
        for epoch_id, err in errors.items():
            self._log(f"[biophys_ml] {epoch_id}: {err}")

    def _bio_param_index_path(self) -> Optional[Path]:
        if self._bio_run_context is None:
            return None
        out_dir = self._bio_run_context.run_root / "biophys_ml"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / "param_samples_index.json"

    def _bio_save_param_index(self) -> None:
        path = self._bio_param_index_path()
        if path is None:
            return
        payload = {
            "run_tag": self._bio_run_context.run_tag if self._bio_run_context else "",
            "params_by_epoch": self._bio_param_samples_by_epoch,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _bio_load_param_index_from_cache(self, *, log_missing: bool = False) -> None:
        if self._bio_run_context is None:
            return
        path = self._bio_param_index_path()
        if path is None or not path.exists():
            if log_missing:
                self._log("No cached param_samples index found for this Biophys run.")
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            params = payload.get("params_by_epoch", {}) if isinstance(payload, dict) else {}
            if not isinstance(params, dict):
                raise ValueError("params_by_epoch must be a dict.")
            loaded: Dict[str, str] = {}
            for epoch_id, param_path in params.items():
                p = Path(str(param_path))
                if p.exists():
                    loaded[str(epoch_id)] = str(p)
            self._bio_param_samples_by_epoch = loaded
            self._log(f"[biophys_ml] Loaded {len(loaded)} cached param_samples entries.")
        except Exception as exc:
            self._log(f"[biophys_ml] Failed to load cached param index: {exc}")

    def _bio_generate_synthetic(self) -> None:
        if self._bio_synth_worker and self._bio_synth_worker.isRunning():
            self._log("Synthetic generation is already running.")
            return
        if not self._ensure_bio_run_context():
            return
        assert self._bio_run_context is not None
        ensure_run_dirs(self._bio_run_context)
        if self._bio_use_cache_check.isChecked():
            self._bio_load_param_index_from_cache(log_missing=False)
        epochs = self._bio_selected_epochs()
        param_paths: List[Path] = []
        for epoch in epochs:
            path = self._bio_param_samples_by_epoch.get(epoch.epoch_id)
            if path:
                p = Path(path)
                if p.exists():
                    param_paths.append(p)
        if not param_paths:
            if self._bio_use_cache_check.isChecked():
                self._log("No param_samples files available in cache for this Biophys run. Run 'Calculate Cell Parameters' or switch run tag.")
            else:
                self._log("No param_samples files available in session. Run 'Calculate Cell Parameters' first.")
            return
        try:
            config = json.loads(self._bio_synth_editor.toPlainText())
            if not isinstance(config, dict):
                raise ValueError("Synthetic config must be a JSON object.")
        except Exception as exc:
            self._log(f"Synthetic config parse error: {exc}")
            return
        run_tag = f"bio_ml_{self._bio_run_context.run_tag}"
        self._bio_generate_synth_btn.setEnabled(False)
        self._bio_synth_worker = BioMlSyntheticWorker(
            param_samples=param_paths,
            run_root=self._bio_run_context.run_root,
            run_tag=run_tag,
            synthetic_config=config,
        )
        self._bio_synth_worker.status.connect(self._log)
        self._bio_synth_worker.finished.connect(self._bio_on_synth_finished)
        self._bio_synth_worker.start()

    def _bio_edit_synthetic_config(self) -> None:
        if not self._ensure_bio_run_context():
            return
        assert self._bio_run_context is not None
        ensure_run_dirs(self._bio_run_context)
        cfg_dir = self._bio_run_context.run_root / "biophys_ml"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / "synthetic_config.json"
        current_text = self._bio_synth_editor.toPlainText()
        if not current_text.strip():
            current_text = json.dumps(default_synthetic_config(), indent=2)
        cfg_path.write_text(current_text, encoding="utf-8")
        dialog = ConfigEditorDialog(cfg_path, self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        edited = dialog.text()
        self._bio_synth_editor.setPlainText(edited)
        cfg_path.write_text(edited, encoding="utf-8")
        self._log(f"Saved synthetic config to {cfg_path}")

    def _bio_load_last_synthetic_config(self) -> None:
        if not self._ensure_bio_run_context():
            return
        assert self._bio_run_context is not None
        cfg_path = self._bio_run_context.run_root / "biophys_ml" / "synthetic_config.json"
        if not cfg_path.exists():
            self._log(f"No synthetic config found for this run at {cfg_path}")
            return
        try:
            text = cfg_path.read_text(encoding="utf-8")
            json.loads(text)
            self._bio_synth_editor.setPlainText(text)
            self._log(f"Loaded synthetic config from {cfg_path}")
        except Exception as exc:
            self._log(f"Failed to load synthetic config: {exc}")

    def _bio_on_synth_finished(self, bundles: list, error: str) -> None:
        self._bio_generate_synth_btn.setEnabled(True)
        if error:
            self._log(f"[biophys_ml] Synthetic generation failed: {error}")
            return
        self._bio_bundles = [dict(b) for b in bundles]
        self._bio_bundle_list.clear()
        for b in self._bio_bundles:
            bundle_id = str(b.get("bundle_id", "unknown"))
            synth_dirs = b.get("synth_dirs", [])
            n_dirs = len(synth_dirs) if isinstance(synth_dirs, list) else 0
            self._bio_bundle_list.addItem(f"{bundle_id} ({n_dirs} synth dirs)")
        self._log(f"[biophys_ml] Generated {len(self._bio_bundles)} synthetic bundle(s).")

    def _bio_on_train_family_changed(self, checked: bool) -> None:
        if not checked:
            return
        self._bio_cascade_cfg_text = self._bio_cascade_cfg_text or json.dumps(default_cascade_train_config(), indent=2)
        self._bio_ens2_cfg_text = self._bio_ens2_cfg_text or json.dumps(default_ens2_train_config(), indent=2)
        current = self._bio_train_editor.toPlainText().strip()
        if self._bio_train_cascade_radio.isChecked():
            self._bio_ens2_cfg_text = current or self._bio_ens2_cfg_text
            self._bio_train_editor.setPlainText(self._bio_cascade_cfg_text)
        else:
            self._bio_cascade_cfg_text = current or self._bio_cascade_cfg_text
            self._bio_train_editor.setPlainText(self._bio_ens2_cfg_text)

    def _bio_train_models(self) -> None:
        if self._bio_train_worker and self._bio_train_worker.isRunning():
            self._log("Biophys model training already running.")
            return
        if not self._ensure_bio_run_context():
            return
        assert self._bio_run_context is not None
        if not self._bio_bundles:
            bundle_file = self._bio_run_context.run_root / "biophys_ml" / "synthetic_bundles.json"
            if bundle_file.exists():
                try:
                    payload = json.loads(bundle_file.read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        self._bio_bundles = [dict(p) for p in payload if isinstance(p, dict)]
                except Exception:
                    pass
        if not self._bio_bundles:
            self._log("No synthetic bundles found. Generate synthetic datasets first.")
            return
        family = "cascade" if self._bio_train_cascade_radio.isChecked() else "ens2"
        text = self._bio_train_editor.toPlainText()
        try:
            cfg = json.loads(text) if text.strip() else {}
            if not isinstance(cfg, dict):
                raise ValueError("Training config must be a JSON object.")
        except Exception as exc:
            self._log(f"Training config parse error: {exc}")
            return
        if family == "ens2":
            self._bio_ens2_cfg_text = text
        else:
            self._bio_cascade_cfg_text = text
        self._bio_train_btn.setEnabled(False)
        self._bio_train_worker = BioMlTrainWorker(
            bundles=self._bio_bundles,
            run_root=self._bio_run_context.run_root,
            model_family=family,
            model_root=BIOPHYS_ROOT,
            train_config=cfg,
        )
        self._bio_train_worker.status.connect(self._log)
        self._bio_train_worker.finished.connect(self._bio_on_train_finished)
        self._bio_train_worker.start()

    def _bio_on_train_finished(self, records: list, error: str) -> None:
        self._bio_train_btn.setEnabled(True)
        if error:
            self._log(f"[biophys_ml] Training failed: {error}")
            return
        self._log(f"[biophys_ml] Trained {len(records)} model(s).")
        for rec in records:
            model_name = rec.get("model_name")
            model_dir = rec.get("model_dir")
            if model_name and model_dir:
                self._log(f"[biophys_ml] {model_name} -> {model_dir}")
        self._refresh_biophys_models()

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

        selected = self._selected_values(self._cascade_model_list)
        self._cascade_model_list.clear()
        for name in local:
            item = QtWidgets.QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            self._cascade_model_list.addItem(item)
            if name in selected:
                item.setSelected(True)
        if self._cascade_model_list.count() and not self._cascade_model_list.selectedItems():
            self._cascade_model_list.item(0).setSelected(True)

        self._cascade_download_combo.clear()
        for name in available:
            self._cascade_download_combo.addItem(name)

    def _update_cascade_available_models(self) -> None:
        if self._download_worker and self._download_worker.isRunning():
            self._log("Another download is in progress.")
            return
        self._download_worker = DownloadWorker("update_models", CASCADE_ROOT)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.start()
        self._log("Refreshing Cascade model list...")

    def _refresh_ens2_models(self) -> None:
        selected = self._selected_values(self._ens2_official_list)
        self._ens2_official_list.clear()
        for model_dir in list_ens2_model_dirs(ENS2_ROOT):
            label = format_model_dir_label(ENS2_ROOT, model_dir)
            item = QtWidgets.QListWidgetItem(label)
            item.setData(Qt.UserRole, model_dir)
            self._ens2_official_list.addItem(item)
            if str(model_dir) in selected:
                item.setSelected(True)
        if self._ens2_official_list.count() and not self._ens2_official_list.selectedItems():
            self._ens2_official_list.item(0).setSelected(True)

    def _refresh_biophys_models(self) -> None:
        selected = self._selected_values(self._biophys_list)
        self._biophys_list.clear()
        for model_dir in list_biophys_model_dirs(BIOPHYS_ROOT):
            label = format_model_dir_label(BIOPHYS_ROOT, model_dir)
            item = QtWidgets.QListWidgetItem(label)
            item.setData(Qt.UserRole, model_dir)
            self._biophys_list.addItem(item)
            if str(model_dir) in selected:
                item.setSelected(True)
        if self._biophys_list.count() and not self._biophys_list.selectedItems():
            self._biophys_list.item(0).setSelected(True)

    def _selected_values(self, widget: QtWidgets.QListWidget) -> set[str]:
        values: set[str] = set()
        for item in widget.selectedItems():
            data = item.data(Qt.UserRole)
            if isinstance(data, Path):
                values.add(str(data))
            elif data is not None:
                values.add(str(data))
            else:
                values.add(item.text())
        return values

    def _refresh_pgas_lists(self) -> None:
        self._pgas_constants_combo.clear()
        if hasattr(self, "_bio_pgas_constants_combo"):
            self._bio_pgas_constants_combo.clear()
        if PGAS_PARAMS_ROOT.exists():
            for path in sorted(PGAS_PARAMS_ROOT.glob("*.json")):
                self._pgas_constants_combo.addItem(path.name, path)
                if hasattr(self, "_bio_pgas_constants_combo"):
                    self._bio_pgas_constants_combo.addItem(path.name, path)
        self._pgas_gparam_combo.clear()
        if hasattr(self, "_bio_pgas_gparam_combo"):
            self._bio_pgas_gparam_combo.clear()
        if PGAS_GPARAM_ROOT.exists():
            for path in sorted(PGAS_GPARAM_ROOT.glob("*.dat")):
                self._pgas_gparam_combo.addItem(path.name, path)
                if hasattr(self, "_bio_pgas_gparam_combo"):
                    self._bio_pgas_gparam_combo.addItem(path.name, path)

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

    def _bio_edit_pgas_constants(self) -> None:
        path = self._bio_current_pgas_constants_path()
        if path is None:
            self._log("No Biophys ML PGAS constants file selected.")
            return
        dialog = ConfigEditorDialog(path, self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        if not self._ensure_bio_run_context():
            return
        assert self._bio_run_context is not None
        ensure_run_dirs(self._bio_run_context)
        temp_path = self._bio_run_context.pgas_temp_root / path.name
        temp_path.write_text(dialog.text(), encoding="utf-8")
        label = f"{path.name} (edited)"
        self._bio_pgas_constants_combo.addItem(label, temp_path)
        self._bio_pgas_constants_combo.setCurrentIndex(self._bio_pgas_constants_combo.count() - 1)
        self._log(f"Saved Biophys ML edited constants to {temp_path}")

    def _download_selected_cascade(self) -> None:
        model_name = self._cascade_download_combo.currentText().strip()
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

    def _bio_current_pgas_constants_path(self) -> Optional[Path]:
        data = self._bio_pgas_constants_combo.currentData()
        if isinstance(data, Path):
            return data
        if self._bio_pgas_constants_combo.currentText():
            return PGAS_PARAMS_ROOT / self._bio_pgas_constants_combo.currentText()
        return None

    def _bio_current_pgas_gparam_path(self) -> Optional[Path]:
        data = self._bio_pgas_gparam_combo.currentData()
        if isinstance(data, Path):
            return data
        if self._bio_pgas_gparam_combo.currentText():
            return PGAS_GPARAM_ROOT / self._bio_pgas_gparam_combo.currentText()
        return None

    def _ensure_run_context(self) -> bool:
        data_dir = self._data_dir_edit.text().strip()
        if not data_dir:
            self._log("Select a dataset directory first.")
            return False
        run_tag = self._run_tag_edit.text().strip()
        context = build_run_context(Path(data_dir), run_tag, run_parent="spike_inference")
        self._run_context = context
        self._run_root_label.setText(f"Run root: {context.run_root}")
        if not run_tag:
            self._run_tag_edit.setText(context.run_tag)
        return True

    def _ensure_bio_run_context(self) -> bool:
        data_dir = self._bio_data_dir_edit.text().strip()
        if not data_dir:
            self._log("Select a dataset directory first.")
            return False
        run_tag = self._bio_run_tag_edit.text().strip()
        context = build_run_context(Path(data_dir), run_tag, run_parent="biophys_ml")
        self._bio_run_context = context
        self._bio_run_root_label.setText(f"Run root: {context.run_root}")
        if not run_tag:
            self._bio_run_tag_edit.setText(context.run_tag)
        if self._bio_use_cache_check.isChecked():
            self._bio_load_param_index_from_cache(log_missing=False)
        return True

    def _collect_epoch_selection(self) -> List[EpochRef]:
        selected_rows = sorted({index.row() for index in self._epoch_list.selectedIndexes()})
        self._last_batch_rows = selected_rows
        if selected_rows:
            return [self._epoch_refs[row] for row in selected_rows if 0 <= row < len(self._epoch_refs)]
        self._last_batch_rows = []
        if self._current_epoch_index is None and self._epoch_refs:
            self._current_epoch_index = 0
        if self._current_epoch_index is None:
            return []
        return [self._epoch_refs[self._current_epoch_index]]

    def _method_base(self, method_key: str) -> str:
        token = str(method_key)
        if "::" in token:
            return token.split("::", 1)[0]
        return token

    def _method_variant(self, method_key: str) -> str:
        token = str(method_key)
        if "::" in token:
            return token.split("::", 1)[1]
        return ""

    def _method_sort_key(self, method_key: str) -> tuple[int, str]:
        base = self._method_base(method_key)
        try:
            idx = METHOD_ORDER.index(base)
        except ValueError:
            idx = len(METHOD_ORDER)
        return idx, str(method_key)

    def _method_display_label(self, method_key: str) -> str:
        base = self._method_base(method_key)
        variant = self._method_variant(method_key)
        label = self._family_display_name(base)
        if not variant:
            return label
        return f"{label} | {variant}"

    def _family_display_name(self, base_method: str) -> str:
        names = {
            "pgas": "BiophysSMC",
            "biophys_ml": "BiophysML",
            "cascade": "Cascade",
            "ens2": "ENS2",
        }
        return names.get(base_method, base_method)

    def _numbered_method_labels(
        self, method_keys: List[str]
    ) -> tuple[Dict[str, str], Dict[str, str]]:
        display_labels: Dict[str, str] = {}
        axis_labels: Dict[str, str] = {}
        family_index: Dict[str, int] = {}
        model_families = {"biophys_ml", "cascade", "ens2"}
        for method_key in method_keys:
            base = self._method_base(method_key)
            variant = self._method_variant(method_key)
            family = self._family_display_name(base)
            idx = family_index.get(base, 0) + 1
            family_index[base] = idx
            if base in model_families:
                axis_labels[method_key] = f"{family} {idx}"
                if variant:
                    display_labels[method_key] = f"{family} {idx}: {variant}"
                else:
                    display_labels[method_key] = f"{family} {idx}"
            else:
                axis_labels[method_key] = family
                if variant:
                    display_labels[method_key] = f"{family}: {variant}"
                else:
                    display_labels[method_key] = family
        return display_labels, axis_labels

    def _refresh_display_list_for_epoch(self, epoch_id: str, methods: Dict[str, object]) -> None:
        available = sorted(methods.keys(), key=self._method_sort_key)
        available_set = set(available)
        previous = self._display_available_by_epoch.get(epoch_id, set())
        selected = self._display_selection_by_epoch.get(epoch_id)
        if selected is None:
            selected = set(available_set)
        else:
            selected = set(selected)
            selected |= available_set - previous
            selected &= available_set
        self._display_selection_by_epoch[epoch_id] = selected
        self._display_available_by_epoch[epoch_id] = available_set
        display_labels, axis_labels = self._numbered_method_labels(available)
        self._axis_labels_by_epoch[epoch_id] = axis_labels

        self._updating_display_list = True
        try:
            self._display_list.clear()
            for key in available:
                item = QtWidgets.QListWidgetItem(display_labels.get(key, self._method_display_label(key)))
                item.setData(Qt.UserRole, key)
                self._display_list.addItem(item)
                if key in selected:
                    item.setSelected(True)
        finally:
            self._updating_display_list = False

    def _filter_methods_for_display(self, epoch_id: str, methods: Dict[str, object]) -> Dict[str, object]:
        selected = self._display_selection_by_epoch.get(epoch_id)
        if selected is None:
            return methods
        return {key: value for key, value in methods.items() if key in selected}

    def _filter_method_labels_for_display(
        self, epoch_id: str, methods: Dict[str, object]
    ) -> Dict[str, str]:
        labels = self._axis_labels_by_epoch.get(epoch_id, {})
        if not labels:
            return {}
        return {key: labels[key] for key in methods.keys() if key in labels}

    def _on_display_selection_changed(self) -> None:
        if self._updating_display_list:
            return
        if self._current_epoch_index is None or not self._epoch_refs:
            return
        epoch = self._epoch_refs[self._current_epoch_index]
        selected = {
            str(item.data(Qt.UserRole))
            for item in self._display_list.selectedItems()
            if item.data(Qt.UserRole) is not None
        }
        self._display_selection_by_epoch[epoch.epoch_id] = selected
        self._on_epoch_selected(self._current_epoch_index)

    def _select_all_display_results(self) -> None:
        if self._display_list.count() == 0:
            return
        self._display_list.selectAll()

    def _clear_display_results(self) -> None:
        self._display_list.clearSelection()

    def _selected_cascade_models(self) -> List[str]:
        models: List[str] = []
        for item in self._cascade_model_list.selectedItems():
            value = item.data(Qt.UserRole)
            name = str(value).strip() if value is not None else item.text().strip()
            if name:
                models.append(name)
        return models

    def _selected_ens2_models(self) -> List[ModelSpec]:
        models: List[ModelSpec] = []
        for item in self._ens2_official_list.selectedItems():
            path = item.data(Qt.UserRole)
            if not isinstance(path, Path):
                continue
            label = item.text().strip() or path.name
            models.append(ModelSpec(label=label, path=path))
        return models

    def _selected_biophys_models(self) -> List[BiophysModelSpec]:
        models: List[BiophysModelSpec] = []
        for item in self._biophys_list.selectedItems():
            path = item.data(Qt.UserRole)
            if not isinstance(path, Path):
                continue
            kind = detect_biophys_model_kind(path)
            if kind is None:
                continue
            label = item.text().strip() or path.name
            models.append(BiophysModelSpec(label=label, kind=kind, path=path))
        return models

    def _build_settings(self) -> Optional[InferenceSettings]:
        run_cascade = self._cascade_check.isChecked()
        run_ens2 = self._ens2_check.isChecked()
        run_biophys = self._biophys_check.isChecked()
        run_pgas = self._pgas_check.isChecked()
        if not (run_cascade or run_ens2 or run_biophys or run_pgas):
            self._log("Select at least one method to run.")
            return None

        cascade_root = CASCADE_ROOT
        if run_cascade:
            cascade_models = self._selected_cascade_models()
            if not cascade_models:
                self._log("Select at least one Cascade model.")
                return None
            for cascade_name in cascade_models:
                model_dir = cascade_root / cascade_name
                if (model_dir / "config.yaml").exists():
                    continue
                self._log(
                    f"Cascade model not found locally: {model_dir}. Download or select a different model."
                )
                return None
        else:
            cascade_models = []

        if run_ens2:
            ens2_models = self._selected_ens2_models()
            if not ens2_models:
                self._log("Select at least one ENS2 model directory.")
                return None
            for ens2_model in ens2_models:
                ens2_dir = ens2_model.path
                if (ens2_dir / "exc_ens2_pub.pt").exists() or (ens2_dir / "inh_ens2_pub.pt").exists():
                    continue
                self._log(f"ENS2 weights not found in {ens2_dir}.")
                return None
        else:
            ens2_models = []

        if run_biophys:
            biophys_models = self._selected_biophys_models()
            if not biophys_models:
                self._log("Select at least one BiophysML model directory.")
                return None
            for model in biophys_models:
                if detect_biophys_model_kind(model.path) is not None:
                    continue
                self._log(f"BiophysML model not recognized in {model.path}.")
                return None
        else:
            biophys_models = []

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
            cascade_model_names=cascade_models,
            ens2_models=ens2_models,
            biophys_models=biophys_models,
            pgas_constants_file=constants_path,
            pgas_gparam_file=gparam_path,
        )
        return settings

    def _run_inference(self) -> None:
        try:
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
            if self._last_batch_rows:
                rows_display = ", ".join(str(row + 1) for row in self._last_batch_rows)
                self._log(f"Batch rows (1-based): {rows_display}")
                labels = ", ".join(epoch.display for epoch in epochs)
                self._log(f"Batch labels: {labels}")

            batch_ids: List[str] = []
            for epoch in epochs:
                try:
                    batch_ids.append(str(epoch.epoch_id))
                except Exception:
                    batch_ids.append("<unknown>")
            self._log("Batch order: " + ", ".join(batch_ids))
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
        except Exception as exc:
            traceback.print_exc()
            self._log(f"Inference failed: {exc}")
            QtWidgets.QMessageBox.critical(self, "Inference Error", str(exc))
            self._run_btn.setEnabled(True)

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
                "cascade_model_names": list(settings.cascade_model_names),
                "ens2_models": [
                    {"label": model.label, "path": str(model.path)}
                    for model in settings.ens2_models
                ],
                "biophys_models": [
                    {"label": model.label, "kind": model.kind, "path": str(model.path)}
                    for model in settings.biophys_models
                ],
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

    def _notify_epoch_boundary(self, *, reached_end: bool, context: str) -> None:
        if reached_end:
            message = f"{context}: reached the last epoch."
            title = "Last Epoch"
        else:
            message = f"{context}: already at the first epoch."
            title = "First Epoch"
        self._log(message)
        QtWidgets.QMessageBox.information(self, title, message)

    def _render_edge_plot(self) -> None:
        if self._edge_time is None or self._edge_dff is None:
            return
        self._edge_figure.clear()
        ax = self._edge_figure.add_subplot(1, 1, 1)
        ax.plot(self._edge_time, self._edge_dff, color="black", linewidth=1.0)
        self._plot_edge_spikes(ax, self._edge_time, self._edge_spikes)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("dF/F")

        current_edges = self._get_edges_for_current_epoch()
        if current_edges is not None and np.all(np.isfinite(current_edges)):
            start, end = float(current_edges[0]), float(current_edges[1])
            if end < start:
                start, end = end, start
            ax.axvspan(start, end, color="#999999", alpha=0.2)
            ax.axvline(start, color="#666666", linestyle="--", linewidth=0.8)
            ax.axvline(end, color="#666666", linestyle="--", linewidth=0.8)

        self._edge_figure.tight_layout()
        if self._edge_click_cid is None:
            self._edge_click_cid = self._edge_canvas.mpl_connect("button_press_event", self._on_edge_plot_click)
        self._edge_canvas.draw()

    def _on_edge_plot_click(self, event) -> None:
        if event.inaxes is None or self._edge_time is None or self._edge_epoch_index is None:
            return
        if event.xdata is None:
            return
        time = self._edge_time
        x = float(event.xdata)
        idx_start = int(np.nanargmin(np.abs(time - x)))
        start = float(time[idx_start])
        width = float(self._edge_width_spin.value())
        target_end = start + max(width, 0.0)
        idx_end = int(np.nanargmin(np.abs(time - target_end)))
        end = float(time[idx_end])
        if end < start:
            end = start
        epoch = self._epoch_refs[self._edge_epoch_index]
        self._set_edges_for_epoch(epoch, np.array([start, end], dtype=float))
        self._render_edge_plot()

    def _plot_edge_spikes(self, ax, time: np.ndarray, spike_times: Optional[np.ndarray]) -> None:
        if spike_times is None:
            return
        spikes = np.asarray(spike_times, dtype=np.float64).ravel()
        if spikes.size == 0:
            return
        t_min = float(np.nanmin(time))
        t_max = float(np.nanmax(time))
        spikes = spikes[(spikes >= t_min) & (spikes <= t_max)]
        if spikes.size == 0:
            return
        for s in spikes:
            ax.axvline(float(s), color="#888888", linestyle=":", linewidth=0.6, alpha=0.9, zorder=0)

    def _get_edges_for_current_epoch(self) -> Optional[np.ndarray]:
        if self._edge_epoch_index is None or not self._epoch_refs:
            return None
        epoch = self._epoch_refs[self._edge_epoch_index]
        edges_arr = self._ensure_edges_array(epoch)
        if edges_arr is None:
            return None
        return edges_arr[epoch.epoch_index]

    def _ensure_edges_array(self, epoch: EpochRef) -> Optional[np.ndarray]:
        if self._edges_map is None:
            self._edges_map = {}
        key = epoch.file_path.stem
        edges_arr = self._edges_map.get(key)
        n_epochs = epoch.epoch_count
        if edges_arr is None or np.asarray(edges_arr).shape != (n_epochs, 2):
            new_arr = np.full((n_epochs, 2), np.nan, dtype=float)
            if edges_arr is not None:
                old = np.asarray(edges_arr, dtype=float)
                if old.ndim == 2 and old.shape[1] == 2:
                    n_copy = min(old.shape[0], n_epochs)
                    new_arr[:n_copy] = old[:n_copy]
            edges_arr = new_arr
            self._edges_map[key] = edges_arr
        return edges_arr

    def _set_edges_for_epoch(self, epoch: EpochRef, edges: Optional[np.ndarray]) -> None:
        edges_arr = self._ensure_edges_array(epoch)
        if edges_arr is None:
            return
        if edges is None:
            edges_arr[epoch.epoch_index] = np.array([np.nan, np.nan], dtype=float)
        else:
            edges_arr[epoch.epoch_index] = np.asarray(edges, dtype=float)
        self._save_edges_map()

    def _save_edges_map(self) -> None:
        if self._edges_map is None:
            return
        data_dir = self._data_dir_edit.text().strip()
        if not data_dir:
            return
        if self._edges_path is None:
            edges_dir = Path(data_dir) / "edges"
            edges_dir.mkdir(parents=True, exist_ok=True)
            self._edges_path = edges_dir / "edges.npy"
        np.save(self._edges_path, self._edges_map, allow_pickle=True)
        self._edges_path_edit.setText(str(self._edges_path))

    def _reset_edges_state(self) -> None:
        self._edges_map = None
        self._edges_path = None
        self._edges_enabled = False
        if hasattr(self, "_edges_check") and self._edges_check.isChecked():
            self._edges_check.setChecked(False)
        self._edges_path_edit.setText("")

    def _auto_load_edges_for_dir(self, directory: Path) -> None:
        edges_dir = Path(directory) / "edges"
        if not edges_dir.exists():
            return
        candidates = list(edges_dir.glob("edges*.npy"))
        if not candidates:
            return
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        self._load_edges_file(newest, log=False)

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
            if log:
                self._log(f"Loaded edges file: {path}")
            if self._edge_time is not None:
                self._render_edge_plot()
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

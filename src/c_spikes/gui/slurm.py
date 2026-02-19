from __future__ import annotations

import copy
import shlex
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


DEFAULT_SLURM_PROFILE: Dict[str, object] = {
    "job_name_template": "pgas_{run_tag}",
    "account": "",
    "partition": "",
    "qos": "",
    "time": "04:00:00",
    "mem": "32G",
    "cpus_per_task": 4,
    "gpus": 0,
    # "euo_pipefail" is stricter but can break module/conda init on some clusters.
    "strict_mode": "eo_pipefail",
    "module_setup": [
        "source /etc/profile.d/modules.sh",
        "module load anaconda3/2024.6",
    ],
    "env_activation": [
        "conda activate c_spikes",
    ],
    "command_template": "{command}",
    "log_dir_template": "{run_root}/slurm/logs",
    "output_log_template": "{log_dir}/{job_name}.out",
    "error_log_template": "{log_dir}/{job_name}.err",
    "extra_sbatch_options": [],
    "pre_commands": [],
    "post_commands": [],
}


def default_slurm_profile() -> Dict[str, object]:
    return copy.deepcopy(DEFAULT_SLURM_PROFILE)


def _as_string(field: str, value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_int(field: str, value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"Expected integer for '{field}', got: {value!r}") from exc


def _as_lines(field: str, value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    if not isinstance(value, list):
        raise ValueError(f"Expected list of shell command lines for '{field}'.")
    out: List[str] = []
    for item in value:
        token = str(item).strip()
        if token:
            out.append(token)
    return out


def _normalize_strict_mode(value: object) -> str:
    if value is None:
        return "eo_pipefail"
    token = str(value).strip().lower().replace("-", "_")
    alias = {
        "off": "off",
        "none": "off",
        "false": "off",
        "0": "off",
        "eo_pipefail": "eo_pipefail",
        "safe": "eo_pipefail",
        "euo_pipefail": "euo_pipefail",
        "strict": "euo_pipefail",
    }
    normalized = alias.get(token)
    if normalized is None:
        raise ValueError(
            "Invalid strict_mode. Use one of: off, eo_pipefail, euo_pipefail."
        )
    return normalized


def _strict_mode_line(mode: str) -> str:
    if mode == "off":
        return ":"
    if mode == "eo_pipefail":
        return "set -eo pipefail"
    if mode == "euo_pipefail":
        return "set -euo pipefail"
    raise ValueError(f"Unknown strict_mode mode: {mode!r}")


def _format_template(template: str, values: Mapping[str, str], *, field: str) -> str:
    try:
        return str(template).format_map(values).strip()
    except KeyError as exc:
        key = str(exc).strip("'")
        raise ValueError(f"Unknown template variable '{key}' in '{field}'.") from exc


def normalize_slurm_profile(profile: Mapping[str, object]) -> Dict[str, object]:
    merged = default_slurm_profile()
    merged.update(dict(profile))

    normalized: Dict[str, object] = {
        "job_name_template": _as_string("job_name_template", merged.get("job_name_template")),
        "account": _as_string("account", merged.get("account")),
        "partition": _as_string("partition", merged.get("partition")),
        "qos": _as_string("qos", merged.get("qos")),
        "time": _as_string("time", merged.get("time")),
        "mem": _as_string("mem", merged.get("mem")),
        "cpus_per_task": _as_int("cpus_per_task", merged.get("cpus_per_task")),
        "gpus": _as_int("gpus", merged.get("gpus")),
        "strict_mode": _normalize_strict_mode(merged.get("strict_mode")),
        "module_setup": _as_lines("module_setup", merged.get("module_setup")),
        "env_activation": _as_lines("env_activation", merged.get("env_activation")),
        "command_template": _as_string("command_template", merged.get("command_template")),
        "log_dir_template": _as_string("log_dir_template", merged.get("log_dir_template")),
        "output_log_template": _as_string("output_log_template", merged.get("output_log_template")),
        "error_log_template": _as_string("error_log_template", merged.get("error_log_template")),
        "extra_sbatch_options": _as_lines("extra_sbatch_options", merged.get("extra_sbatch_options")),
        "pre_commands": _as_lines("pre_commands", merged.get("pre_commands")),
        "post_commands": _as_lines("post_commands", merged.get("post_commands")),
    }

    if not normalized["job_name_template"]:
        normalized["job_name_template"] = "pgas_{run_tag}"
    if not normalized["command_template"]:
        normalized["command_template"] = "{command}"
    if not normalized["time"]:
        normalized["time"] = "04:00:00"
    if not normalized["mem"]:
        normalized["mem"] = "32G"
    if int(normalized["cpus_per_task"]) <= 0:
        raise ValueError("cpus_per_task must be >= 1.")
    if int(normalized["gpus"]) < 0:
        raise ValueError("gpus must be >= 0.")

    return normalized


def _append_sbatch_option(lines: List[str], option: str) -> None:
    token = option.strip()
    if not token:
        return
    if token.startswith("#SBATCH"):
        lines.append(token)
        return
    if not token.startswith("--"):
        token = f"--{token.lstrip('-')}"
    lines.append(f"#SBATCH {token}")


def render_sbatch_script(
    *,
    profile: Mapping[str, object],
    command: str,
    run_root: Path,
    data_dir: Path,
    run_tag: str,
) -> Tuple[str, str]:
    cfg = normalize_slurm_profile(profile)
    variables: Dict[str, str] = {
        "run_root": str(Path(run_root)),
        "data_dir": str(Path(data_dir)),
        "run_tag": str(run_tag),
        "command": str(command).strip(),
        "command_args": str(command).strip(),
    }

    job_name = _format_template(str(cfg["job_name_template"]), variables, field="job_name_template")
    if not job_name:
        raise ValueError("Resolved job_name is empty.")
    variables["job_name"] = job_name

    log_dir = _format_template(str(cfg["log_dir_template"]), variables, field="log_dir_template")
    if not log_dir:
        raise ValueError("Resolved log_dir is empty.")
    variables["log_dir"] = log_dir

    output_log = _format_template(str(cfg["output_log_template"]), variables, field="output_log_template")
    error_log = _format_template(str(cfg["error_log_template"]), variables, field="error_log_template")
    if not output_log or not error_log:
        raise ValueError("Resolved output/error log paths must be non-empty.")
    variables["output_log"] = output_log
    variables["error_log"] = error_log

    command_line = _format_template(str(cfg["command_template"]), variables, field="command_template")
    if not command_line:
        raise ValueError("Resolved command line is empty.")

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --time={cfg['time']}",
        f"#SBATCH --mem={cfg['mem']}",
        f"#SBATCH --cpus-per-task={int(cfg['cpus_per_task'])}",
        f"#SBATCH --output={output_log}",
        f"#SBATCH --error={error_log}",
    ]

    account = str(cfg["account"]).strip()
    partition = str(cfg["partition"]).strip()
    qos = str(cfg["qos"]).strip()
    gpus = int(cfg["gpus"])
    if account:
        lines.append(f"#SBATCH --account={account}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    if gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")

    for option in list(cfg["extra_sbatch_options"]):  # type: ignore[arg-type]
        _append_sbatch_option(lines, str(option))

    lines.extend(
        [
            "",
            _strict_mode_line(str(cfg["strict_mode"])),
            f"mkdir -p {shlex.quote(log_dir)}",
            "",
        ]
    )

    for cmd in list(cfg["module_setup"]):  # type: ignore[arg-type]
        lines.append(str(cmd))
    for cmd in list(cfg["env_activation"]):  # type: ignore[arg-type]
        lines.append(str(cmd))
    for cmd in list(cfg["pre_commands"]):  # type: ignore[arg-type]
        lines.append(str(cmd))

    lines.append(command_line)

    for cmd in list(cfg["post_commands"]):  # type: ignore[arg-type]
        lines.append(str(cmd))

    script_text = "\n".join(lines).rstrip() + "\n"
    return job_name, script_text


__all__ = [
    "DEFAULT_SLURM_PROFILE",
    "default_slurm_profile",
    "normalize_slurm_profile",
    "render_sbatch_script",
]

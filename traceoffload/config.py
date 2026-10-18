"""Pilot 配置加载、最小校验、默认路径与环境变量文件加载。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PILOT_CONFIG = Path("configs/pilot.yaml")


class PilotConfigError(ValueError):
    pass


def default_results_dir() -> Path:
    """默认实验输出目录：traceoffload/result（相对包位置解析，与当前工作目录无关）。

    所有运行产物（JSONL、summary、task trace 等）默认写入这里。
    """
    return Path(__file__).resolve().parent / "result"


def default_env_file() -> Path:
    """默认 .env 文件位置：项目根 configs/.env（相对包位置解析）。"""
    return Path(__file__).resolve().parent.parent / "configs" / ".env"


def load_env_file(path: Path | str | None = None) -> Path | None:
    """从 .env 文件加载环境变量（不覆盖已设置的环境变量）。

    默认读取项目根 configs/.env；文件不存在时返回 None 不做任何事。
    支持 `KEY=VALUE` 行、行首 `#` 注释与可选的单/双引号包裹值。
    """
    env_path = Path(path) if path is not None else default_env_file()
    if not env_path.is_file():
        return None
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
    return env_path


def load_pilot_config(path: Path | str = DEFAULT_PILOT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise PilotConfigError(f"pilot configuration not found: {config_path}")

    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise PilotConfigError("pilot configuration must contain a YAML mapping")

    for key_path in (
        ("pilot", "id"),
        ("agent", "source_path"),
        ("workload", "benchmark", "dataset"),
        ("models", "large", "model_id"),
        ("models", "small", "model_id"),
        ("evaluation", "workflow_noninferiority_epsilon"),
        ("evaluation", "unsafe_episode_upper_bound_delta"),
    ):
        _require_value(data, key_path)
    return data


def _require_value(config: dict[str, Any], key_path: tuple[str, ...]) -> None:
    value: Any = config
    for key in key_path:
        if not isinstance(value, dict) or key not in value:
            dotted_path = ".".join(key_path)
            raise PilotConfigError(f"missing required pilot setting: {dotted_path}")
        value = value[key]
    if value is None or value == "":
        dotted_path = ".".join(key_path)
        raise PilotConfigError(f"empty required pilot setting: {dotted_path}")


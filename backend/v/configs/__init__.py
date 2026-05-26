"""Pydantic settings models for every module. Single source of truth for config."""

from backend.v.configs.base import AppSettings, get_settings

__all__ = ["AppSettings", "get_settings"]

"""Load prompt templates from the YAML prompt file and render them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml

REQUIRED_TASKS = ("review_analysis", "insight_synthesis", "review_qa")


@dataclass(frozen=True)
class PromptPair:
    """A system prompt plus a user-message template for one task."""

    system: Template
    user: Template

    def render(self, system_vars: dict | None = None, **user_vars) -> tuple[str, str]:
        # substitute() (not safe_substitute) so a missing variable fails loudly.
        return self.system.substitute(system_vars or {}), self.user.substitute(user_vars)


class PromptLibrary:
    """All prompts used by the application, loaded from one YAML file."""

    def __init__(self, raw: dict):
        missing = [task for task in REQUIRED_TASKS if task not in raw]
        if missing:
            raise ValueError(f"Prompt file is missing tasks: {', '.join(missing)}")
        self.version = str(raw.get("version", "unversioned"))
        self._raw = raw
        self.tasks = {
            task: PromptPair(Template(raw[task]["system"]), Template(raw[task]["user"]))
            for task in REQUIRED_TASKS
        }

    @classmethod
    def from_file(cls, path: str | Path) -> PromptLibrary:
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def __getitem__(self, task: str) -> PromptPair:
        return self.tasks[task]

    def new_aspect_rule(self, allow_new: bool) -> str:
        key = "new_aspect_rule_on" if allow_new else "new_aspect_rule_off"
        return self._raw["review_analysis"][key]

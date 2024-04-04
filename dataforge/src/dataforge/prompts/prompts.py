from __future__ import annotations

import pkgutil
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from yaml import safe_load


@dataclass
class PromptConfig:
    name: str
    system: Template
    user: Template
    json_schema: dict[str, Any]

    @staticmethod
    def load_all(lang: str) -> list[PromptConfig]:
        data = pkgutil.get_data(__name__, str(Path("data") / f"qapairs_{lang}.yaml"))
        yaml_data = safe_load(data)
        return [
            PromptConfig(
                name=prompt["name"],
                system=Template(prompt["system"]),
                user=Template(prompt["user"]),
                json_schema=prompt["json_schema"],
            )
            for prompt in yaml_data["prompts"]
        ]


def generate_messages(
    prompt_config: PromptConfig,
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": prompt_config.system.safe_substitute(data),
        },
        {
            "role": "user",
            "content": prompt_config.user.safe_substitute(data),
        },
    ]

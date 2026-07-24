from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """An immutable, named prompt revision ready for a future renderer.

    The registry deliberately stores template text without rendering it. Context
    assembly and variable interpolation belong to later generation stages.
    """

    name: str
    version: str
    system_template: str
    user_template: str

    def __post_init__(self) -> None:
        _validate_non_blank("name", self.name)
        _validate_non_blank("version", self.version)
        _validate_non_blank("system_template", self.system_template)
        _validate_non_blank("user_template", self.user_template)


class PromptRegistry:
    """Keep prompt revisions and select the active revision explicitly.

    Registering a new revision never changes an existing active revision. This
    keeps a production prompt stable until a release decision calls ``activate``.
    """

    def __init__(self) -> None:
        self._prompts: dict[str, dict[str, PromptTemplate]] = {}
        self._active_versions: dict[str, str] = {}

    def register(self, prompt: PromptTemplate, *, activate: bool = False) -> None:
        """Register one unique prompt revision and optionally activate it."""

        versions = self._prompts.setdefault(prompt.name, {})
        if prompt.version in versions:
            msg = f"prompt {prompt.name!r} version {prompt.version!r} is already registered"
            raise ValueError(msg)

        versions[prompt.version] = prompt
        if activate:
            self._active_versions[prompt.name] = prompt.version

    def activate(self, name: str, *, version: str) -> None:
        """Make one already-registered revision the active revision for a name."""

        self._get_version(name, version)
        self._active_versions[name] = version

    def get(self, name: str, *, version: str | None = None) -> PromptTemplate:
        """Return an exact revision or the explicitly activated revision."""

        if version is not None:
            return self._get_version(name, version)

        active_version = self._active_versions.get(name)
        if active_version is None:
            msg = f"prompt {name!r} has no active version"
            raise LookupError(msg)
        return self._get_version(name, active_version)

    def versions(self, name: str) -> tuple[str, ...]:
        """Return registered versions in their registration order."""

        versions = self._prompts.get(name)
        if versions is None:
            msg = f"prompt {name!r} is not registered"
            raise LookupError(msg)
        return tuple(versions)

    def _get_version(self, name: str, version: str) -> PromptTemplate:
        versions = self._prompts.get(name)
        if versions is None or version not in versions:
            msg = f"prompt {name!r} version {version!r} is not registered"
            raise LookupError(msg)
        return versions[version]


def _validate_non_blank(field_name: str, value: str) -> None:
    if not value.strip():
        msg = f"{field_name} must not be blank"
        raise ValueError(msg)

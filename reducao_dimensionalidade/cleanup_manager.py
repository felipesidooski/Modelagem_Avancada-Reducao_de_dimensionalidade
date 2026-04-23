from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from reducao_dimensionalidade.config import DEFAULT_CLEANUP_REPO_URL


@dataclass(frozen=True)
class CleanupRepositoryManager:
    """Gerencia o clone local do projeto de pre-processamento."""

    project_root: Path
    cleanup_dir_name: str
    repo_url: str = DEFAULT_CLEANUP_REPO_URL

    @property
    def cleanup_dir(self) -> Path:
        return self.project_root / self.cleanup_dir_name

    def clone_or_refresh(self) -> Path:
        """Remove o clone local existente e clona o repositorio novamente."""

        target = self.cleanup_dir.resolve()
        root = self.project_root.resolve()
        self._assert_safe_target(target=target, root=root)

        if target.exists():
            shutil.rmtree(target)

        subprocess.run(
            ["git", "clone", self.repo_url, str(target)],
            cwd=str(root),
            check=True,
        )
        return target

    @staticmethod
    def _assert_safe_target(target: Path, root: Path) -> None:
        """Garante que a remocao fique limitada ao diretorio do projeto."""

        if target == root or root not in target.parents:
            raise ValueError(f"Diretorio de cleanup inseguro para remocao: {target}")


"""Proteina-Complexa package initialization."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _add_vendored_community_models_to_path() -> None:
    """Expose vendored community models (e.g. OpenFold) when running from source.

    This allows source-based runs (for example ``PYTHONPATH=src``) to import
    ``openfold`` from ``community_models/openfold`` without requiring a separate
    pip installation of OpenFold.
    """
    repo_root = Path(__file__).resolve().parents[2]
    community_models_path = repo_root / "community_models"

    if not community_models_path.is_dir():
        return

    community_models_str = str(community_models_path)
    if community_models_str not in sys.path:
        sys.path.insert(0, community_models_str)


_add_vendored_community_models_to_path()


def _add_foundry_atomworks_fallback_to_path() -> None:
    """Fallback to atomworks from foundry env when unavailable in current env.

    This keeps Complexa running in a dedicated environment (e.g. ``proteina``)
    while reusing the atomworks package installed in a separate ``foundry``
    environment on the same machine.
    """
    try:
        import atomworks  # noqa: F401

        return
    except Exception:
        pass

    # Lock core scientific stack to the active environment before temporarily
    # exposing foundry site-packages.
    for module_name in ("numpy", "scipy", "pandas"):
        try:
            __import__(module_name)
        except Exception:
            pass

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    home = Path.home()
    candidates: list[Path] = []

    env_override = os.environ.get("ATOMWORKS_SITE_PACKAGES")
    if env_override:
        candidates.append(Path(env_override).expanduser())

    candidates.extend(
        [
            home / "miniconda3" / "envs" / "foundry" / "lib" / f"python{py_ver}" / "site-packages",
            home / "miniforge3" / "envs" / "foundry" / "lib" / f"python{py_ver}" / "site-packages",
            home / "mambaforge" / "envs" / "foundry" / "lib" / f"python{py_ver}" / "site-packages",
        ]
    )

    for site_packages in candidates:
        if not site_packages.is_dir():
            continue
        if not (site_packages / "atomworks").is_dir():
            continue

        site_packages_str = str(site_packages)
        inserted = False
        bootstrapped = False
        if site_packages_str not in sys.path:
            # Temporarily prioritize foundry site-packages only for atomworks bootstrap.
            sys.path.insert(0, site_packages_str)
            inserted = True
        try:
            # Load compatible foundry versions into sys.modules.
            import biotite  # noqa: F401
            import atomworks  # noqa: F401
            bootstrapped = True
        except Exception:
            # Keep trying other candidate locations.
            pass
        finally:
            if inserted:
                try:
                    sys.path.remove(site_packages_str)
                except ValueError:
                    pass

        if bootstrapped:
            # Keep foundry packages available as a low-priority fallback for
            # atomworks transitive deps not installed in the active env.
            if site_packages_str not in sys.path:
                sys.path.append(site_packages_str)
            return


_add_foundry_atomworks_fallback_to_path()

# Apply patches early - before any other imports that might use atomworks/biotite
# import proteinfoundation.patches.atomworks_patches
# import proteinfoundation.patches.biotite_patches

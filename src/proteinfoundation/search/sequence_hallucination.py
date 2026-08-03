"""SequenceHallucination algorithm for protein generation.

This module provides sequence hallucination for improving protein structures
using ColabDesign's AlphaFold2 optimization pipeline.
"""

import inspect
import math
import os
import re
import shutil
import tempfile
import time
from typing import Any

import torch
from loguru import logger

from proteinfoundation.utils.pdb_utils import get_chain_ids_from_pdb, load_pdb, write_prot_to_pdb
from proteinfoundation.utils.tensor_utils import concat_dict_tensors

_DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "pae": 0.4,
    "plddt": 0.1,
    "i_pae": 0.1,
    "con": 1.0,
    "i_con": 1.0,
    "dgram_cce": 0.0,
    "rg": 0.3,
    "i_ptm": 0.05,
    "helix_binder": -0.3,
}

_JAX_BACKEND_CHOICES = {"auto", "cpu", "gpu", "mps", "metal"}


class SequenceHallucination:
    """SequenceHallucination refinement algorithm.

    Optimises binder sequences using ColabDesign's AF2 design pipeline.
    Three optional stages run in order:
      Stage 2-3: Softmax + one-hot optimisation (``enable_soft_optimization``)
      Stage 4:   PSSM semigreedy optimisation (``enable_greedy_optimization``)

    If a sample fails during refinement (e.g. residue count mismatch from
    ColabDesign), the original unrefined structure is kept and processing
    continues for remaining samples.
    """

    def __init__(self, proteina_instance: Any, inf_cfg: Any) -> None:
        self.proteina = proteina_instance
        self.inf_cfg = inf_cfg

    # ------------------------------------------------------------------
    # Hotspot parsing  (NEW -- original passed hotspot=None)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_hotspots(
        target_hotspot_mask: torch.Tensor | None,
        chain_index: torch.Tensor,
        res_mask: torch.Tensor,
    ) -> list[str] | None:
        """Convert a boolean hotspot mask tensor into ColabDesign hotspot strings.

        ColabDesign expects hotspot as a list of target-chain residue indices
        (0-based integers).  The mask is True for residues that are hotspots.

        Returns None when no hotspots are available.
        """
        if target_hotspot_mask is None:
            return None

        # BUG FIX: guard against fully-padded samples where res_mask is all
        # False -- indexing chain_index[res_mask.bool()][0] would crash.
        if not res_mask.any():
            return None

        mask_bool = target_hotspot_mask.bool() & res_mask.bool()
        if not mask_bool.any():
            return None

        target_chain_id = chain_index[res_mask.bool()][0].item()
        hotspot_indices: list[str] = []
        target_pos = 0
        for j in range(res_mask.shape[0]):
            if not res_mask[j]:
                continue
            if chain_index[j].item() == target_chain_id:
                if mask_bool[j]:
                    hotspot_indices.append(str(target_pos))
                target_pos += 1

        return hotspot_indices if hotspot_indices else None

    # ------------------------------------------------------------------
    # Loss configuration
    # ------------------------------------------------------------------

    _BUILTIN_WEIGHT_KEYS = {"pae", "plddt", "i_pae", "con", "i_con", "dgram_cce"}

    @staticmethod
    def _jax_devices(platform: str) -> list[Any]:
        """Return available JAX devices for a platform, or [] if unavailable.

        Different Apple PJRT plugins expose backend names as ``mps`` or
        ``metal`` (sometimes uppercase). Probe variants for robustness.
        """
        canonical = str(platform).strip()
        candidates = [canonical]
        if canonical.lower() == "metal":
            candidates += ["METAL", "metal"]
        elif canonical.lower() == "mps":
            candidates += ["MPS", "mps"]
        elif canonical.upper() == "METAL":
            candidates += ["metal"]
        elif canonical.upper() == "MPS":
            candidates += ["mps"]

        try:
            import jax
        except Exception as exc:
            logger.debug(f"Failed to import JAX while probing platform '{platform}': {exc}")
            return []

        last_exc: Exception | None = None
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                return list(jax.devices(candidate))
            except Exception as exc:
                last_exc = exc

        # Fallback: inspect all discovered devices and filter by platform.
        try:
            all_devices = list(jax.devices())
            target = canonical.lower()
            filtered = [d for d in all_devices if str(getattr(d, "platform", "")).lower() == target]
            if filtered:
                return filtered
            if target in {"metal", "mps"}:
                apple_like = [
                    d
                    for d in all_devices
                    if str(getattr(d, "platform", "")).lower() in {"metal", "mps"}
                    or str(getattr(d, "platform", "")).upper() in {"METAL", "MPS"}
                ]
                if apple_like:
                    return apple_like
        except Exception as exc:
            last_exc = exc

        if last_exc is not None:
            logger.debug(f"JAX platform '{platform}' unavailable: {last_exc}")
        return []

    @staticmethod
    def _is_apple_jax_device(device: Any) -> bool:
        """Heuristic check for Apple Metal/MPS-backed JAX devices."""
        platform = str(getattr(device, "platform", "")).lower()
        if platform in {"metal", "mps"}:
            return True
        text = str(device).lower()
        return "metal" in text or "mps" in text

    @staticmethod
    def _is_apple_unsupported_primitive_error(exc: Exception) -> bool:
        """Detect known Apple backend primitive gaps (e.g. eigh)."""
        msg = str(exc).lower()
        return "primitive 'eigh'" in msg and ("platform metal" in msg or "platform mps" in msg)

    @staticmethod
    def _is_apple_backend_tag(tag: str) -> bool:
        lower = str(tag).lower()
        return lower.startswith("mps") or lower.startswith("metal")

    @classmethod
    def _select_jax_device(cls, requested_backend: str) -> tuple[Any, str]:
        """Resolve JAX backend preference to a concrete JAX device.

        Supported values:
          - auto: prefer CUDA GPU, then Apple JAX backend (mps/metal), then CPU
          - gpu: require JAX GPU
          - mps: require Apple JAX backend (prefers jax-mps, then jax-metal)
          - metal: legacy alias for mps
          - cpu: force CPU
        """
        backend = str(requested_backend).lower().strip()
        if backend not in _JAX_BACKEND_CHOICES:
            logger.warning(
                f"Unknown refinement.jax_backend='{requested_backend}', "
                "falling back to 'auto'."
            )
            backend = "auto"

        has_cuda = torch.cuda.is_available()
        mps_backend = getattr(torch.backends, "mps", None)
        has_mps = bool(mps_backend and mps_backend.is_available())

        if backend == "cpu":
            # Force CPU platform to avoid accidental Apple PJRT plugin init.
            os.environ["JAX_PLATFORMS"] = "cpu"
            cpu_devices = cls._jax_devices("cpu")
            if not cpu_devices:
                raise RuntimeError("refinement.jax_backend=cpu requested but no JAX CPU device is available.")
            return cpu_devices[0], "cpu"

        if backend == "gpu":
            os.environ["JAX_PLATFORMS"] = "gpu,cpu"
            gpu_devices = cls._jax_devices("gpu")
            if not gpu_devices:
                raise RuntimeError("refinement.jax_backend=gpu requested but no JAX GPU device is available.")
            device_id = torch.cuda.current_device() if has_cuda else 0
            return gpu_devices[min(device_id, len(gpu_devices) - 1)], "gpu"

        if backend == "metal":
            logger.warning("refinement.jax_backend=metal is deprecated; use refinement.jax_backend=mps.")
            backend = "mps"

        if backend == "mps":
            if not has_mps:
                raise RuntimeError(
                    "refinement.jax_backend=mps requested but torch MPS is unavailable. "
                    "Check torch.backends.mps.is_available() or use refinement.jax_backend=cpu."
                )
            # Do not force JAX_PLATFORMS=mps before importing JAX: if the host only
            # has legacy jax-metal (platform METAL), JAX import would fail and block
            # fallback probing.
            os.environ.pop("JAX_PLATFORMS", None)
            mps_devices = cls._jax_devices("mps")
            if mps_devices:
                return mps_devices[0], "mps"

            os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")
            os.environ.pop("JAX_PLATFORMS", None)
            metal_devices = cls._jax_devices("metal")
            if metal_devices:
                return metal_devices[0], "metal-legacy"

            gpu_devices = cls._jax_devices("gpu")
            apple_like_gpu = [d for d in gpu_devices if cls._is_apple_jax_device(d)]
            if apple_like_gpu:
                return apple_like_gpu[0], "mps-via-gpu"

            raise RuntimeError(
                "refinement.jax_backend=mps requested but no Apple JAX backend was found. "
                "Install/configure jax-mps (or jax-metal legacy), or use refinement.jax_backend=auto/cpu."
            )

        # backend == "auto"
        if has_cuda:
            os.environ["JAX_PLATFORMS"] = "gpu,cpu"
            gpu_devices = cls._jax_devices("gpu")
            if gpu_devices:
                device_id = torch.cuda.current_device()
                return gpu_devices[min(device_id, len(gpu_devices) - 1)], "gpu"
            logger.warning("CUDA is available to PyTorch, but JAX has no GPU device. Trying Apple/CPU backends.")

        if has_mps:
            os.environ.pop("JAX_PLATFORMS", None)
            mps_devices = cls._jax_devices("mps")
            if mps_devices:
                return mps_devices[0], "mps"
            os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")
            os.environ.pop("JAX_PLATFORMS", None)
            metal_devices = cls._jax_devices("metal")
            if metal_devices:
                return metal_devices[0], "metal-legacy"
            gpu_devices = cls._jax_devices("gpu")
            apple_like_gpu = [d for d in gpu_devices if cls._is_apple_jax_device(d)]
            if apple_like_gpu:
                return apple_like_gpu[0], "mps-via-gpu"
            logger.warning("Torch MPS is available, but no Apple JAX backend is available. Falling back to CPU.")

        os.environ["JAX_PLATFORMS"] = "cpu"
        cpu_devices = cls._jax_devices("cpu")
        if cpu_devices:
            return cpu_devices[0], "cpu"

        raise RuntimeError("No JAX device is available for sequence_hallucination refinement.")

    @staticmethod
    def _set_builtin_weights(af_model: Any, loss_weights: dict[str, float]) -> None:
        """(Re-)set opt["weights"] for ColabDesign's built-in loss terms.

        Must be called after every ``prep_inputs`` because the internal
        ``restart()`` resets ``opt`` to its saved defaults.
        """
        af_model.opt["weights"].update(
            {k: loss_weights[k] for k in SequenceHallucination._BUILTIN_WEIGHT_KEYS if k in loss_weights}
        )

    @staticmethod
    def _register_loss_callbacks(af_model: Any, loss_weights: dict[str, float]) -> None:
        """Append custom loss callbacks to the AF2 model **once**.

        BUG FIX: Each ``add_*_loss`` call appends to
        ``af_model._callbacks["model"]["loss"]``.  ColabDesign never
        clears this list between ``prep_inputs`` calls.  The original
        code avoided this by creating a fresh af_model per sample; we
        now create the model once for performance, so callbacks must be
        registered exactly once -- calling this again would duplicate
        every callback and corrupt the loss.
        """
        from proteinfoundation.rewards.alphafold2_reward_utils import (
            add_helix_binder_loss,
            add_i_ptm_loss,
            add_rg_loss,
        )

        add_rg_loss(af_model, loss_weights.get("rg", 0.0))
        add_i_ptm_loss(af_model, loss_weights.get("i_ptm", 0.0))
        add_helix_binder_loss(af_model, loss_weights.get("helix_binder", 0.0))

    @staticmethod
    def _make_afdesign_model(*, num_recycles: int, device: Any) -> Any:
        """Create mk_afdesign_model with best-effort compatibility across versions.

        ColabDesign constructor kwargs differ between releases. We inspect the
        callable signature and drop unsupported kwargs so refinement keeps
        running on older/newer variants.
        """
        from colabdesign import mk_afdesign_model

        model_kwargs = {
            "protocol": "binder",
            "debug": False,
            "data_dir": os.environ.get("AF2_DIR"),
            "use_multimer": True,
            "num_recycles": num_recycles,
            "use_initial_guess": False,
            "use_initial_atom_pos": False,
            "best_metric": "loss",
            "device": device,
        }

        try:
            signature = inspect.signature(mk_afdesign_model)
            accepts_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
            )
            if not accepts_var_kwargs:
                accepted_keys = set(signature.parameters.keys())
                dropped_keys = sorted(k for k in model_kwargs if k not in accepted_keys)
                if dropped_keys:
                    logger.warning(
                        "mk_afdesign_model does not accept constructor kwargs "
                        f"{dropped_keys}; dropping them for compatibility."
                    )
                model_kwargs = {k: v for k, v in model_kwargs.items() if k in accepted_keys}
        except (TypeError, ValueError) as sig_exc:
            logger.debug(f"Could not inspect mk_afdesign_model signature: {sig_exc}")
        current_kwargs = dict(model_kwargs)
        for _ in range(4):
            try:
                return mk_afdesign_model(**current_kwargs)
            except AssertionError as exc:
                msg = str(exc)
                if "following inputs were not set" not in msg:
                    raise

                # Some ColabDesign versions accept **kwargs in signature but
                # validate supported keys at runtime. Drop unknown keys and
                # retry.
                # Extract key names from "{'k': v, ...}" even when values are
                # non-literal objects like CpuDevice(id=0).
                key_candidates = re.findall(r"'([^']+)':", msg)
                dropped_keys = [k for k in key_candidates if k in current_kwargs]

                if not dropped_keys:
                    raise

                for key in dropped_keys:
                    current_kwargs.pop(key, None)
                logger.warning(
                    "mk_afdesign_model rejected constructor kwargs "
                    f"{dropped_keys}; retrying without them."
                )

        return mk_afdesign_model(**current_kwargs)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def refine(
        self,
        sample_prots: dict[str, torch.Tensor],
        target_hotspot_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Refine protein samples using sequence hallucination.

        Args:
            sample_prots: Dictionary containing at least ``coors``,
                ``residue_type``, ``chain_index``, and ``mask``.  All
                other keys are preserved in the output.
            target_hotspot_mask: Optional boolean tensor of shape
                ``[batch_size, n_residues]`` marking target hotspot
                residues.

        Returns:
            A copy of *sample_prots* with ``coors`` and ``residue_type``
            replaced by the refined versions for samples that succeeded.
            Samples that failed refinement retain their original values.
        """
        ref_cfg = self.inf_cfg.refinement
        requested_jax_backend = ref_cfg.get("jax_backend", "auto")
        jax_device, resolved_jax_backend = self._select_jax_device(requested_jax_backend)
        logger.info(
            "Sequence hallucination JAX backend "
            f"(requested={requested_jax_backend}, resolved={resolved_jax_backend}, device={jax_device})"
        )

        bs = sample_prots["coors"].shape[0]
        n_hard_iters = ref_cfg.get("n_hard_iters", 5)
        n_temp_iters = ref_cfg.get("n_temp_iters", 45)
        n_greedy_iters = ref_cfg.get("n_greedy_iters", 15)
        n_recycles = ref_cfg.get("n_recycles", 3)
        enable_soft = ref_cfg.get("enable_soft_optimization", True)
        enable_greedy = ref_cfg.get("enable_greedy_optimization", True)
        greedy_percentage = ref_cfg.get("greedy_percentage", 1)

        # IMPROVEMENT: loss weights are now configurable via
        # refinement.loss_weights in the YAML; defaults match the
        # original hardcoded values exactly.
        user_weights = ref_cfg.get("loss_weights", {})
        if hasattr(user_weights, "items"):
            user_weights = dict(user_weights)
        else:
            user_weights = {}
        loss_weights = {**_DEFAULT_LOSS_WEIGHTS, **user_weights}

        logger.info(f"Refinement: {bs} samples, soft={enable_soft}, greedy={enable_greedy}")

        # IMPROVEMENT: temp dir is now cleaned up in a finally block;
        # the original left it behind.
        temp_dir = tempfile.mkdtemp()
        try:
            target_chain, binder_chain = None, None
            # BUG FIX (performance): the original created a new
            # mk_afdesign_model per sample which is correct but very
            # slow (reloads weights each time).  We create the model
            # once and guard against callback accumulation via the
            # _register_loss_callbacks / _set_builtin_weights split.
            af_model = self._make_afdesign_model(num_recycles=n_recycles, device=jax_device)

            refined_sample_prots: list[dict[str, torch.Tensor]] = []
            callbacks_registered = False
            cpu_fallback_model = None
            cpu_callbacks_registered = False

            for i in range(bs):
                t0 = time.time()

                coors = sample_prots["coors"][i]
                residue_type = sample_prots["residue_type"][i]
                chain_index_all = sample_prots.get("chain_index")
                chain_index = chain_index_all[i] if chain_index_all is not None else None
                res_mask = sample_prots["mask"][i].bool()
                n = int(res_mask.sum().item())

                # BUG FIX: wrap per-sample processing so a single
                # failure (e.g. ColabDesign residue mismatch) does not
                # abort the entire batch.
                try:
                    temp_pdb_path = os.path.join(temp_dir, f"temp_sample_{i}.pdb")
                    write_prot_to_pdb(
                        prot_pos=coors.detach().cpu().numpy(),
                        aatype=residue_type.detach().cpu().numpy(),
                        file_path=temp_pdb_path,
                        chain_index=(chain_index.detach().cpu().numpy() if chain_index is not None else None),
                        overwrite=True,
                        no_indexing=True,
                    )

                    if target_chain is None:
                        target_chain, binder_chain = get_chain_ids_from_pdb(temp_pdb_path)

                    # IMPROVEMENT: parse and pass hotspots (original
                    # hard-coded hotspot=None).
                    hotspot_mask_i = target_hotspot_mask[i] if target_hotspot_mask is not None else None
                    hotspot_list = self._parse_hotspots(hotspot_mask_i, chain_index, res_mask)

                    def _run_with_model(model, callbacks_flag, backend_tag: str):
                        model.prep_inputs(
                            pdb_filename=temp_pdb_path,
                            target_chain=target_chain,
                            binder_chain=binder_chain,
                            mode="wildtype",
                            rm_target=False,
                            rm_target_seq=False,
                            rm_target_sc=False,
                            hotspot=hotspot_list,
                            use_binder_template=True,
                            rm_template_ic=True,
                        )

                        if not callbacks_flag:
                            self._register_loss_callbacks(model, loss_weights)
                            callbacks_flag = True
                        self._set_builtin_weights(model, loss_weights)

                        if enable_soft:
                            logger.info(f"Sample {i + 1}/{bs} - Stage 2: Softmax optimisation ({backend_tag})")
                            model.design_soft(
                                n_temp_iters,
                                e_temp=1e-2,
                                models=[0],
                                num_models=1,
                                sample_models=False,
                                ramp_recycles=False,
                            )

                            logger.info(f"Sample {i + 1}/{bs} - Stage 3: One-hot optimisation ({backend_tag})")
                            model.design_hard(
                                n_hard_iters,
                                temp=1e-2,
                                models=[0],
                                num_models=1,
                                sample_models=False,
                                dropout=False,
                                ramp_recycles=False,
                            )

                        if enable_greedy:
                            logger.info(f"Sample {i + 1}/{bs} - Stage 4: PSSM semigreedy optimisation ({backend_tag})")
                            greedy_tries = math.ceil(n * (greedy_percentage / 100))
                            model.design_pssm_semigreedy(
                                soft_iters=0,
                                hard_iters=n_greedy_iters,
                                tries=greedy_tries,
                                models=[0],
                                num_models=1,
                                sample_models=False,
                                ramp_models=False,
                                save_best=True,
                            )

                        if enable_soft or enable_greedy:
                            save_pdb_filename = temp_pdb_path.replace(".pdb", "_refolded.pdb")
                            model.save_pdb(save_pdb_filename)
                        else:
                            save_pdb_filename = temp_pdb_path

                        stage_4_sample = load_pdb(save_pdb_filename)
                        refined_coors_local = torch.as_tensor(stage_4_sample.atom_positions, dtype=torch.float32).to(
                            coors.device
                        )
                        refined_residue_type_local = torch.as_tensor(stage_4_sample.aatype, dtype=torch.long).to(
                            coors.device
                        )
                        return refined_coors_local, refined_residue_type_local, callbacks_flag

                    try:
                        refined_coors, refined_residue_type, callbacks_registered = _run_with_model(
                            af_model,
                            callbacks_registered,
                            resolved_jax_backend,
                        )
                    except Exception as stage_exc:
                        should_retry_on_cpu = (
                            self._is_apple_backend_tag(resolved_jax_backend)
                            and self._is_apple_unsupported_primitive_error(stage_exc)
                        )
                        if not should_retry_on_cpu:
                            raise

                        logger.warning(
                            "Apple JAX refinement hit unsupported primitive; "
                            f"retrying sample {i + 1}/{bs} on CPU. Original error: {stage_exc}"
                        )

                        if cpu_fallback_model is None:
                            cpu_device, _ = self._select_jax_device("cpu")
                            cpu_fallback_model = self._make_afdesign_model(
                                num_recycles=n_recycles,
                                device=cpu_device,
                            )

                        refined_coors, refined_residue_type, cpu_callbacks_registered = _run_with_model(
                            cpu_fallback_model,
                            cpu_callbacks_registered,
                            "cpu-fallback",
                        )

                    # BUG FIX: validate that ColabDesign returned the
                    # expected number of residues before writing into
                    # the padded tensor.
                    if refined_coors.shape[0] != n:
                        raise ValueError(f"Refined PDB has {refined_coors.shape[0]} residues but mask expects {n}")

                    # Pad refined data back into full-length tensors
                    refined_coors_full = coors.clone()
                    refined_residue_type_full = residue_type.clone()
                    refined_coors_full[res_mask] = refined_coors
                    refined_residue_type_full[res_mask] = refined_residue_type

                    refined_sample = {
                        "coors": refined_coors_full.unsqueeze(0),
                        "residue_type": refined_residue_type_full.unsqueeze(0),
                    }
                    if chain_index is not None:
                        # Keep original chain_index; ColabDesign uses only chain A/B internally.
                        refined_sample["chain_index"] = chain_index.clone().unsqueeze(0)
                    refined_sample_prots.append(refined_sample)
                    elapsed = time.time() - t0
                    logger.info(f"Refined sample {i + 1}/{bs} in {elapsed:.1f}s")

                except Exception as exc:
                    elapsed = time.time() - t0
                    logger.exception(
                        f"Refinement failed for sample {i + 1}/{bs} after {elapsed:.1f}s, keeping original: {exc!r}"
                    )
                    fallback_sample = {
                        "coors": coors.clone().unsqueeze(0),
                        "residue_type": residue_type.clone().unsqueeze(0),
                    }
                    if chain_index is not None:
                        fallback_sample["chain_index"] = chain_index.clone().unsqueeze(0)
                    refined_sample_prots.append(fallback_sample)

            refined = concat_dict_tensors(refined_sample_prots, dim=0)

            # IMPROVEMENT: preserve all keys from the input that
            # refinement does not overwrite (e.g. mask, sample_type,
            # metadata_tag).  The original only returned the three keys
            # built inside the loop.
            result: dict[str, Any] = {}
            for key in sample_prots:
                result[key] = refined[key] if key in refined else sample_prots[key]
            return result

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

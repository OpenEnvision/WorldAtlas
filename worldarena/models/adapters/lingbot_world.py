from __future__ import annotations

from pathlib import Path
from typing import Any

from worldarena.models.adapters.base import ModelAdapter
from worldarena.models.adapters.common import build_env, extend_command_with_options, run_command
from worldarena.benchmark.schemas import BenchmarkSample


class LingBotWorldAdapter(ModelAdapter):
    def generate(
        self,
        sample: BenchmarkSample,
        conditioning_image: Path,
        output_path: Path,
        prompt: str,
    ) -> dict[str, Any]:
        generation = self.config.generation
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.config.python_bin,
            str((self.config.repo_root / self.config.entrypoint).resolve()),
            "--task",
            str(generation.get("task", "i2v-A14B")),
            "--ckpt_dir",
            str(self.config.checkpoint_dir or ""),
            "--save_file",
            str(output_path),
            "--prompt",
            prompt,
            "--image",
            str(conditioning_image),
            "--base_seed",
            str(int(generation.get("base_seed", 42))),
            "--size",
            str(generation.get("size", "832*480")),
            "--frame_num",
            str(int(generation.get("frame_num", 81))),
            "--sample_solver",
            str(generation.get("sample_solver", "unipc")),
        ]
        extend_command_with_options(
            command,
            payload=generation,
            value_options={
                "sample_steps": int,
                "sample_shift": float,
                "sample_guide_scale": float,
                "ulysses_size": int,
                "prompt_extend_method": str,
                "prompt_extend_model": str,
                "prompt_extend_target_lang": str,
            },
            bool_value_options=("offload_model",),
            flag_options=(
                "convert_model_dtype",
                "t5_fsdp",
                "t5_cpu",
                "dit_fsdp",
                "use_prompt_extend",
            ),
        )

        action_path = generation.get("action_path")
        action_string_by_suite = generation.get("action_string_by_suite", {})
        action_string = action_string_by_suite.get(sample.suite) or generation.get("action_string")
        if action_path:
            resolved_action_path = Path(action_path).expanduser()
            if not resolved_action_path.is_absolute():
                resolved_action_path = (self.config.repo_root / resolved_action_path).resolve()
            command.extend(["--action_path", str(resolved_action_path)])
        if action_string:
            command.extend(["--action_string", str(action_string), "--allow_act2cam"])

        env = build_env(
            extra_pythonpaths=[self.config.repo_root],
            overrides=self.config.env,
        )
        run_command(command, cwd=self.config.repo_root, env=env)
        if not output_path.exists():
            raise FileNotFoundError(f"LingBot output was not written: {output_path}")
        return {
            "command": command,
            "prediction_path": str(output_path),
            "prompt": prompt,
        }

# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""用固定纯 neutral 参考音频推理情感控制评测集。

这个脚本刻意不按目标情绪选参考音频。它会在启动时从 BB03 清单中筛出一条
“只有【neutral】标签”的参考，并让所有样本复用同一条参考音频。因此，输出的
情绪变化只能来自输入文本中的标签和文本内容，而不是同情绪 ref_audio。

默认启用 x_vector_only_mode，避免 ICL 参考文本与参考语音的韵律影响输出。

示例：
    python infer_neutral_ref_eval.py \
        --model /path/to/controllable_ckpt \
        --input eval_neutral_ref_50.jsonl \
        --out_dir neutral_ref_out_50 \
        --start 0 --end 50

先只检查数据和参考音频选择：
    python infer_neutral_ref_eval.py --validate_only \
        --input eval_neutral_ref_50.jsonl \
        --ref_jsonl /path/to/BB03_51h_cleaned.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


HERE = Path(__file__).resolve().parent
DEFAULT_BB03_ROOT = Path("/home/ma-user/work/dataset/csh_bj/BB03")
DEFAULT_REF_JSONL = DEFAULT_BB03_ROOT / "BB03_51h_cleaned.jsonl"
DEFAULT_NEUTRAL_REF_KEY = "0822_情感陪伴-目标音色-倾听反馈-002_000004"

# 仅接受模型实际可识别的情绪标签；强度档位写在标签末尾。
EMOTION_TAG_RE = re.compile(r"【([A-Za-z][A-Za-z ]*?)([123]?)】")
ALL_BRACKET_TAG_RE = re.compile(r"【([^】]+)】")
SQUARE_EVENT_RE = re.compile(r"\[([^\]]+)\]")
XML_EVENT_RE = re.compile(r"<[^>]+>")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"期望布尔值，收到: {value!r}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSON: {exc}") from exc
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{line_no} 顶层必须是对象")
            records.append(rec)
    return records


def parse_emotion_tags(text: str) -> list[str]:
    return [f"{name.lower()}{level}" for name, level in EMOTION_TAG_RE.findall(text or "")]


def resolve_audio_path(audio_path: str, audio_root: Path) -> Path:
    path = Path(audio_path)
    return path if path.is_absolute() else audio_root / path


def is_strict_neutral_reference(rec: dict[str, Any], allow_breath: bool) -> bool:
    """只保留一个 neutral 标签，排除其它情绪、停顿、重读与副语言。

    BB03 的绝大多数单句用 [breath] 显式标记自然换气。它不表示情绪，因此默认
    允许；通过 --allow_breath false 可以进一步收紧筛选。
    """
    text = str(rec.get("text", ""))
    all_tags = ALL_BRACKET_TAG_RE.findall(text)
    if all_tags != ["neutral"]:
        return False

    if XML_EVENT_RE.search(text):
        return False

    allowed_square_events = {"breath"} if allow_breath else set()
    return all(event.lower() in allowed_square_events for event in SQUARE_EVENT_RE.findall(text))


def find_neutral_references(
    ref_jsonl: Path,
    audio_root: Path,
    allow_breath: bool,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for rec in load_jsonl(ref_jsonl):
        if not is_strict_neutral_reference(rec, allow_breath):
            continue
        audio_rel = str(rec.get("audio_path", "")).strip()
        if not audio_rel:
            continue
        candidates.append(
            {
                "audio_path": str(resolve_audio_path(audio_rel, audio_root)),
                "audio_rel": audio_rel,
                "ref_text": str(rec.get("text", "")),
                "ref_key": str(rec.get("key", "")),
            }
        )

    candidates.sort(key=lambda item: (item["ref_key"], item["audio_rel"]))
    return candidates


def choose_reference(
    args: argparse.Namespace,
    candidates: list[dict[str, str]],
) -> dict[str, str]:
    """选择一次固定参考，之后所有任务都使用它。"""
    if args.neutral_ref_audio:
        audio_path = str(resolve_audio_path(args.neutral_ref_audio, Path(args.audio_root)))
        matched = next(
            (
                candidate
                for candidate in candidates
                if Path(candidate["audio_path"]) == Path(audio_path)
                or candidate["audio_rel"] == args.neutral_ref_audio
            ),
            None,
        )
        if matched is not None:
            return matched
        if not args.x_vector_only and not args.neutral_ref_text:
            raise ValueError(
                "关闭 x_vector_only 时，手动指定 --neutral_ref_audio 必须同时提供 "
                "--neutral_ref_text，确保参考文本与音频一致。"
            )
        return {
            "audio_path": audio_path,
            "audio_rel": args.neutral_ref_audio,
            "ref_text": args.neutral_ref_text,
            "ref_key": "manual",
        }

    if not candidates:
        breathing_note = "允许 [breath]" if args.allow_breath else "不允许 [breath]"
        raise RuntimeError(
            f"在 {args.ref_jsonl} 中没有找到纯 neutral 参考（{breathing_note}）。"
        )

    if args.neutral_ref_key:
        matches = [item for item in candidates if item["ref_key"] == args.neutral_ref_key]
        if not matches:
            raise RuntimeError(
                f"找不到 --neutral_ref_key={args.neutral_ref_key!r} 对应的纯 neutral 参考。"
            )
        return matches[0]

    if not 0 <= args.neutral_ref_index < len(candidates):
        raise ValueError(
            f"--neutral_ref_index 必须在 [0, {len(candidates) - 1}]，"
            f"当前为 {args.neutral_ref_index}。"
        )
    return candidates[args.neutral_ref_index]


def validate_tasks(tasks: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    seen_ids: set[str] = set()

    for row_no, task in enumerate(tasks, start=1):
        task_id = str(task.get("ID", "")).strip()
        text = str(task.get("输入文本", ""))
        expected_track = str(task.get("情感轨迹", "")).strip()
        tags = parse_emotion_tags(text)

        if not task_id:
            errors.append(f"第 {row_no} 条缺少 ID")
        elif task_id in seen_ids:
            errors.append(f"ID 重复: {task_id}")
        seen_ids.add(task_id)

        if not text:
            errors.append(f"{task_id or f'第 {row_no} 条'} 缺少 输入文本")
        if not tags:
            errors.append(f"{task_id or f'第 {row_no} 条'} 没有有效情绪标签")
        if not expected_track:
            errors.append(f"{task_id or f'第 {row_no} 条'} 缺少 情感轨迹")
        elif [part.strip() for part in expected_track.split("->")] != tags:
            errors.append(
                f"{task_id} 的 情感轨迹={expected_track!r} 与输入标签={tags!r} 不一致"
            )

        # 这套评测必须由脚本统一指定参考，不能在单条数据里偷换参考音频。
        if task.get("ref_audio"):
            errors.append(f"{task_id} 不应在 JSONL 中设置 ref_audio")
        if task.get("参考策略") != "固定纯neutral参考":
            errors.append(f"{task_id} 的 参考策略 必须是 固定纯neutral参考")

    if errors:
        raise ValueError("评测集校验失败：\n- " + "\n- ".join(errors))


def slice_tasks(
    tasks: list[dict[str, Any]],
    start: int,
    end: int,
    limit: int,
) -> list[dict[str, Any]]:
    if start < 0 or end < 0 or limit < 0:
        raise ValueError("--start、--end、--limit 不能为负数")
    end_index = end if end > 0 else None
    selected = tasks[start:end_index]
    return selected[:limit] if limit > 0 else selected


def make_manifest_record(
    task: dict[str, Any],
    ref: dict[str, str],
    wav_path: Path,
    status: str,
    x_vector_only: bool,
    error: str = "",
) -> dict[str, Any]:
    return {
        "ID": task["ID"],
        "status": status,
        "error": error,
        "wav_path": str(wav_path),
        "评测维度": task.get("评测维度", ""),
        "控制项": task.get("控制项", ""),
        "控制值": task.get("控制值", ""),
        "强度": task.get("强度", ""),
        "情感轨迹": task.get("情感轨迹", ""),
        "期望标记": task.get("期望标记", ""),
        "输入文本": task["输入文本"],
        "ref_audio": ref["audio_path"],
        "ref_audio_rel": ref["audio_rel"],
        "ref_text": ref["ref_text"],
        "ref_key": ref["ref_key"],
        "ref_policy": "single_strict_neutral_reference_for_all_items",
        "x_vector_only": x_vector_only,
    }


def generate_one(
    tts: Qwen3TTSModel,
    text: str,
    ref: dict[str, str],
    args: argparse.Namespace,
) -> tuple[Any, int]:
    wavs, sample_rate = tts.generate_voice_clone(
        text=text,
        language=args.language,
        ref_audio=ref["audio_path"],
        ref_text=ref["ref_text"],
        x_vector_only_mode=args.x_vector_only,
        non_streaming_mode=True,
    )
    if wavs is None or (isinstance(wavs, (list, tuple)) and not wavs):
        raise RuntimeError("模型没有返回音频")
    wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    if hasattr(wav, "detach"):
        wav = wav.detach().float().cpu().numpy()
    return wav, sample_rate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="固定纯 neutral ref_audio 的情感标签控制推理"
    )
    parser.add_argument(
        "--model",
        default="/home/ma-user/work/dataset/csh_bj/lite/Qwen3-TTS/script/ckpt/pretrain_v14/checkpoint-step-3800",
        help="可控 TTS 模型/权重路径",
    )
    parser.add_argument("--input", default=str(HERE / "eval_neutral_ref_50.jsonl"))
    parser.add_argument("--out_dir", default=str(HERE / "neutral_ref_out_50"))
    parser.add_argument("--ref_jsonl", default=str(DEFAULT_REF_JSONL))
    parser.add_argument("--audio_root", default=str(DEFAULT_BB03_ROOT))
    parser.add_argument(
        "--neutral_ref_key",
        default=DEFAULT_NEUTRAL_REF_KEY,
        help="固定使用指定 BB03 key；默认是已校验的纯 neutral 参考",
    )
    parser.add_argument(
        "--neutral_ref_index",
        type=int,
        default=0,
        help="未指定 key 时，按排序后的纯 neutral 候选取第几个（从 0 开始）",
    )
    parser.add_argument(
        "--neutral_ref_audio",
        default="",
        help="手动固定参考音频。提供后不再从 --ref_jsonl 选取。",
    )
    parser.add_argument(
        "--neutral_ref_text",
        default="",
        help="手动参考音频对应文本；仅在它不在 ref_jsonl 中时需要。",
    )
    parser.add_argument(
        "--allow_breath",
        type=parse_bool,
        default=True,
        help="是否允许参考文本含 [breath]（默认 true）",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="Auto")
    parser.add_argument(
        "--x_vector_only",
        type=parse_bool,
        default=True,
        help="默认 true，只使用参考音色 embedding，不进行 ICL",
    )
    parser.add_argument("--start", type=int, default=0, help="起始行下标（从 0 开始）")
    parser.add_argument("--end", type=int, default=0, help="结束行下标（0 表示直到末尾）")
    parser.add_argument("--limit", type=int, default=0, help="最多推理条数（0 表示不限制）")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的同 ID WAV；默认保留并在 manifest 标记 skipped_existing",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="仅校验 JSONL 并显示固定参考，不加载模型",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    ref_jsonl = Path(args.ref_jsonl)
    audio_root = Path(args.audio_root)
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wavs"

    tasks = load_jsonl(input_path)
    validate_tasks(tasks)
    tasks = slice_tasks(tasks, args.start, args.end, args.limit)
    if not tasks:
        raise RuntimeError("切片后没有待处理样本，请检查 --start / --end / --limit。")

    candidates = find_neutral_references(ref_jsonl, audio_root, args.allow_breath)
    ref = choose_reference(args, candidates)
    ref_path = Path(ref["audio_path"])
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"参考音频不存在: {ref_path}\n"
            "请检查 --audio_root、--ref_jsonl，或用 --neutral_ref_audio 显式指定。"
        )

    print(f"评测集: {input_path}，总条数={len(tasks)}")
    print(f"纯 neutral 候选数: {len(candidates)}")
    print(f"固定参考 key: {ref['ref_key'] or '(无 key)'}")
    print(f"固定参考音频: {ref['audio_path']}")
    print(f"固定参考文本: {ref['ref_text']}")
    print(f"x_vector_only: {args.x_vector_only}")
    print(f"维度分布: {dict(Counter(task.get('评测维度', '') for task in tasks))}")

    if args.validate_only:
        print("校验通过；未加载模型。")
        return 0

    wav_dir.mkdir(parents=True, exist_ok=True)
    print(f"加载模型: {args.model}")
    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    manifest_path = out_dir / "manifest.jsonl"
    n_ok = 0
    n_error = 0
    n_skipped = 0
    start_time = time.time()
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        for index, task in enumerate(tasks, start=1):
            wav_path = wav_dir / f"{task['ID']}.wav"
            if wav_path.exists() and not args.overwrite:
                rec = make_manifest_record(
                    task, ref, wav_path, "skipped_existing", args.x_vector_only
                )
                n_skipped += 1
            else:
                try:
                    wav, sample_rate = generate_one(tts, task["输入文本"], ref, args)
                    sf.write(str(wav_path), wav, sample_rate)
                    rec = make_manifest_record(
                        task, ref, wav_path, "ok", args.x_vector_only
                    )
                    n_ok += 1
                except Exception as exc:  # noqa: BLE001
                    rec = make_manifest_record(
                        task, ref, wav_path, "error", args.x_vector_only, repr(exc)
                    )
                    n_error += 1
                    print(f"[{task['ID']}] 生成失败: {exc}", file=sys.stderr)

            manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
            manifest.flush()
            print(
                f"进度 {index}/{len(tasks)}  ok={n_ok} "
                f"skipped={n_skipped} err={n_error}"
            )

    elapsed = time.time() - start_time
    print(f"完成：ok={n_ok} skipped={n_skipped} err={n_error}，用时 {elapsed:.1f}s")
    print(f"WAV: {wav_dir}")
    print(f"manifest: {manifest_path}")
    return 1 if n_error else 0


if __name__ == "__main__":
    raise SystemExit(main())

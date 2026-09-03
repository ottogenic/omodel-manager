#!/usr/bin/env python3
"""Download and verify the pinned Qwen3.8 GPTQ model used by the B70 vLLM gate."""

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


REPOSITORY = "SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
REVISION = "9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e"
DEFAULT_DESTINATION = (
    "~/.cache/otools/models/sergiiob-qwen38-vllm/" + REVISION
)
FILES = {
    "chat_template.jinja": (
        8_952,
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    ),
    "config.json": (
        5_085,
        "580f5caae29242b242d3bcab2ad5b8a942fc0a59df1d228bd2b3233cfc760372",
    ),
    "generation_config.json": (
        214,
        "d0d0ed2e37cdfafef4a5067d5ea2407b05f4fb50526e47c008a5b235d50240fb",
    ),
    "model-00001-of-00005.safetensors": (
        3_516_828_472,
        "30c8e2b1c82cdcc840848b5c98bafe2f74269b1e6472a053ce7d7b2d002f39a7",
    ),
    "model-00002-of-00005.safetensors": (
        4_278_206_560,
        "ff66eaf6ecc6e4b214f281ac532dcfcb07c60d5a8c78cf145bc93c38c00c024e",
    ),
    "model-00003-of-00005.safetensors": (
        4_258_595_144,
        "15284cb88d52ea1648b4fcc68901286d7c4795388e05ae1e143c8026fcb0be44",
    ),
    "model-00004-of-00005.safetensors": (
        4_284_981_464,
        "878ae6ebc9553de5340df0d6097aa319f58650382c89e474b39d0c0a98e76932",
    ),
    "model-00005-of-00005.safetensors": (
        3_220_838_576,
        "2a6ebd04c77c2d5ce5952ca81a4197f1c34f712a00c293a25242ec27ac413729",
    ),
    "model.safetensors.index.json": (
        229_820,
        "d511a969b32a16f1890326dadd4d04039779cebce7458b8e5d2ce7f3ce2550ed",
    ),
    "processor_config.json": (
        1_191,
        "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    ),
    "quantize_config.json": (
        1_169,
        "ecbdecd64057569d9ab8e74c68eb329776e682d44746ff7b82bba6fcb0e157db",
    ),
    "tokenizer.json": (
        19_989_325,
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    ),
    "tokenizer_config.json": (
        1_151,
        "c5528e605ca971b6afeb2cb0025fd21244650e6fc7739808c581c6d5cdb70aa3",
    ),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path, expected_size, expected_sha256):
    if not path.is_file():
        return False, "missing"
    size = path.stat().st_size
    if size != expected_size:
        return False, f"size {size:,}, expected {expected_size:,}"
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        return False, f"SHA-256 {actual_sha256}, expected {expected_sha256}"
    return True, "verified"


def safetensors_header(path):
    with path.open("rb") as handle:
        encoded_length = handle.read(8)
        if len(encoded_length) != 8:
            raise RuntimeError(f"invalid Safetensors length in {path}")
        header_length = struct.unpack("<Q", encoded_length)[0]
        if header_length > 128 * 1024 * 1024:
            raise RuntimeError(f"unreasonable Safetensors header in {path}")
        encoded_header = handle.read(header_length)
    try:
        header = json.loads(encoded_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid Safetensors JSON in {path}") from exc

    data_size = path.stat().st_size - 8 - header_length
    ranges = []
    tensors = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            raise RuntimeError(f"invalid metadata for {name} in {path}")
        offsets = metadata.get("data_offsets")
        shape = metadata.get("shape")
        dtype = metadata.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > data_size
            or not isinstance(shape, list)
            or not all(isinstance(value, int) and value >= 0 for value in shape)
            or not isinstance(dtype, str)
        ):
            raise RuntimeError(f"invalid tensor descriptor for {name} in {path}")
        ranges.append((offsets[0], offsets[1], name))
        tensors[name] = metadata

    previous_end = 0
    for start, end, name in sorted(ranges):
        if start != previous_end:
            raise RuntimeError(f"non-contiguous tensor data before {name} in {path}")
        previous_end = end
    if previous_end != data_size:
        raise RuntimeError(f"unreferenced tensor data in {path}")
    return tensors


def verify_model_layout(destination):
    index_path = destination / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid model index: {index_path}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("model index has no weight_map")

    tensors = {}
    for filename in sorted(set(weight_map.values())):
        if filename not in FILES or not filename.endswith(".safetensors"):
            raise RuntimeError(f"unexpected shard in model index: {filename}")
        for name, metadata in safetensors_header(destination / filename).items():
            if name in tensors:
                raise RuntimeError(f"duplicate tensor: {name}")
            tensors[name] = (filename, metadata)

    if set(tensors) != set(weight_map):
        missing = set(weight_map) - set(tensors)
        extra = set(tensors) - set(weight_map)
        raise RuntimeError(
            f"model index mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    wrong_shards = [
        name for name, filename in weight_map.items() if tensors[name][0] != filename
    ]
    if wrong_shards:
        raise RuntimeError(f"{len(wrong_shards)} tensors are in the wrong shard")

    mtp = {
        name: metadata["dtype"]
        for name, (_, metadata) in tensors.items()
        if ".mtp." in name or name.startswith("mtp.")
    }
    if not mtp or set(mtp.values()) - {"BF16"}:
        raise RuntimeError("MTP tensors are missing or are not all BF16")
    print(
        f"model layout verified: {len(tensors):,} tensors, "
        f"{len(mtp)} BF16 MTP tensors",
        flush=True,
    )


def download_file(destination, filename, expected_size, expected_sha256):
    target = destination / filename
    valid, detail = verify_file(target, expected_size, expected_sha256)
    if valid:
        print(f"verified {filename}", flush=True)
        return
    if target.exists():
        raise RuntimeError(f"refusing to replace invalid {target}: {detail}")

    partial = destination / f"{filename}.part"
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise RuntimeError(f"partial file is too large: {partial}")
    url = (
        f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/"
        f"{urllib.parse.quote(filename)}"
    )
    headers = {"User-Agent": "omodel-manager-card-model-downloader/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)

    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset == expected_size:
            response = None
        else:
            raise

    if response is not None:
        append = offset > 0 and response.status == 206
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        print(
            f"downloading {filename} from {offset:,}/{expected_size:,} bytes",
            flush=True,
        )
        with response, partial.open(mode) as handle:
            received = offset
            next_report = received + 512 * 1024 * 1024
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                if received >= next_report:
                    print(f"  {received:,}/{expected_size:,} bytes", flush=True)
                    next_report += 512 * 1024 * 1024

    valid, detail = verify_file(partial, expected_size, expected_sha256)
    if not valid:
        raise RuntimeError(f"download verification failed for {filename}: {detail}")
    os.replace(partial, target)
    print(f"verified {filename}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default=DEFAULT_DESTINATION)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    destination = Path(args.destination).expanduser()
    if not args.verify_only:
        destination.mkdir(parents=True, exist_ok=True)

    failures = []
    for filename, (expected_size, expected_sha256) in FILES.items():
        if args.verify_only:
            valid, detail = verify_file(
                destination / filename, expected_size, expected_sha256
            )
            print(f"{filename}: {detail}")
            if not valid:
                failures.append(filename)
        else:
            download_file(
                destination, filename, expected_size, expected_sha256
            )
    if failures:
        print(f"verification failed for {len(failures)} file(s)", file=sys.stderr)
        return 1
    verify_model_layout(destination)
    print(f"model verified at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

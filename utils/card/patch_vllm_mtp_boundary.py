"""Apply the pinned Qwen3.8 partial-MTP-group boundary fix."""

import hashlib
from pathlib import Path


TARGET = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/gdn_attn.py"
)
EXPECTED_SHA256 = "fda86b96ab5daaf50bd02d022518779c220401dbedc7b28cf478f4c48e72d3d3"
PATCHED_SHA256 = "135799921da0d842aae828a23bdbce010ca08ab76848ea08b1e1c1736caf401a"
MARKER = "B70_MTP_PARTIAL_FINAL_GROUP"

ORIGINAL = '''            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
                non_spec_state_indices_tensor = None
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
'''

PATCHED = '''            if num_prefills == 0 and num_decodes == 0:
                expected_spec_token_size = num_spec_decodes * (self.num_spec + 1)
                actual_spec_token_size = query_start_loc_cpu[-1].item()
                if actual_spec_token_size < expected_spec_token_size:
                    # B70_MTP_PARTIAL_FINAL_GROUP: The max-sequence boundary can
                    # truncate the final speculative group. The XPU GDN kernel
                    # requires complete groups, so process this final partial
                    # group through the existing stateful non-spec prefill path.
                    spec_sequence_masks = None
                    spec_sequence_masks_cpu = None
                    num_prefills = num_spec_decodes
                    num_prefill_tokens = actual_spec_token_size
                    num_spec_decodes = 0
                    num_spec_decode_tokens = 0
                    spec_token_indx = None
                    non_spec_token_indx = None
                    spec_state_indices_tensor = None
                    non_spec_state_indices_tensor = block_table_tensor[:, 0]
                    spec_query_start_loc = None
                    non_spec_query_start_loc = query_start_loc
                    non_spec_query_start_loc_cpu = query_start_loc_cpu
                    num_accepted_tokens = None
                else:
                    spec_token_indx = torch.arange(
                        expected_spec_token_size,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                    non_spec_token_indx = torch.empty(
                        0, dtype=torch.int32, device=query_start_loc.device
                    )
                    # Filter by spec_sequence_masks to exclude padded sequences
                    spec_state_indices_tensor = block_table_tensor[
                        spec_sequence_masks_cpu, : self.num_spec + 1
                    ]
                    non_spec_state_indices_tensor = None
                    # Padded sequences are always at the back, so the first
                    # num_spec_decodes + 1 entries of query_start_loc already
                    # contain the correct cumulative token counts.
                    spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                    non_spec_query_start_loc = None
                    non_spec_query_start_loc_cpu = None
            else:
'''

ORIGINAL_FINALIZE = '''            assert num_accepted_tokens is not None
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
'''

PATCHED_FINALIZE = '''            if spec_sequence_masks_cpu is not None:
                assert num_accepted_tokens is not None
                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]
'''


def patch_source(source):
    if MARKER in source:
        return source
    if source.count(ORIGINAL) != 1:
        raise RuntimeError("expected exactly one pinned GDN pure-spec block")
    if source.count(ORIGINAL_FINALIZE) != 1:
        raise RuntimeError("expected exactly one pinned GDN finalize block")
    source = source.replace(ORIGINAL, PATCHED, 1)
    return source.replace(ORIGINAL_FINALIZE, PATCHED_FINALIZE, 1)


def apply_patch():
    source = TARGET.read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest == PATCHED_SHA256:
        compile(source, str(TARGET), "exec")
        print(f"MTP boundary already guarded; sha256={digest}", flush=True)
        return
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"refusing to patch {TARGET}: expected {EXPECTED_SHA256} or "
            f"{PATCHED_SHA256}, got {digest}"
        )

    patched = patch_source(source)
    compile(patched, str(TARGET), "exec")
    patched_digest = hashlib.sha256(patched.encode()).hexdigest()
    if patched_digest != PATCHED_SHA256:
        raise RuntimeError(
            f"patched {TARGET} has unexpected SHA-256 {patched_digest}"
        )
    TARGET.write_text(patched, encoding="utf-8")
    print(f"guarded exact MTP boundary; patched_sha256={patched_digest}", flush=True)

import os
import csv
import json
import time
import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from datetime import datetime, timezone

from langchain_community.vectorstores import Chroma
from modules.document_input import DocumentInputService


# -------------------------
# Small utils
# -------------------------

def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- Token estimation and prompt/completion synth utils ---


def _approx_tokens_from_text(s: str, chars_per_token: float = 4.0) -> int:
	"""
	Approximate token count from character length.
	Uses a conservative heuristic: tokens ~= ceil(chars / 4).
	"""
	s = s or ""
	if not s:
		return 0
	return int(math.ceil(len(s) / float(chars_per_token)))


def _chunked(items: List[str], size: int) -> List[List[str]]:
	out: List[List[str]] = []
	for i in range(0, len(items), size):
		out.append(items[i : i + size])
	return out


def _build_step4_prompt_text(results_chunks: List[Dict[str, Any]]) -> str:
	"""
	Closer mirror of CorrelationExtractionService:
	- System-like instructions + user-provided Results sentences
	- Output: STRICT JSON ARRAY of objects with id/source/target/evidence
	"""
	system = (
		"You are helping extract variable–variable relationships from a biomedical research article.\n\n"
		"Task:\n"
		"- Identify pairs of variables explicitly reported as correlated/associated/related.\n"
		"- ONLY use evidence from the Results section sentences provided.\n"
		"- Do NOT infer causality or direction.\n\n"
		"Output format:\n"
		"- Return STRICT JSON only.\n"
		"- Return a JSON ARRAY of objects.\n"
		"- Each object: {\"id\":\"...\",\"source\":\"...\",\"target\":\"...\","
		"\"evidence\":[{\"chu_idx\":123,\"text\":\"...\"}, ...]}\n"
		"- If an edge has no explicit evidence, set evidence: [].\n"
	)
	user = (
		"Here are Results sentences as a JSON list of objects (each has chu_idx and text). "
		"Extract relationship edges.\n\n"
		f"{json.dumps(results_chunks, ensure_ascii=False)}"
	)
	return system + "\n\n" + user


def _build_step4_synth_completion(
	edges: List[Dict[str, Any]],
	edge_to_idxs: Dict[str, List[int]],
	idx_to_text: Dict[int, str],
) -> str:
	"""
	Synthesize a completion shaped like CorrelationExtractionService output:
	JSON ARRAY of {id, source, target, evidence:[{chu_idx, text}, ...]}.
	"""
	out: List[Dict[str, Any]] = []
	for e in edges:
		edge_id = (e.get("edge_id") or "").strip()
		source = (e.get("source") or "")
		target = (e.get("target") or "")
		idxs = edge_to_idxs.get(edge_id, []) or []
		evidence: List[Dict[str, Any]] = []
		for idx in idxs:
			try:
				idx_i = int(idx)
			except Exception:
				continue
			evidence.append(
				{
					"chu_idx": idx_i,
					"text": (idx_to_text.get(idx_i) or ""),
				}
			)
		out.append(
			{
				"id": edge_id,
				"source": source,
				"target": target,
				"evidence": evidence,
			}
		)
	return json.dumps(out, ensure_ascii=False)



def _build_step5_prompt_text(raw_names: List[str]) -> str:
	return (
		"You are reconciling variable names extracted from biomedical research papers.\n"
		"Task: map each raw variable string to ONE canonical variable name.\n"
		"Rules:\n"
		"- Keep canonical names concise but specific.\n"
		"- Preserve clinically meaningful qualifiers (e.g., 'serum', 'maternal', 'baseline', 'fasting') "
		"ONLY if they are part of the variable itself rather than a study design detail.\n"
		"- If two raw names are synonyms, map them to the same canonical name.\n"
		"- Do not invent new variables.\n"
		"- If a raw name is already good, keep it unchanged.\n"
		"- Output JSON only.\n\n"
		"Raw variable names:\n"
		f"{json.dumps(raw_names, ensure_ascii=False)}\n\n"
		"Return JSON with this exact shape:\n"
		"{\"mappings\":[{\"raw\":\"...\",\"canonical\":\"...\"}, ...]}\n"
	)



def _build_step5_synth_completion(raw_names: List[str], raw_to_can: Dict[str, str]) -> str:
	out: Dict[str, Any] = {"mappings": []}
	for raw in raw_names:
		raw_s = str(raw or "").strip()
		if not raw_s:
			continue
		out["mappings"].append(
			{
				"raw": raw_s,
				"canonical": str(raw_to_can.get(raw_s, raw_s) or raw_s).strip(),
			}
		)
	return json.dumps(out, ensure_ascii=False)


def _read_csv_set(path: Path, key_cols: List[str]) -> set:
	"""
	Read an existing CSV and return a set of tuple keys for resume.
	"""
	if not path.exists():
		return set()
	out = set()
	with path.open("r", newline="", encoding="utf-8") as f:
		r = csv.DictReader(f)
		for row in r:
			key = tuple((row.get(k) or "").strip() for k in key_cols)
			out.add(key)
	return out


def _append_csv_row(path: Path, row: Dict[str, Any], fieldnames: List[str]):
	path.parent.mkdir(parents=True, exist_ok=True)
	write_header = not path.exists()
	with path.open("a", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		if write_header:
			w.writeheader()
		w.writerow({k: row.get(k, "") for k in fieldnames})


def _safe_json_loads(x: Any, default):
	if x is None:
		return default
	if isinstance(x, (list, dict)):
		return x
	if not isinstance(x, str):
		return default
	s = x.strip()
	if not s:
		return default
	try:
		return json.loads(s)
	except Exception:
		return default


def _to_int_list(x: Any) -> List[int]:
	"""
	Parse chu_idx_list stored as JSON string in Chroma metadata.
	Return sorted unique ints, preserving order by numeric sort.
	"""
	arr = _safe_json_loads(x, default=[])
	out: List[int] = []
	if isinstance(arr, list):
		for it in arr:
			try:
				out.append(int(it))
			except Exception:
				continue
	# unique + sort (paper/human wants stable order)
	out = sorted(set(out))
	return out


def _format_consecutive_groups(nums: List[int]) -> str:
	"""
	[1,2,3,7,9,10] -> "1-3; 7; 9-10"
	"""
	if not nums:
		return ""
	parts: List[str] = []
	start = nums[0]
	prev = nums[0]
	for n in nums[1:]:
		if n == prev + 1:
			prev = n
			continue
		# close group
		if start == prev:
			parts.append(str(start))
		else:
			parts.append(f"{start}-{prev}")
		start = n
		prev = n
	# last group
	if start == prev:
		parts.append(str(start))
	else:
		parts.append(f"{start}-{prev}")
	return "; ".join(parts)


def _build_evidence_block(chu_idx_list: List[int], idx_to_text: Dict[int, str]) -> Tuple[str, int]:
	"""
	Return a single merged evidence string with chu_idx headers, plus n_found.
	"""
	blocks: List[str] = []
	found = 0
	for idx in chu_idx_list:
		txt = (idx_to_text.get(idx) or "").strip()
		if txt:
			found += 1
		blocks.append(f"[chu_idx={idx}]\n{txt}")
	return "\n\n".join(blocks), found


# -------------------------
# Chroma readers (READ-ONLY)
# -------------------------

def _get_collection(chroma_dir: Path, collection_name: str, embedder) -> Chroma:
	return Chroma(
		persist_directory=str(chroma_dir),
		embedding_function=embedder,
		collection_name=collection_name,
	)


def _load_phase1_map(phase1_csv: Path) -> Dict[str, Dict[str, Any]]:
	"""
	phase1_ingest_log.csv columns (observed):
	- paper_id, doc_sha256, pdf_path, run_id, status, phase
	There can be multiple rows per doc_sha256 (different paper_id). We aggregate.
	"""
	if not phase1_csv.exists():
		return {}
	tmp: Dict[str, Dict[str, Any]] = {}
	with phase1_csv.open("r", newline="", encoding="utf-8") as f:
		r = csv.DictReader(f)
		for row in r:
			doc = (row.get("doc_sha256") or "").strip()
			if not doc:
				continue
			paper_id = (row.get("paper_id") or "").strip()
			pdf_path = (row.get("pdf_path") or "").strip()
			status = (row.get("status") or "").strip()
			phase = (row.get("phase") or "").strip()

			if doc not in tmp:
				tmp[doc] = {
					"paper_id_list": [],
					"pdf_path": pdf_path,
					"phase1_status": status,
					"phase1_phase": phase,
				}
			if paper_id and paper_id not in tmp[doc]["paper_id_list"]:
				tmp[doc]["paper_id_list"].append(paper_id)
			# keep a non-empty pdf_path if we ever see it
			if pdf_path and not tmp[doc].get("pdf_path"):
				tmp[doc]["pdf_path"] = pdf_path
	return tmp


# -------------------------
# Phase-2 doc log join helper
# -------------------------

def _load_phase2_doc_log_map(phase2_doc_log_csv: Optional[Path]) -> Dict[str, Dict[str, Any]]:
	"""
	Optional join helper.
	
	If provided and exists, reads per-doc Phase-2 log (e.g.,
	data_registry/phase2_doc_log__gpt-5.csv) to attach runtime/count metadata
	into the adjudication input CSV.
	
	This is READ-ONLY and safe for Phase 1/2.
	
	Expected columns (if present):
	- doc_sha256
	- results_sentences
	- edges_created_step
	- traces_created_step
	- variables_created_step
	- sec_per_doc
	- run_tag
	- timestamp_utc
	- prompt_tokens / completion_tokens / total_tokens (optional)
	"""
	if not phase2_doc_log_csv:
		return {}
	if not phase2_doc_log_csv.exists():
		return {}

	out: Dict[str, Dict[str, Any]] = {}
	with phase2_doc_log_csv.open("r", newline="", encoding="utf-8") as f:
		r = csv.DictReader(f)
		for row in r:
			doc = (row.get("doc_sha256") or "").strip()
			if not doc:
				continue
			# keep last-seen row for doc (should be unique anyway)
			out[doc] = {
				"results_sentences": (row.get("results_sentences") or "").strip(),
				"edges_created_step": (row.get("edges_created_step") or "").strip(),
				"traces_created_step": (row.get("traces_created_step") or "").strip(),
				"variables_created_step": (row.get("variables_created_step") or "").strip(),
				"sec_per_doc": (row.get("sec_per_doc") or "").strip(),
				"phase2_timestamp_utc": (row.get("timestamp_utc") or "").strip(),
				# token counts are optional (only populated if your Phase-2 logger provides them)
				"prompt_tokens": (row.get("prompt_tokens") or "").strip(),
				"completion_tokens": (row.get("completion_tokens") or "").strip(),
				"total_tokens": (row.get("total_tokens") or "").strip(),
			}
	return out


def export_adjudication_input(
	chroma_dir: Path,
	out_csv: Path,
	phase1_csv: Optional[Path] = None,
	doc_limit: Optional[int] = None,
	phase2_doc_log_csv: Optional[Path] = None,
) -> None:
	"""
	Export one row per extracted edge.
	Reads:
	- edges collection: edge metadata (doc_sha256, edge_id, source, target)
	- trace collection: parent_id=edge_id -> chu_idx_list
	- evidences collection: sentence text by chu_idx, filtered to Results
	Writes CSV for human + LLM judges to consume.
	"""
	doc_service = DocumentInputService(
		chroma_dir=chroma_dir,
	)

	phase1_map = _load_phase1_map(phase1_csv) if phase1_csv else {}
	phase2_doc_log_map = _load_phase2_doc_log_map(phase2_doc_log_csv)

	edges_db = _get_collection(chroma_dir, doc_service.edges_collection, doc_service.text_embedder)
	trace_db = _get_collection(chroma_dir, doc_service.trace_collection, doc_service.text_embedder)
	variables_db = _get_collection(
		chroma_dir,
		doc_service.variables_collection,
		doc_service.text_embedder,
	)

	# Pull all edges in this chroma snapshot
	got_edges = edges_db._collection.get(include=["metadatas"])
	edge_metas: List[Dict[str, Any]] = got_edges.get("metadatas") or []

	# Group edges by doc
	doc_to_edges: Dict[str, List[Dict[str, Any]]] = {}
	for meta in edge_metas:
		meta = meta or {}
		doc = (meta.get("doc_sha256") or "").strip()
		if not doc:
			continue
		doc_to_edges.setdefault(doc, []).append(meta)

	docs = sorted(doc_to_edges.keys())
	if doc_limit is not None:
		docs = docs[: max(0, int(doc_limit))]

	fieldnames = [
		"doc_sha256",
		"edge_id",
		"var1",
		"var2",
		# phase1 convenience
		"paper_id_list",
		"pdf_path",
		"phase1_status",
		# phase2 convenience (optional join)
		"results_sentences",
		"edges_created_step",
		"traces_created_step",
		"variables_created_step",
		"sec_per_doc",
		"phase2_timestamp_utc",
		"prompt_tokens",
		"completion_tokens",
		"total_tokens",
		# --- Post-hoc token estimation columns (inserted after total_tokens) ---
		"est_prompt_tokens_step4",
		"est_completion_tokens_step4",
		"est_total_tokens_step4",
		"est_prompt_tokens_step5",
		"est_completion_tokens_step5",
		"est_total_tokens_step5",
		"est_total_tokens_phase2",
		# evidence indices
		"chu_idx_list_json",
		"chu_idx_list_str",
		"chu_idx_groups",
		"n_chu_idx",
		"n_chunks_found",
		"has_missing_chunks",
		# evidence text
		"evidence_text",
		# adjudication column (empty on export)
		"label",
	]

	# overwrite out_csv deterministically (export step is cheap)
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=fieldnames)
		w.writeheader()

	for doc_sha256 in docs:
		# Build Results-only chu_idx -> text mapping
		rows = doc_service._read_child_rows_for_doc(doc_sha256)
		idx_to_text: Dict[int, str] = {}
		for _, chu_idx, lab, txt in rows:
			if lab != "Results":
				continue
			if not txt:
				continue
			idx_to_text[int(chu_idx)] = txt

		# Load all traces for the doc and map parent_id(edge_id) -> chu_idx_list
		got_tr = trace_db._collection.get(
			where={"doc_sha256": doc_sha256},
			include=["metadatas"],
		)
		trace_metas: List[Dict[str, Any]] = got_tr.get("metadatas") or []
		edge_to_idxs: Dict[str, List[int]] = {}
		for tm in trace_metas:
			tm = tm or {}
			parent_id = (tm.get("parent_id") or "").strip()
			if not parent_id:
				continue
			if (tm.get("evidence_type") or "") != "edge":
				continue
			idxs = _to_int_list(tm.get("chu_idx_list"))
			edge_to_idxs[parent_id] = idxs

		# --- Token estimation (post-hoc, approximate) ---
		# Step 4 prompt: Results chunks JSON
		results_chunks_for_prompt: List[Dict[str, Any]] = []
		for _, chu_idx, lab, txt in rows:
			if lab != "Results":
				continue
			if not txt:
				continue
			results_chunks_for_prompt.append({"chu_idx": int(chu_idx), "text": txt})
		results_chunks_for_prompt = sorted(results_chunks_for_prompt, key=lambda d: d.get("chu_idx", 0))

		step4_prompt_text = _build_step4_prompt_text(results_chunks_for_prompt)
		est_prompt_tokens_step4 = _approx_tokens_from_text(step4_prompt_text)

		# Step 4 completion: synthesize from stored edges + trace
		edges_for_doc = doc_to_edges.get(doc_sha256, [])
		step4_completion_text = _build_step4_synth_completion(
			edges_for_doc,
			edge_to_idxs,
			idx_to_text,
		)
		est_completion_tokens_step4 = _approx_tokens_from_text(step4_completion_text)
		est_total_tokens_step4 = est_prompt_tokens_step4 + est_completion_tokens_step4

		# Step 5 prompt/completion: synthesize from stored variables (raw -> canonical)
		raw_to_can: Dict[str, str] = {}
		try:
			got_vars = variables_db._collection.get(
				where={"doc_sha256": doc_sha256},
				include=["metadatas"],
			)
			var_metas: List[Dict[str, Any]] = got_vars.get("metadatas") or []
			for vm in var_metas:
				vm = vm or {}
				raw = (
					vm.get("raw_name")
					or vm.get("raw")
					or vm.get("source")
					or vm.get("variable_raw")
					or ""
				)
				can = (
					vm.get("name")
					or vm.get("canonical_name")
					or vm.get("target")
					or vm.get("variable")
					or ""
				)
				raw = str(raw or "").strip()
				can = str(can or "").strip()
				if raw:
					raw_to_can[raw] = can
		except Exception:
			raw_to_can = {}

		raw_names = sorted([k for k in raw_to_can.keys() if k])

		# VariableReconciliationService batches raw names (default batch_size=25)
		batch_size = 25
		est_prompt_tokens_step5 = 0
		est_completion_tokens_step5 = 0
		for batch in _chunked(raw_names, batch_size):
			step5_prompt_text = _build_step5_prompt_text(batch)
			est_prompt_tokens_step5 += _approx_tokens_from_text(step5_prompt_text)

			step5_completion_text = _build_step5_synth_completion(batch, raw_to_can)
			est_completion_tokens_step5 += _approx_tokens_from_text(step5_completion_text)

		est_total_tokens_step5 = est_prompt_tokens_step5 + est_completion_tokens_step5

		est_total_tokens_phase2 = est_total_tokens_step4 + est_total_tokens_step5

		p1 = phase1_map.get(doc_sha256, {})
		paper_id_list = p1.get("paper_id_list") or []
		pdf_path = p1.get("pdf_path") or ""
		phase1_status = p1.get("phase1_status") or ""
		p2 = phase2_doc_log_map.get(doc_sha256, {})

		for meta in doc_to_edges.get(doc_sha256, []):
			edge_id = (meta.get("edge_id") or "").strip()
			var1 = (meta.get("source") or "").strip()
			var2 = (meta.get("target") or "").strip()

			chu_idx_list = edge_to_idxs.get(edge_id, [])
			chu_idx_groups = _format_consecutive_groups(chu_idx_list)
			evidence_text, n_found = _build_evidence_block(chu_idx_list, idx_to_text)
			has_missing = bool(chu_idx_list) and (n_found < len(chu_idx_list))

			row = {
				"doc_sha256": doc_sha256,
				"edge_id": edge_id,
				"var1": var1,
				"var2": var2,
				"paper_id_list": json.dumps(paper_id_list, ensure_ascii=False),
				"pdf_path": pdf_path,
				"phase1_status": phase1_status,
				"results_sentences": p2.get("results_sentences", ""),
				"edges_created_step": p2.get("edges_created_step", ""),
				"traces_created_step": p2.get("traces_created_step", ""),
				"variables_created_step": p2.get("variables_created_step", ""),
				"sec_per_doc": p2.get("sec_per_doc", ""),
				"phase2_timestamp_utc": p2.get("phase2_timestamp_utc", ""),
				"prompt_tokens": p2.get("prompt_tokens", ""),
				"completion_tokens": p2.get("completion_tokens", ""),
				"total_tokens": p2.get("total_tokens", ""),
				"est_prompt_tokens_step4": str(est_prompt_tokens_step4),
				"est_completion_tokens_step4": str(est_completion_tokens_step4),
				"est_total_tokens_step4": str(est_total_tokens_step4),
				"est_prompt_tokens_step5": str(est_prompt_tokens_step5),
				"est_completion_tokens_step5": str(est_completion_tokens_step5),
				"est_total_tokens_step5": str(est_total_tokens_step5),
				"est_total_tokens_phase2": str(est_total_tokens_phase2),
				"chu_idx_list_json": json.dumps(chu_idx_list, ensure_ascii=False),
				"chu_idx_list_str": " ".join(str(x) for x in chu_idx_list),
				"chu_idx_groups": chu_idx_groups,
				"n_chu_idx": len(chu_idx_list),
				"n_chunks_found": n_found,
				"has_missing_chunks": "1" if has_missing else "0",
				"evidence_text": evidence_text,
				"label": "",
			}
			_append_csv_row(out_csv, row, fieldnames)


# -------------------------
# Judge runner
# -------------------------

def _build_judge_llm(judge_code: str):
	"""
	Supported judge_code:
	- gpt-4.1-mini  (OpenAI, uses JUDGE_OPENAI_API_KEY)
	- claude-sonnet-4-5-20250929  (Anthropic, uses JUDGE_ANTHROPIC_API_KEY)
	Also accepts 'claude-sonnet-4-5' alias (mapped to the dated model string).
	"""
	if judge_code == "claude-sonnet-4-5":
		judge_code = "claude-sonnet-4-5-20250929"

	if judge_code == "gpt-4.1-mini":
		from langchain_openai import ChatOpenAI
		key = os.getenv("JUDGE_OPENAI_API_KEY", "").strip()
		if not key:
			raise RuntimeError("Missing JUDGE_OPENAI_API_KEY for gpt-4.1-mini judge.")
		return ChatOpenAI(
			api_key=key,
			model="gpt-4.1-mini",
			temperature=0,
		)

	if judge_code == "claude-sonnet-4-5-20250929":
		from langchain_anthropic import ChatAnthropic
		key = os.getenv("JUDGE_ANTHROPIC_API_KEY", "").strip()
		if not key:
			raise RuntimeError("Missing JUDGE_ANTHROPIC_API_KEY for Claude judge.")
		return ChatAnthropic(
			api_key=key,
			model="claude-sonnet-4-5-20250929",
			temperature=0,
		)

	raise ValueError(f"Unsupported judge_code: {judge_code}")


def _judge_prompt(var1: str, var2: str, chu_idx_list: List[int], evidence_text: str) -> str:
	# Keep rubric short but strict, force JSON output.
	return (
		"You are a blinded judge evaluating whether a reported variable–variable "
		"relationship is explicitly supported by the provided Results sentences.\n\n"
		"Inputs:\n"
		f"- Variable pair: ({var1}, {var2})\n"
		f"- chu_idx_list (ordered): {chu_idx_list}\n\n"
		"Provided Results sentences (each preceded by [chu_idx=...]):\n"
		f"{evidence_text}\n\n"
		"Rubric (choose exactly ONE label):\n"
		"1) C&G = Correct & Grounded: at least one sentence explicitly states an association/correlation/relationship between var1 and var2.\n"
		"2) INC = Incorrect (Misgrounded): sentences provided but they do NOT explicitly state an association between the pair.\n"
		"3) UNG = Ungrounded: chu_idx_list empty OR indices invalid/useless OR evidence text unusable.\n\n"
		"Rules:\n"
		"- Judge ONLY on provided sentences; do not infer missing context.\n"
		"- Do NOT require causality or direction.\n"
		"- If uncertain, choose INC.\n\n"
		"Return STRICT JSON only with keys: label, rationale.\n"
		"label must be one of: C&G, INC, UNG.\n"
	)


def run_judge(
	judge_code: str,
	in_csv: Path,
	out_csv: Path,
	sleep_s: float = 0.0,
	max_rows: Optional[int] = None,
) -> None:
	"""
	Run the judge, filling the single 'label' column.
	Resume-safe: skips rows where the label is already filled in out_csv.
	Writes output CSV with the same schema as input CSV.
	"""
	llm = _build_judge_llm(judge_code)

	# Read fieldnames from input CSV
	with in_csv.open("r", newline="", encoding="utf-8") as f:
		rdr = csv.DictReader(f)
		fieldnames = rdr.fieldnames
		if not fieldnames:
			raise RuntimeError("Input CSV has no header/columns.")

	# Only fill 'label'
	label_col = "label"
	key_cols = ["doc_sha256", "edge_id"]

	# Resume-safe: read existing out_csv and collect keys where label_col is nonempty
	done = set()
	if out_csv.exists():
		with out_csv.open("r", newline="", encoding="utf-8") as f:
			rdr = csv.DictReader(f)
			for row in rdr:
				key = tuple((row.get(k) or "").strip() for k in key_cols)
				if (row.get(label_col) or "").strip():
					done.add(key)

	n_written = 0
	with in_csv.open("r", newline="", encoding="utf-8") as f:
		r = csv.DictReader(f)
		for row in r:
			doc_sha256 = (row.get("doc_sha256") or "").strip()
			edge_id = (row.get("edge_id") or "").strip()
			key = (doc_sha256, edge_id)
			if key in done:
				continue

			var1 = (row.get("var1") or "").strip()
			var2 = (row.get("var2") or "").strip()
			chu_idx_list = _safe_json_loads(row.get("chu_idx_list_json"), default=[])
			if not isinstance(chu_idx_list, list):
				chu_idx_list = []
			chu_idx_list = [int(x) for x in chu_idx_list if str(x).strip().isdigit()]
			evidence_text = (row.get("evidence_text") or "").strip()

			# Deterministic UNG shortcut (saves tokens; matches rubric)
			if (not chu_idx_list) or (not evidence_text) or ("[chu_idx=" not in evidence_text):
				out_row = dict(row)
				out_row[label_col] = "UNG"
				_append_csv_row(out_csv, out_row, fieldnames)
				n_written += 1
				if max_rows is not None and n_written >= int(max_rows):
					return
				continue

			prompt = _judge_prompt(var1, var2, chu_idx_list, evidence_text)

			try:
				t0 = time.monotonic()
				resp = llm.invoke(prompt)
				_ = time.monotonic() - t0

				content = getattr(resp, "content", "")
				if not isinstance(content, str):
					content = str(content)

				data = _safe_json_loads(content, default={})
				label = str((data or {}).get("label") or "").strip()
				# Only accept C&G, INC, UNG
				if label not in {"C&G", "INC", "UNG"}:
					label = "INC"

				out_row = dict(row)
				out_row[label_col] = label
			except Exception as e:
				out_row = dict(row)
				out_row[label_col] = ""
			# Always write the row (with filled or empty label column)
			_append_csv_row(out_csv, out_row, fieldnames)
			n_written += 1

			if sleep_s and sleep_s > 0:
				time.sleep(float(sleep_s))

			if max_rows is not None and n_written >= int(max_rows):
				return


# -------------------------
# CLI
# -------------------------

def main():
	parser = argparse.ArgumentParser()

	sub = parser.add_subparsers(dest="cmd", required=True)

	p_exp = sub.add_parser("export", help="Export per-edge CSV from a model's Chroma snapshot.")
	p_exp.add_argument("--chroma_dir", required=True)
	p_exp.add_argument("--out_csv", required=True)
	p_exp.add_argument("--phase1_csv", default="data_registry/phase1_ingest_log.csv")
	p_exp.add_argument(
		"--phase2_doc_log_csv",
		default="",
		help="Optional per-doc Phase-2 log CSV to attach runtime/count metadata.",
	)
	p_exp.add_argument("--doc_limit", type=int, default=None)

	p_j = sub.add_parser("judge", help="Run one judge on an exported CSV, resume-safe.")
	p_j.add_argument("--judge_code", required=True)
	p_j.add_argument("--in_csv", required=True)
	p_j.add_argument("--out_csv", required=True)
	p_j.add_argument("--sleep_s", type=float, default=0.0)
	p_j.add_argument("--max_rows", type=int, default=None)

	args = parser.parse_args()

	if args.cmd == "export":
		export_adjudication_input(
			chroma_dir=Path(args.chroma_dir),
			out_csv=Path(args.out_csv),
			phase1_csv=Path(args.phase1_csv) if args.phase1_csv else None,
			doc_limit=args.doc_limit,
			phase2_doc_log_csv=Path(args.phase2_doc_log_csv) if args.phase2_doc_log_csv else None,
		)
		return

	if args.cmd == "judge":
		run_judge(
			judge_code=args.judge_code,
			in_csv=Path(args.in_csv),
			out_csv=Path(args.out_csv),
			sleep_s=float(args.sleep_s),
			max_rows=args.max_rows,
		)
		return


if __name__ == "__main__":
	main()
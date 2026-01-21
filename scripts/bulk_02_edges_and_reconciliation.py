import os
import csv
import json
import time
import uuid
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timezone

from modules.document_input import DocumentInputService
from modules.correlation_extraction import CorrelationExtractionService
from modules.variable_reconciliation import VariableReconciliationService


# -------------------------
# Environment contract
# -------------------------
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "/chroma"))
FILES_DIR = Path(os.getenv("FILES_DIR", "/files"))
RUN_ID = os.getenv("RUN_ID", "default")

# If you mount repo at /app inside container, this will resolve correctly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_BASE = PROJECT_ROOT / "snapshots" / "chroma_base"

RUN_LOG_PATH = PROJECT_ROOT / "data_registry" / "phase2_run_log.csv"


# -------------------------
# Logging helpers
# -------------------------
def _utc_now_iso():
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_csv_row(path: Path, row: dict):
	path.parent.mkdir(parents=True, exist_ok=True)
	write_header = not path.exists()
	with path.open("a", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=list(row.keys()))
		if write_header:
			w.writeheader()
		w.writerow(row)


# -------------------------
# Chroma bootstrap (idempotent)
# -------------------------
def _dir_is_empty(p: Path) -> bool:
	if not p.exists():
		return True
	return not any(p.iterdir())


def ensure_chroma_initialized_from_base(chroma_dir: Path):
	chroma_dir.mkdir(parents=True, exist_ok=True)

	if _dir_is_empty(chroma_dir):
		if not CHROMA_BASE.exists():
			raise RuntimeError(f"Missing base snapshot: {CHROMA_BASE}")
		# copy base snapshot into the mounted chroma_dir (model-specific)
		# This guarantees chroma_base is never touched.
		shutil.copytree(CHROMA_BASE, chroma_dir, dirs_exist_ok=True)


# -------------------------
# LLM builder (per model code name)
# -------------------------
def build_extraction_llm(model_code: str):
	"""
	Builds a LangChain chat model for Phase 2 extraction + reconciliation.
	We DO NOT use llm_factory.py here because that's frozen for Phase 1 (eligibility).
	"""
	model_code = (model_code or "").strip()

	# OpenAI
	if model_code == "gpt-5":
		try:
			from langchain_openai import ChatOpenAI
		except Exception as e:
			raise RuntimeError(
				"Missing dependency for OpenAI. Install langchain-openai."
			) from e

		api_key = os.getenv("OPENAI_GPT5_API_KEY") or ""
		if not api_key:
			raise RuntimeError("OPENAI_GPT5_API_KEY missing in .env")

		# Use explicit api_key to avoid any confusion with Phase 1 OPENAI_API_KEY
		return ChatOpenAI(
			model="gpt-5",
			temperature=0,
			api_key=api_key,
		)

	# Anthropic
	if model_code == "claude-4.5-sonnet":
		try:
			from langchain_anthropic import ChatAnthropic
		except Exception as e:
			raise RuntimeError(
				"Missing dependency for Anthropic. Install langchain-anthropic."
			) from e

		api_key = os.getenv("ANTHROPIC_API_KEY") or ""
		if not api_key:
			raise RuntimeError("ANTHROPIC_API_KEY missing in .env")

		return ChatAnthropic(
			model="claude-4.5-sonnet",
			temperature=0,
			api_key=api_key,
		)

	# Google Gemini
	if model_code == "gemini-3-pro":
		try:
			from langchain_google_genai import ChatGoogleGenerativeAI
		except Exception as e:
			raise RuntimeError(
				"Missing dependency for Google GenAI. Install langchain-google-genai."
			) from e

		api_key = os.getenv("GOOGLE_API_KEY") or ""
		if not api_key:
			raise RuntimeError("GOOGLE_API_KEY missing in .env")

		# langchain_google_genai uses env var GOOGLE_API_KEY by default too,
		# but we keep it explicit for clarity.
		return ChatGoogleGenerativeAI(
			model="gemini-3-pro",
			temperature=0,
			google_api_key=api_key,
		)

	# Mistral
	if model_code == "mistral-large-2":
		try:
			from langchain_mistralai import ChatMistralAI
		except Exception as e:
			raise RuntimeError(
				"Missing dependency for Mistral. Install langchain-mistralai."
			) from e

		api_key = os.getenv("MISTRAL_API_KEY") or ""
		if not api_key:
			raise RuntimeError("MISTRAL_API_KEY missing in .env")

		return ChatMistralAI(
			model="mistral-large-2",
			temperature=0,
			api_key=api_key,
		)

	# DeepSeek (OpenAI-compatible)
	if model_code == "deepseek-v3":
		try:
			from langchain_openai import ChatOpenAI
		except Exception as e:
			raise RuntimeError(
				"Missing dependency for OpenAI-compatible clients. "
				"Install langchain-openai."
			) from e

		api_key = os.getenv("DEEPSEEK_API_KEY") or ""
		if not api_key:
			raise RuntimeError("DEEPSEEK_API_KEY missing in .env")

		# If you use a different endpoint, set DEEPSEEK_BASE_URL in the container env.
		base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

		return ChatOpenAI(
			model="deepseek-v3",
			temperature=0,
			api_key=api_key,
			base_url=base_url,
		)

	# llama-4 (open-weight via OpenAI-compatible endpoint)
	if model_code == "llama-4":
		try:
			from langchain_openai import ChatOpenAI
		except Exception as e:
			raise RuntimeError(
				"Missing dependency for OpenAI-compatible clients. "
				"Install langchain-openai."
			) from e

		base_url = (os.getenv("LLAMA_ENDPOINT") or "").strip()
		api_key = (os.getenv("LLAMA_API_KEY") or "").strip()
		if not base_url:
			raise RuntimeError("LLAMA_ENDPOINT missing in .env")
		if not api_key:
			raise RuntimeError("LLAMA_API_KEY missing in .env")

		# Assume endpoint is OpenAI-compatible.
		# If your server uses a different model id, set LLAMA_MODEL in .env.
		llama_model = (os.getenv("LLAMA_MODEL") or "llama-4").strip()
		return ChatOpenAI(
			model=llama_model,
			temperature=0,
			api_key=api_key,
			base_url=base_url,
		)

	raise RuntimeError(
		"Unknown model_code. Choose one of:\n"
		"- gpt-5\n"
		"- claude-4.5-sonnet\n"
		"- gemini-3-pro\n"
		"- mistral-large-2\n"
		"- deepseek-v3\n"
		"- llama-4"
	)


# -------------------------
# Doc gating + results chunk selection
# -------------------------
def is_doc_phase2_eligible(doc_service: DocumentInputService, doc_sha256: str) -> bool:
	summary = doc_service._get_doc_summary(doc_sha256)
	if not summary:
		return False
	return bool(summary.get("scientific")) and (
		bool(summary.get("meta_analysis")) or bool(summary.get("empirical_study"))
	)


def get_results_child_chunks(doc_service: DocumentInputService, doc_sha256: str):
	"""
	Return list of {"chu_idx": int, "text": str} from child sentences labeled Results.
	"""
	rows = doc_service._read_child_rows_for_doc(doc_sha256)
	out = []
	for _, chu_idx, lab, sent in rows:
		if lab != "Results":
			continue
		text = (sent or "").strip()
		if not text:
			continue
		try:
			out.append({"chu_idx": int(chu_idx), "text": text})
		except Exception:
			continue
	return out


def get_doc_done_flag_key(model_code: str) -> str:
	# stored in docs metadata (per model-specific Chroma)
	return f"phase2_done_{model_code}"


def is_doc_already_done(doc_service: DocumentInputService, doc_sha256: str, model_code: str) -> bool:
	try:
		# Use docs collection directly for a lightweight flag lookup
		metas = doc_service.docs_db._collection.get(
			where={"doc_sha256": doc_sha256},
			include=["metadatas"],
			limit=1,
		).get("metadatas") or []
		if not metas:
			return False
		meta = metas[0] or {}
		key = get_doc_done_flag_key(model_code)
		v = meta.get(key)
		if isinstance(v, bool):
			return v
		if isinstance(v, str):
			return v.strip().lower() in {"1", "true", "t", "yes", "y"}
		if isinstance(v, (int, float)):
			return bool(int(v))
		return False
	except Exception:
		return False


def mark_doc_done(doc_service: DocumentInputService, doc_sha256: str, model_code: str):
	"""
	Update docs metadata with a per-model completion flag.
	We re-add the same doc id with same text (or empty) and updated metadata.
	DocumentInputService uses Chroma; simplest is upsert via add_texts.
	"""
	key = get_doc_done_flag_key(model_code)
	try:
		got = doc_service.docs_db._collection.get(
			where={"doc_sha256": doc_sha256},
			include=["ids", "documents", "metadatas"],
			limit=1,
		)
		ids = got.get("ids") or []
		if not ids:
			return
		doc_id = ids[0]
		doc_texts = got.get("documents") or [""]
		meta = (got.get("metadatas") or [{}])[0] or {}
		meta[key] = True
		meta[f"{key}_at"] = _utc_now_iso()

		# Upsert by re-adding with same id
		doc_service.docs_db.add_texts(
			texts=[doc_texts[0] or ""],
			ids=[doc_id],
			metadatas=[meta],
		)
		doc_service.docs_db.persist()
	except Exception:
		# non-fatal; resume-safety still mostly works via existing edges/trace
		return


def count_doc_edges(doc_service: DocumentInputService, doc_sha256: str) -> int:
	try:
		got = doc_service.edges_db._collection.get(
			where={"doc_sha256": doc_sha256},
			include=["ids"],
		)
		return len(got.get("ids") or [])
	except Exception:
		return 0


def count_doc_traces_and_ungrounded(doc_service: DocumentInputService, doc_sha256: str):
	"""
	Ungrounded edges = trace record with chu_idx_list == [] (stored as JSON string).
	This supports Phase 3 metric computation without exporting edge CSVs.
	"""
	try:
		got = doc_service.trace_db._collection.get(
			where={"doc_sha256": doc_sha256, "evidence_type": "edge"},
			include=["metadatas"],
		)
		metas = got.get("metadatas") or []
		total = len(metas)
		ungrounded = 0
		for m in metas:
			raw = (m or {}).get("chu_idx_list")
			if raw is None:
				ungrounded += 1
				continue
			if isinstance(raw, list):
				if len(raw) == 0:
					ungrounded += 1
				continue
			if isinstance(raw, str):
				s = raw.strip()
				if s in {"[]", ""}:
					ungrounded += 1
					continue
				try:
					arr = json.loads(s)
					if isinstance(arr, list) and len(arr) == 0:
						ungrounded += 1
				except Exception:
					# if malformed, treat as ungrounded
					ungrounded += 1
		return total, ungrounded
	except Exception:
		return 0, 0


# -------------------------
# Main
# -------------------------
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--model_code",
		required=True,
		choices=[
			"gpt-5",
			"claude-4.5-sonnet",
			"gemini-3-pro",
			"mistral-large-2",
			"deepseek-v3",
			"llama-4",
		],
	)
	parser.add_argument("--window_size", type=int, default=8)
	parser.add_argument("--max_docs", type=int, default=0)  # 0 = no limit
	args = parser.parse_args()

	model_code = args.model_code

	if not FILES_DIR.exists():
		raise RuntimeError(f"FILES_DIR does not exist: {FILES_DIR}")

	CHROMA_DIR.mkdir(parents=True, exist_ok=True)

	# Initialize model-specific Chroma from frozen base (only if empty).
	ensure_chroma_initialized_from_base(CHROMA_DIR)

	# Build the LLM for this run
	llm = build_extraction_llm(model_code=model_code)

	# Services (must match app.py tabs 4-5 behavior)
	doc_service = DocumentInputService(
		chroma_dir=CHROMA_DIR,
		files_dir=FILES_DIR,
	)

	corr_extractor = CorrelationExtractionService(
		llm=llm,
		window_size=args.window_size,
	)

	var_reconciler = VariableReconciliationService(
		llm=llm,
	)

	run_id = f"{model_code}__{RUN_ID}__{uuid.uuid4().hex[:10]}"
	started_at = _utc_now_iso()
	t0 = time.time()

	# Enumerate docs from docs collection
	# We assume Phase 1 already indexed all eligible PDFs into docs_db
	got = doc_service.docs_db._collection.get(include=["metadatas"])
	metas = got.get("metadatas") or []

	# Unique doc_sha256s
	doc_ids = []
	seen = set()
	for m in metas:
		doc_sha = (m or {}).get("doc_sha256")
		if not doc_sha or not isinstance(doc_sha, str):
			continue
		if doc_sha in seen:
			continue
		seen.add(doc_sha)
		doc_ids.append(doc_sha)

	docs_seen = 0
	docs_processed = 0
	edges_created_total = 0
	variables_created_total = 0
	traces_created_total = 0

	# Optional per-doc log for later runtime/density calculations (article-level)
	doc_log_path = PROJECT_ROOT / "data_registry" / f"phase2_doc_log__{model_code}.csv"

	try:
		for doc_sha256 in doc_ids:
			docs_seen += 1

			if args.max_docs and docs_processed >= args.max_docs:
				break

			# Gate by Phase 1 doc type flags stored in docs metadata
			if not is_doc_phase2_eligible(doc_service, doc_sha256):
				continue

			# Resume-safe per-model flag
			if is_doc_already_done(doc_service, doc_sha256, model_code):
				continue

			t_doc0 = time.time()

			# Tab 4 behavior: Results-only child sentences
			results_chunks = get_results_child_chunks(doc_service, doc_sha256)

			# Extract edges (writes to edges + trace inside model-specific Chroma)
			res_extract = corr_extractor.extract_edges(
				doc_service=doc_service,
				doc_sha256=doc_sha256,
				results_chunks=results_chunks,
			)
			if not res_extract.get("ok"):
				raise RuntimeError(
					f"extract_edges failed for {doc_sha256}: {res_extract.get('error')}"
				)

			# Tab 5 behavior: reconcile variables (writes to variables; updates edges)
			res_recon = var_reconciler.reconcile_variables(
				doc_service=doc_service,
				doc_sha256=doc_sha256,
			)
			if not res_recon.get("ok"):
				raise RuntimeError(
					f"reconcile_variables failed for {doc_sha256}: {res_recon.get('error')}"
				)

			docs_processed += 1
			edges_created_total += int(res_extract.get("edges_created") or 0)
			traces_created_total += int(res_extract.get("traces_created") or 0)
			variables_created_total += int(res_recon.get("variables_created") or 0)

			# Mark doc done for this model (in docs metadata)
			mark_doc_done(doc_service, doc_sha256, model_code)

			# Compute article-level metrics proxies now (for later aggregation)
			n_edges = count_doc_edges(doc_service, doc_sha256)
			n_traces, n_ungrounded = count_doc_traces_and_ungrounded(doc_service, doc_sha256)

			_append_csv_row(
				doc_log_path,
				{
					"run_id": run_id,
					"model_code": model_code,
					"doc_sha256": doc_sha256,
					"results_sentences": len(results_chunks),
					"edges_created_step": int(res_extract.get("edges_created") or 0),
					"traces_created_step": int(res_extract.get("traces_created") or 0),
					"variables_created_step": int(res_recon.get("variables_created") or 0),
					"edges_in_db": n_edges,
					"trace_in_db": n_traces,
					"ungrounded_edges": n_ungrounded,
					"sec_per_doc": round(time.time() - t_doc0, 3),
					"run_tag": RUN_ID,
					"timestamp_utc": _utc_now_iso(),
				},
			)

			print(f"[{docs_processed}] ok doc_sha256={doc_sha256} edges={n_edges} ungrounded={n_ungrounded}")

		status = "ok"
		error_msg = ""

	except Exception as e:
		status = "error"
		error_msg = str(e)

	ended_at = _utc_now_iso()
	elapsed = round(time.time() - t0, 3)

	_append_csv_row(
		RUN_LOG_PATH,
		{
			"run_id": run_id,
			"model_code": model_code,
			"started_at_utc": started_at,
			"ended_at_utc": ended_at,
			"elapsed_sec": elapsed,
			"docs_seen": docs_seen,
			"docs_processed": docs_processed,
			"edges_created_total": edges_created_total,
			"variables_created_total": variables_created_total,
			"traces_created_total": traces_created_total,
			"status": status,
			"error": error_msg,
			"run_tag": RUN_ID,
		},
	)

	if status != "ok":
		raise RuntimeError(error_msg)

	print("[DONE] Phase 2 extraction + reconciliation finished (per-model Chroma).")


if __name__ == "__main__":
	main()
import os
import csv
from pathlib import Path
import pandas as pd

from modules.document_input import DocumentInputService
from modules.chunk_imrd_labeling import IMRDChunkClassifier
from modules.document_eligibility import DocumentEligibilityService


# -------------------------
# Environment contract
# -------------------------
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "/chroma"))
FILES_DIR = Path(os.getenv("FILES_DIR", "/files"))
RUN_ID = os.getenv("RUN_ID", "default")

if not FILES_DIR.exists():
	raise RuntimeError(f"FILES_DIR does not exist: {FILES_DIR}")

if not CHROMA_DIR.exists():
	CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "data_registry" / "search_log.xlsx"
LOG_PATH = PROJECT_ROOT / "data_registry" / "phase1_ingest_log.csv"

if not REGISTRY_PATH.exists():
	raise RuntimeError(f"Missing registry: {REGISTRY_PATH}")

completed_ids = set()
if LOG_PATH.exists():
    prev = pd.read_csv(LOG_PATH)
    if "status" in prev.columns and "paper_id" in prev.columns:
        completed_ids = set(
            prev.loc[prev["status"] == "ok", "paper_id"]
            .astype(str)
            .tolist()
        )


# -------------------------
# Initialize services
# -------------------------
doc_service = DocumentInputService(
	chroma_dir=CHROMA_DIR,
	files_dir=FILES_DIR,
)

chunk_labeler = IMRDChunkClassifier(
	chroma_dir=CHROMA_DIR,
	parents_collection="chunks",
	children_collection="evidences",
)

doc_eligibility = DocumentEligibilityService(
	chroma_dir=CHROMA_DIR,
)


# -------------------------
# Helper for incremental logging
# -------------------------
def append_log(row):
    write_header = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# -------------------------
# Main loop
# -------------------------
df = pd.read_excel(REGISTRY_PATH)

total = int((df["eligible"].astype(str).str.lower() == "yes").sum())
processed = 0

for idx, row in df.iterrows():
	paper_id = f"row_{idx+1:03d}"
	pdf_rel = row["filename"]
	eligible = str(row["eligible"]).strip().lower() == "yes"

	if paper_id in completed_ids:
		continue

	if not eligible:
		continue

	processed += 1
	print(f"[{processed}/{total}] Processing {pdf_rel}")

	pdf_path = FILES_DIR / pdf_rel
	if not pdf_path.exists():
		append_log({
			"paper_id": paper_id,
			"pdf_path": pdf_rel,
			"run_id": RUN_ID,
			"status": "error",
			"error": "PDF not found",
			"phase": "phase1",
		})
		continue

	try:
		# 1) Document input
		res_index = doc_service.index_pdf(str(pdf_path))
		if not res_index.get("ok"):
			raise RuntimeError(res_index.get("error"))

		doc_sha256 = res_index["doc_sha256"]

		# 2) IMRD labeling (deterministic, model-agnostic; run once per document)
		chunk_labeler.label_doc_parent_child(
			doc_sha256=doc_sha256,
			batch_size=200,
			skip_labeled=True,
		)

		# 3) Eligibility classification
		sections = doc_service.get_imrd_section_texts(doc_sha256)
		methods_text = sections.get("Methods") or ""
		results_text = sections.get("Results") or ""
		if not results_text.strip():
			raise RuntimeError("Empty Results section after IMRD labeling")

		res_elig = doc_eligibility.evaluate_from_text(
			doc_sha256=doc_sha256,
			methods_text=methods_text,
			results_text=results_text,
		)

		append_log({
			"paper_id": paper_id,
			"doc_sha256": doc_sha256,
			"pdf_path": pdf_rel,
			"run_id": RUN_ID,
			"status": "ok",
			"is_scientific": res_elig.get("is_scientific"),
			"is_meta_analysis": res_elig.get("is_meta_analysis"),
			"is_empirical": res_elig.get("is_empirical"),
			"phase": "phase1",
		})

	except Exception as e:
		append_log({
			"paper_id": paper_id,
			"pdf_path": pdf_rel,
			"run_id": RUN_ID,
			"status": "error",
			"error": str(e),
			"phase": "phase1",
		})


print("[DONE] Phase 1 ingest finished (resume-safe).")
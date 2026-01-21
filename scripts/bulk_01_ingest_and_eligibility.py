import os
import csv
from pathlib import Path

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
REGISTRY_PATH = PROJECT_ROOT / "data_registry" / "papers.csv"
LOG_PATH = PROJECT_ROOT / "data_registry" / "phase1_ingest_log.csv"

if not REGISTRY_PATH.exists():
	raise RuntimeError(f"Missing registry: {REGISTRY_PATH}")


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
# Helpers
# -------------------------
def parse_bool(x):
	return str(x).lower() in {"1", "true", "yes", "y"}


# -------------------------
# Main loop
# -------------------------
rows_out = []

with REGISTRY_PATH.open(newline="", encoding="utf-8") as f:
	reader = csv.DictReader(f)
	papers = list(reader)

for row in papers:
	paper_id = row.get("paper_id")
	pdf_rel = row.get("pdf_path")
	eligible = parse_bool(row.get("eligible_full_text", "false"))

	if not eligible:
		continue

	pdf_path = FILES_DIR / pdf_rel
	if not pdf_path.exists():
		rows_out.append({
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

		rows_out.append({
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
		rows_out.append({
			"paper_id": paper_id,
			"pdf_path": pdf_rel,
			"run_id": RUN_ID,
			"status": "error",
			"error": str(e),
			"phase": "phase1",
		})


# -------------------------
# Write log
# -------------------------
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

fieldnames = sorted({k for r in rows_out for k in r.keys()})

with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
	writer = csv.DictWriter(f, fieldnames=fieldnames)
	writer.writeheader()
	for r in rows_out:
		writer.writerow(r)

print(f"[DONE] Phase 1 ingest complete. Log written to {LOG_PATH}")
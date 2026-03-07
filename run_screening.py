import argparse
from pathlib import Path

import pandas as pd

from src.resume_screening import ResumeScreeningSystem, load_skill_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume screening and ranking system")
    parser.add_argument("--resumes", default="data/sample_resumes.csv", help="Path to resume CSV")
    parser.add_argument("--job", default="data/sample_job_description.txt", help="Path to job description TXT")
    parser.add_argument("--skills", default=None, help="Optional skill taxonomy JSON path")
    parser.add_argument("--output", default="outputs/ranked_candidates.csv", help="Output CSV path")
    args = parser.parse_args()

    resumes_path = Path(args.resumes)
    job_path = Path(args.job)
    output_path = Path(args.output)

    resumes_df = pd.read_csv(resumes_path)
    job_description = job_path.read_text(encoding="utf-8")

    taxonomy = load_skill_taxonomy(args.skills)
    screener = ResumeScreeningSystem(skill_taxonomy=taxonomy)

    ranked = screener.rank_resumes(resumes_df, job_description)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_path, index=False)

    display_cols = ["rank", "name", "final_score", "matched_skills", "missing_required_skills"]
    print(ranked[display_cols].to_string(index=False))
    print(f"\nSaved full results to: {output_path}")


if __name__ == "__main__":
    main()

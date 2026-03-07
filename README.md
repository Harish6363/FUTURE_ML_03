# Resume / Candidate Screening System

Machine Learning Task 3 (2026) - Future Interns

## What this project does
This system screens resumes against a job description, then ranks candidates by role fit.

It provides:
- Resume text preprocessing
- NLP-based skill extraction
- Job description parsing
- Resume-to-job similarity scoring
- Candidate ranking
- Missing skill identification
- Human-readable explanation per rank

## Project structure
- `src/resume_screening.py`: core NLP + ranking engine
- `run_screening.py`: command-line entrypoint
- `data/sample_resumes.csv`: demo resumes
- `data/sample_job_description.txt`: demo role description
- `data/skill_taxonomy.json`: editable skill alias dictionary
- `outputs/ranked_candidates.csv`: generated screening result

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python run_screening.py \
  --resumes data/sample_resumes.csv \
  --job data/sample_job_description.txt \
  --skills data/skill_taxonomy.json \
  --output outputs/ranked_candidates.csv
```

## Scoring logic
Final score is a weighted combination:
- `50%` semantic similarity (TF-IDF + cosine similarity)
- `35%` required skill match rate
- `10%` preferred skill match rate
- `5%` years of experience (normalized)

Formula:

`Final Score = 100 * (0.50*semantic + 0.35*required + 0.10*preferred + 0.05*experience)`

## How missing skills are identified
- The system extracts skills from both resume and job description using a skill taxonomy.
- `missing_required_skills = required_skills - candidate_skills`
- Output includes missing items for each candidate.

## Input format
Resume CSV requires at least:
- `name`
- `resume_text`

Optional columns (kept in output):
- `candidate_id`
- any metadata fields

## Example output columns
- `rank`
- `name`
- `final_score`
- `semantic_similarity`
- `required_skill_score`
- `preferred_skill_score`
- `experience_score`
- `matched_skills`
- `missing_required_skills`
- `explanation`

## Notes
- This implementation uses deterministic scoring and interpretable rules for HR transparency.
- You can tune weights and expand skill aliases in `data/skill_taxonomy.json`.

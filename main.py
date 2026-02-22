import os
from preprocess import clean_text
from skill_extractor import extract_skills
from scorer import calculate_similarity, skill_gap
import csv


# ===============================
# 1️⃣  SET BASE DIRECTORY PATH
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

job_path = os.path.join(BASE_DIR, "data", "job_description.txt")
resume_folder = os.path.join(BASE_DIR, "data", "resumes")


# ===============================
# 2️⃣  LOAD JOB DESCRIPTION
# ===============================

if not os.path.exists(job_path):
    print("❌ job_description.txt not found in data folder.")
    exit()

with open(job_path, "r", encoding="utf-8") as f:
    job_text = f.read()

clean_job = clean_text(job_text)
job_skills = extract_skills(clean_job)

if len(job_skills) == 0:
    print("⚠ Warning: No skills detected in job description.")
    

# ===============================
# 3️⃣  PROCESS RESUMES
# ===============================

if not os.path.exists(resume_folder):
    print("❌ resumes folder not found inside data.")
    exit()

resume_files = [f for f in os.listdir(resume_folder) if f.endswith(".txt")]

if len(resume_files) == 0:
    print("⚠ No .txt resume files found in resumes folder.")
    exit()

results = []

for file in resume_files:
    file_path = os.path.join(resume_folder, file)

    with open(file_path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    clean_resume = clean_text(resume_text)
    resume_skills = extract_skills(clean_resume)

    # -------------------------------
    # 4️⃣  CALCULATE SCORES
    # -------------------------------

    similarity_score = calculate_similarity(clean_job, clean_resume)  # 0–100

    if len(job_skills) > 0:
        skill_score = len(resume_skills) / len(job_skills)
    else:
        skill_score = 0

    # Hybrid scoring formula
    final_score = (0.7 * (similarity_score / 100)) + (0.3 * skill_score)
    final_score = final_score * 100

    missing_skills = skill_gap(job_skills, resume_skills)

    results.append({
        "name": file,
        "score": round(final_score, 2),
        "matched_skills": resume_skills,
        "missing_skills": missing_skills
    })


# ===============================
# 5️⃣  RANK CANDIDATES
# ===============================

ranked = sorted(results, key=lambda x: x['score'], reverse=True)



# ===============================
# 6️⃣  PRINT RESULTS
# ===============================

print("\n========== Resume Screening Results ==========\n")

for rank, candidate in enumerate(ranked, start=1):
    print(f"Rank {rank}")
    print("Candidate:", candidate["name"])
    print("Final Score:", candidate["score"], "%")
    print("Matched Skills:", candidate["matched_skills"])
    print("Missing Skills:", candidate["missing_skills"])
    print("---------------------------------------------")

print("\n✅ Screening Completed Successfully.\n")

# ===============================
# 7️⃣  EXPORT RESULTS TO CSV
# ===============================

output_path = os.path.join(BASE_DIR, "results.csv")

with open(output_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Header
    writer.writerow(["Rank", "Candidate", "Final Score (%)", "Matched Skills", "Missing Skills"])
    
    # Data rows
    for rank, candidate in enumerate(ranked, start=1):
        writer.writerow([
            rank,
            candidate["name"],
            candidate["score"],
            ", ".join(candidate["matched_skills"]),
            ", ".join(candidate["missing_skills"])
        ])

print(f"\n📁 Results exported successfully to: {output_path}")
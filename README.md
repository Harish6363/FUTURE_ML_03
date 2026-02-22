🚀 Resume / Candidate Screening System (ML + NLP)

A Machine Learning-based Resume Screening and Candidate Ranking System that automatically evaluates resumes against a job description using Natural Language Processing (NLP).

This project simulates how modern HR-tech platforms shortlist candidates efficiently.

📌 Problem Statement

Recruiters receive hundreds of resumes for a single job role.

Manual screening:

Is time-consuming

Is inconsistent

Is prone to human bias

Increases recruiter workload

This system automates resume analysis and ranking using Machine Learning techniques.

🎯 Objective

Build a decision-support ML system that:

Cleans and preprocesses resume text

Extracts important skills

Matches resumes with job descriptions

Scores and ranks candidates

Identifies missing required skills

🧠 System Workflow

Resumes (Text Files)
→ Text Cleaning (NLP Preprocessing)
→ Skill Extraction
→ Job Description Processing
→ TF-IDF Vectorization
→ Cosine Similarity
→ Hybrid Scoring
→ Candidate Ranking + Skill Gap Detection

🛠️ Tech Stack

Python

NLTK (Text preprocessing)

Scikit-learn (TF-IDF, Cosine Similarity)

Matplotlib (Visualization)

CSV (Export results)

⚙️ Key Features

✔ Resume text cleaning and preprocessing
✔ Skill extraction using predefined skill database
✔ Job description parsing
✔ TF-IDF similarity scoring
✔ Hybrid scoring logic
✔ Candidate ranking
✔ Skill gap identification
✔ CSV export for recruiters
✔ Score visualization using bar chart

📊 Scoring Logic

The final score is calculated using a hybrid approach:

Final Score =
70% → Text Similarity (TF-IDF + Cosine Similarity)
30% → Skill Match Ratio

Skill Match Ratio =
(Number of matched skills) / (Total required job skills)

This improves ranking reliability compared to using text similarity alone.

📁 Project Structure

FUTURE_ML_03/
│
├── data/
│ ├── resumes/
│ ├── job_description.txt
│
├── src/
│ ├── preprocess.py
│ ├── skill_extractor.py
│ ├── scorer.py
│ ├── main.py
│
├── results.csv
├── README.md
├── requirements.txt

▶️ How to Run
1️⃣ Install Dependencies

pip install -r requirements.txt

2️⃣ Run the System

python src/main.py

3️⃣ Output

Ranked candidates displayed in terminal

results.csv file generated

Bar chart visualization displayed

📈 Sample Output

Rank 1
Candidate: resume1.txt
Final Score: 26.24 %
Matched Skills: ['python', 'machine learning', 'flask', 'sql', 'html']
Missing Skills: ['aws', 'nlp']

💼 Business Value

This system helps organizations:

Reduce manual resume screening time

Standardize candidate evaluation

Identify skill gaps instantly

Improve hiring efficiency

Enable data-driven recruitment decisions

🚀 Future Improvements

PDF resume parsing

spaCy-based advanced skill extraction

Weighted required skills

Web-based UI using Flask

Database integration

Resume classification model




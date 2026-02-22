SKILLS_DB = [
    "python", "machine learning", "deep learning",
    "flask", "django", "sql", "mongodb",
    "html", "css", "javascript",
    "aws", "data analysis", "nlp",
    "scikit-learn", "tensorflow"
]

def extract_skills(text):
    found_skills = []
    for skill in SKILLS_DB:
        if skill in text:
            found_skills.append(skill)
    return found_skills
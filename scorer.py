from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(job_text, resume_text):
    documents = [job_text, resume_text]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return float(score[0][0]) * 100


def skill_gap(job_skills, resume_skills):
    missing = list(set(job_skills) - set(resume_skills))
    return missing
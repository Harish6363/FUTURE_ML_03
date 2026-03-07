from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS


DEFAULT_SKILL_TAXONOMY: Dict[str, List[str]] = {
    "python": ["python", "python3"],
    "sql": ["sql", "postgresql", "mysql", "sqlite"],
    "machine_learning": ["machine learning", "ml", "supervised learning", "unsupervised learning"],
    "deep_learning": ["deep learning", "neural network", "cnn", "rnn", "transformer"],
    "nlp": ["nlp", "natural language processing", "text mining", "bert", "spacy", "nltk"],
    "data_analysis": ["data analysis", "data analytics", "analysis", "eda", "statistics"],
    "pandas": ["pandas"],
    "scikit_learn": ["scikit-learn", "sklearn"],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "visualization": ["matplotlib", "seaborn", "plotly", "tableau", "power bi"],
    "cloud": ["aws", "azure", "gcp", "cloud"],
    "deployment": ["docker", "kubernetes", "fastapi", "flask", "api deployment", "mlops"],
    "communication": ["communication", "stakeholder management", "presentation"],
    "git": ["git", "github", "gitlab"],
}


@dataclass
class ScoreWeights:
    semantic_similarity: float = 0.50
    required_skills: float = 0.35
    preferred_skills: float = 0.10
    experience: float = 0.05


class ResumeScreeningSystem:
    def __init__(
        self,
        skill_taxonomy: Dict[str, List[str]] | None = None,
        weights: ScoreWeights | None = None,
    ) -> None:
        self.stop_words: Set[str] = set(STOP_WORDS)
        self.nlp = English()
        self.stemmer = PorterStemmer()
        self.skill_taxonomy = skill_taxonomy or DEFAULT_SKILL_TAXONOMY
        self.weights = weights or ScoreWeights()

    def preprocess_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        doc = self.nlp(text)
        tokens = [self.stemmer.stem(t.text) for t in doc if t.text not in self.stop_words and len(t.text) > 1]
        return " ".join(tokens)

    def normalize_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def extract_skills(self, text: str) -> Set[str]:
        clean_text = self.normalize_text(text)
        found_skills: Set[str] = set()

        for skill, aliases in self.skill_taxonomy.items():
            for alias in aliases:
                pattern = r"\b" + re.escape(alias.lower()) + r"\b"
                if re.search(pattern, clean_text):
                    found_skills.add(skill)
                    break
        return found_skills

    def extract_years_experience(self, text: str) -> float:
        text = str(text).lower()
        patterns = [
            r"(\d+(?:\.\d+)?)\+?\s+years?\s+(?:of\s+)?experience",
            r"experience\s+of\s+(\d+(?:\.\d+)?)\+?\s+years?",
            r"(\d+(?:\.\d+)?)\+?\s+yrs?",
        ]

        values: List[float] = []
        for pattern in patterns:
            values.extend(float(v) for v in re.findall(pattern, text))

        return max(values) if values else 0.0

    def parse_job_requirements(self, job_description: str) -> Tuple[Set[str], Set[str]]:
        jd_skills = self.extract_skills(job_description)

        required: Set[str] = set()
        preferred: Set[str] = set()

        for skill in jd_skills:
            required_markers = [
                f"required {skill}",
                f"must have {skill}",
                f"mandatory {skill}",
            ]
            preferred_markers = [
                f"preferred {skill}",
                f"nice to have {skill}",
                f"plus {skill}",
            ]
            jd_lower = job_description.lower()

            if any(marker in jd_lower for marker in required_markers):
                required.add(skill)
            elif any(marker in jd_lower for marker in preferred_markers):
                preferred.add(skill)
            else:
                required.add(skill)

        return required, preferred

    def rank_resumes(
        self,
        resumes_df: pd.DataFrame,
        job_description: str,
        required_skills: Set[str] | None = None,
        preferred_skills: Set[str] | None = None,
    ) -> pd.DataFrame:
        if "resume_text" not in resumes_df.columns:
            raise ValueError("Input DataFrame must contain a 'resume_text' column.")

        required, preferred = self.parse_job_requirements(job_description)
        if required_skills is not None:
            required = required_skills
        if preferred_skills is not None:
            preferred = preferred_skills

        working_df = resumes_df.copy()
        working_df["processed_resume"] = working_df["resume_text"].apply(self.preprocess_text)
        processed_jd = self.preprocess_text(job_description)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform([processed_jd] + working_df["processed_resume"].tolist())
        jd_vector = matrix[0:1]
        resume_vectors = matrix[1:]

        semantic_scores = cosine_similarity(resume_vectors, jd_vector).flatten()

        extracted_skills = working_df["resume_text"].apply(self.extract_skills)
        years_exp = working_df["resume_text"].apply(self.extract_years_experience)

        required_count = max(1, len(required))
        preferred_count = max(1, len(preferred))

        req_skill_scores = extracted_skills.apply(lambda s: len(s & required) / required_count if required else 0.0)
        pref_skill_scores = extracted_skills.apply(lambda s: len(s & preferred) / preferred_count if preferred else 0.0)

        exp_normalized = np.clip(years_exp / 10.0, 0, 1)

        final_score = (
            self.weights.semantic_similarity * semantic_scores
            + self.weights.required_skills * req_skill_scores
            + self.weights.preferred_skills * pref_skill_scores
            + self.weights.experience * exp_normalized
        )

        working_df["semantic_similarity"] = semantic_scores.round(4)
        working_df["required_skill_score"] = req_skill_scores.round(4)
        working_df["preferred_skill_score"] = pref_skill_scores.round(4)
        working_df["experience_score"] = exp_normalized.round(4)
        working_df["final_score"] = (final_score * 100).round(2)
        working_df["matched_skills"] = extracted_skills.apply(lambda s: sorted(list(s)))
        working_df["missing_required_skills"] = extracted_skills.apply(lambda s: sorted(list(required - s)))
        working_df["years_experience_detected"] = years_exp

        working_df = working_df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
        working_df["rank"] = working_df.index + 1

        working_df["explanation"] = working_df.apply(
            lambda row: self._build_explanation(row, required, preferred), axis=1
        )

        return working_df

    def _build_explanation(self, row: pd.Series, required: Set[str], preferred: Set[str]) -> str:
        req_matched = sorted(list(set(row["matched_skills"]) & required))
        pref_matched = sorted(list(set(row["matched_skills"]) & preferred))

        parts = [
            f"Semantic relevance: {row['semantic_similarity']:.2f}",
            f"Required skills matched: {len(req_matched)}/{len(required)}",
        ]
        if preferred:
            parts.append(f"Preferred skills matched: {len(pref_matched)}/{len(preferred)}")
        if row["missing_required_skills"]:
            parts.append("Missing: " + ", ".join(row["missing_required_skills"]))

        return " | ".join(parts)


def load_skill_taxonomy(path: str | Path | None) -> Dict[str, List[str]]:
    if not path:
        return DEFAULT_SKILL_TAXONOMY

    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    return {str(k): [str(x) for x in v] for k, v in data.items()}

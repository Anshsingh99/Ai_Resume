from fuzzywuzzy import fuzz
import spacy
import pdfplumber
import spacy
from spacy.matcher import Matcher
import pdfplumber
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from fuzzywuzzy import fuzz

text = ""
nlp = spacy.load("en_core_web_trf")
with pdfplumber.open("sample_resume.pdf") as pdf:
    for p in pdf.pages:
        text += p.extract_text() + "\n"
doc = nlp(text)
def extract_skills_and_titles(text):
    doc = nlp(text)
    
    skills = []
    job_titles = []

    # Extract skills from Named Entities
    for ent in doc.ents:
        # Ignore organizations, locations, dates, and numbers
        if ent.label_ in ["ORG", "GPE", "PERSON", "DATE", "CARDINAL", "MONEY", "TIME","Work_Of_Art"]:
            continue
        
        # Extract job titles (better entity labels)
        if ent.label_ in ["JOB_TITLE", "NORP",  "TITLE"]:
            job_titles.append(ent.text)
        else:
            skills.append(ent.text)

    # **Custom Job Title Extraction using Matcher**
    matcher = Matcher(nlp.vocab)
    job_title_patterns = [
        [{"POS": "PROPN"}, {"POS": "PROPN"}],  # Example: "Software Engineer"
        [{"POS": "ADJ"}, {"POS": "NOUN"}],  # Example: "Senior Developer"
    ]
    matcher.add("JOB_TITLE", job_title_patterns)
    matches = matcher(doc)

    for match_id, start, end in matches:
        job_titles.append(doc[start:end].text)
    print("skills",skills)
    print("job titles", job_titles)
    return set(skills), set(job_titles)

resume_skills , job_titles = extract_skills_and_titles(text)



def match_skills(resume_skills, job_skills):
    matched = {}
    for r_skill in resume_skills:
        best_match = max(job_skills, key=lambda j_skill: fuzz.ratio(r_skill.lower(), j_skill.lower()))
        match_score = fuzz.ratio(r_skill.lower(), best_match.lower())
        if match_score > 70:  # Acceptable similarity
            matched[r_skill] = best_match
    return matched

# Example usage
job_description_skills = {"Python", "TensorFlow"}
matched_skills = match_skills(resume_skills, job_description_skills)
print("Matched Skills:", matched_skills)


# Sample dataset with different numbers of required skills
X_train = np.array([
    [2, 100],  # 2 required, all matched
    [3, 90],   # 3 required, mostly matched
    [5, 100],  # 5 required, all matched
    [7, 85],   # 7 required, high match
    [10, 70],  # 10 required, medium match
    [12, 50],  # 12 required, poor match
])
y_train = np.array([100, 90, 100, 85, 70, 50])  # Corresponding scores

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Function to predict resume score dynamically
def predict_resume_score(matched_skills, required_skills):
    num_required = len(required_skills)  # Total required skills
    num_matched = len(matched_skills)  # Total matched skills

    if num_required == 0:
        return 0  # No skills required means zero score

    # Compute similarity scores for each required skill
    match_scores = [
        fuzz.ratio(matched_skills.get(req, ""), req) for req in required_skills
    ]

    # Weighted match calculation
    weighted_match_score = sum(
        1.0 if score >= 90 else 0.5 if score >= 60 else 0 for score in match_scores
    ) / num_required * 100  # Normalize to percentage

    # Special case: if all required skills match fully, give 100
    if num_matched == num_required and min(match_scores) >= 90:
        return 100

    # Predict score using trained model
    return model.predict(np.array([[num_required, weighted_match_score]]))[0]

# Example usage
required_skills = ["Python", "Machine Learning", "SQL", "Flask", "Data Science"]

resume_score = predict_resume_score(matched_skills, job_description_skills)
print(f"Resume Score: {resume_score:.2f}/100")


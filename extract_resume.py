import spacy
import pdfplumber
nlp = spacy.load("en_core_web_trf")
# text = ""
# with pdfplumber.open("sample_resume.pdf") as pdf:
#     for p in pdf.pages:
#         text += p.extract_text() + "\n"
# doc = nlp(text)

# skills = []
# job_titles = []

# for ent in doc.ents:
#     if ent.label_ in ["TITLE", "NORP"]:  
#         job_titles.append(ent.text)
#     elif ent.label_ not in ["ORG", "GPE", "PERSON", "DATE", "CARDINAL"]: 
#         skills.append(ent.text)

# print("Skills:", set(skills))
# print("Job Titles:", set(job_titles))


import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_trf")
import pdfplumber
text = ""
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
        if ent.label_ in ["ORG", "GPE"]:  # Ignore organizations & locations
            continue
        if ent.label_ in ["PERSON", "DATE", "CARDINAL"]:  # Ignore names/dates
            continue
        if ent.label_ in ["WORK_OF_ART"]:  
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

extract_skills_and_titles(text)
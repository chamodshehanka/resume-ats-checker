import os
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text (remove stop words, punctuation, etc.)
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_text = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_text)

# Function to convert PDF to plain text
def pdf_to_text(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText()
    return text

# Function to calculate the similarity score between resume and job description
def calculate_ats_score(resume_text, job_description_text):
    # Preprocess the texts
    resume_processed = preprocess(resume_text)
    job_description_processed = preprocess(job_description_text)
    
    # Create a CountVectorizer object
    vectorizer = CountVectorizer().fit_transform([resume_processed, job_description_processed])
    vectors = vectorizer.toarray()
    
    # Compute the cosine similarity between the two vectors
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]  # Similarity between resume and job description

def main():
    # Paths to resume PDF and job description text file
    resume_path = 'resume.pdf'
    job_description_path = 'job_description.txt'

    # Convert PDF resume to plain text
    resume_text = pdf_to_text(resume_path)

    # Read job description from text file
    with open(job_description_path, 'r') as file:
        job_description_text = file.read()

    # Calculate ATS score
    ats_score = calculate_ats_score(resume_text, job_description_text)
    print(f'ATS Score: {ats_score:.2f}')

if __name__ == "__main__":
    main()

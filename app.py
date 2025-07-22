import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Sample job data
job_data = pd.DataFrame({
    'Job Title': [
        'Software Engineer',
        'Data Scientist',
        'Project Manager',
        'Web Developer',
        'Data Analyst'
    ],
    'Description': [
        'Develop and maintain software applications',
        'Analyze data and build machine learning models',
        'Manage project timelines and resources',
        'Build and maintain websites',
        'Analyze data and create reports'
    ],
    'Skills': [
        'Python, Java, SQL',
        'Python, Machine Learning, SQL',
        'Communication, Leadership, Project Management',
        'HTML, CSS, JavaScript',
        'SQL, Excel, Data Visualization'
    ]
})

# ✅ Job recommendation function
def recommend_jobs(user_skills, job_data):
    if not user_skills.strip():
        return None, None

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(job_data['Skills'])
    user_vector = tfidf_vectorizer.transform([user_skills])
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    related_job_indices = cosine_similarities.argsort()[::-1][:3]
    return job_data.iloc[related_job_indices], cosine_similarities[related_job_indices]

# ✅ Streamlit app
st.title("🔍 Job Recommendation System")
st.write("Enter your skills to get job recommendations:")

user_skills = st.text_input("Your Skills (comma-separated):").strip()

if user_skills:
    recommendations, scores = recommend_jobs(user_skills, job_data)

    if recommendations is not None:
        st.write("## 📝 Recommended Jobs:")
        for i, (index, row) in enumerate(recommendations.iterrows()):
            st.write(f"### {i+1}. {row['Job Title']}")
            st.write(f"**Description:** {row['Description']}")
            st.write(f"**Skills:** {row['Skills']}")
            st.write(f"**Match Score:** {scores[i]:.2f}")
            st.write("---")
    else:
        st.warning("Please enter at least one valid skill.")

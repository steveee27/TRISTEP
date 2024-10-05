import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import streamlit as st
import re
import gdown
st.set_page_config(page_title="TriStep - Career and Learning Recommendation System", page_icon="🚀", layout="wide")

def preprocess_text_simple(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\*+', '', text)
    return text

def remove_asterisks(text):
    if pd.isna(text):
        return text
    return re.sub(r'\*+', '', text)

def recommend_job(user_input, df, vectorizer, tfidf_matrix, experience_levels=None, work_types=None, name=None, country=None):
    # Jangan hitung ulang tfidf_matrix di sini
    user_input_processed = preprocess_text_simple(user_input)
    user_tfidf = vectorizer.transform([user_input_processed])
    
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Filter recommendations with cosine similarity > 0 and sort
    above_zero = cosine_similarities > 0
    if not any(above_zero):
        return None

    top_job_indices = np.where(above_zero)[0]
    
    top_job_indices = top_job_indices[np.argsort(cosine_similarities[top_job_indices])[::-1]]
    
    top_jobs = df.iloc[top_job_indices].copy()
    top_jobs['cosine_similarity'] = cosine_similarities[top_job_indices]
    
    # Apply filters after cosine similarity calculation
    if experience_levels:
        top_jobs = top_jobs[top_jobs['formatted_experience_level'].isin(experience_levels)]
    if work_types:
        top_jobs = top_jobs[top_jobs['formatted_work_type'].isin(work_types)]
    if name and name != 'All':
        top_jobs = top_jobs[top_jobs['name'] == name]
    if country and country != 'All':
        top_jobs = top_jobs[top_jobs['country'] == country]
    
    if top_jobs.empty:
        return None

    top_jobs.reset_index(drop=True, inplace=True)
    
    return top_jobs


def recommend_course(user_input, df, vectorizer, tfidf_matrix, selected_sites=None, selected_categories=None, selected_subtitle=None):
    user_input_processed = preprocess_text_simple(user_input)
    user_tfidf = vectorizer.transform([user_input_processed])
    
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    above_zero = cosine_similarities > 0
    if not any(above_zero):
        return None

    threshold = np.percentile(cosine_similarities[above_zero], 95)

    above_threshold = cosine_similarities >= threshold
    top_course_indices = np.where(above_threshold)[0]

    if len(top_course_indices) == 0:
        return None

    top_course_indices = top_course_indices[np.argsort(cosine_similarities[top_course_indices])[::-1]]

    top_courses = df.iloc[top_course_indices].copy()
    top_courses['cosine_similarity'] = cosine_similarities[top_course_indices]

    # Apply filters after cosine similarity calculation
    if selected_sites:
        top_courses = top_courses[top_courses['Site'].isin(selected_sites)]
    if selected_categories:
        top_courses = top_courses[top_courses['Category'].isin(selected_categories)]
    if selected_subtitle and selected_subtitle != 'All':
        top_courses = top_courses[top_courses['Subtitle Languages'].str.contains(selected_subtitle, na=False)]

    if top_courses.empty:
        return None

    top_courses.reset_index(drop=True, inplace=True)
    return top_courses
@st.cache_data
def load_job_data():
    csv_url = 'https://docs.google.com/spreadsheets/d/1huKbxP4W5c5sBWAQ5LzerhdId6TR9glCRFKn7DNOKEE/export?format=csv&gid=1980208131'
    df_job = pd.read_csv(csv_url, on_bad_lines='skip', engine='python')
    
    # Drop duplicates based on 'description_x'
    df_job = df_job.drop_duplicates(subset=['description_x'])
    
    df_job['Combined'] = df_job['title'].fillna('') + ' ' + df_job['description_x'].fillna('') + ' ' + df_job['skills_desc'].fillna('')
    df_job['Combined'] = df_job['Combined'].apply(preprocess_text_simple)
    df_job['title'] = df_job['title'].apply(remove_asterisks)
    vectorizer_job = TfidfVectorizer(stop_words='english')
    tfidf_matrix_job = vectorizer_job.fit_transform(df_job['Combined'])
    return df_job, vectorizer_job, tfidf_matrix_job

@st.cache_data
def load_course_data():
    csv_url = 'https://docs.google.com/spreadsheets/d/1PM_ifqhHQbvVau26xH2rU7xEw8ib1t2D6s_eDRPzJVI/export?format=csv&gid=2031125993'
    df_course = pd.read_csv(csv_url)
    
    df_course = df_course.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df_course = df_course.drop_duplicates(subset=['Title', 'Short Intro'])
    translations = {
        '计算机科学': 'Computer Science',
        'Ciencia de Datos': 'Data Science',
        'Negocios': 'Business',
        'Ciencias de la Computación': 'Computer Science',
        'Negócios': 'Business',
        'データサイエンス': 'Data Science',
        'Tecnologia da informação': 'Information Technology'
    }
    df_course['Category'] = df_course['Category'].replace(translations)
    df_course['Rating'] = df_course['Rating'].str.replace('stars', '', regex=False)
    df_course['Number of viewers'] = df_course['Number of viewers'].str.replace(r'\D+', '', regex=True)
    df_course['combined'] = df_course['Title'] + ' ' + df_course['Short Intro'].fillna('') + ' ' + df_course['Skills'].fillna('') + ' ' + df_course['Category'].fillna('') + ' ' + df_course['Sub-Category'].fillna('')
    df_course['combined'] = df_course['combined'].apply(preprocess_text_simple)
    df_course = df_course.fillna('Unknown')
    df_course['Number of viewers'] = pd.to_numeric(df_course['Number of viewers'], errors='coerce').fillna(0).astype(int)
    df_course['Rating'] = pd.to_numeric(df_course['Rating'], errors='coerce').fillna(0)
    df_course['Subtitle Languages'] = df_course['Subtitle Languages'].str.replace('Subtitles: ', '', regex=False)
    
    keywords = ['Participant', 'Designed', 'Learners', 'prior', 'experience', 'natural', 'space', 'aeronautics']
    
    def remove_keywords(text, keywords):
        if pd.isna(text):
            return np.nan
        if any(keyword in text for keyword in keywords):
            return np.nan
        return text
    
    df_course['Subtitle Languages'] = df_course['Subtitle Languages'].apply(lambda x: remove_keywords(x, keywords))
    
    vectorizer_course = TfidfVectorizer(stop_words='english')
    tfidf_matrix_course = vectorizer_course.fit_transform(df_course['combined'])
    return df_course, vectorizer_course, tfidf_matrix_course
    
@st.cache_data
def download_images():
    url1 = 'https://drive.google.com/uc?id=1lhfFczKatGDEuq3ux2y-AqfPpVC96UZ9'
    output1 = 'Minimalist_Black_and_White_Blank_Paper_Document_1.png'
    gdown.download(url1, output1, quiet=False)

    url2 = 'https://drive.google.com/uc?id=1hbpQIE7Ez0Z4k1Sfq8FSO80_5HRujdjP'
    output2 = 'nobg2.png'
    gdown.download(url2, output2, quiet=False)

    return output1, output2

df_job, vectorizer_job, tfidf_matrix_job = load_job_data()
df_job.fillna("Unknown", inplace=True)
df_course, vectorizer_course, tfidf_matrix_course = load_course_data()

image1_path, image2_path = download_images()

st.markdown(
    """
    <style>
    .main {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #4169e1;
        color: white;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #1e90ff;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .stButton.get-recommendations button {
        font-size: 18px !important;
        padding: 10px 20px !important;
        width: auto !important;
        height: auto !important;
    }
    .stButton.navigation button {
        font-size: 14px !important;
        width: 100% !important;
        height: 40px !important;
        white-space: nowrap !important;
    }
    .logo-link {
    cursor: pointer;
    text-align: center;
    display: block;
    }
    .logo-link:hover {
        opacity: 0.8;
    }
    .st-expander {
        border: 1px solid var(--secondary-background-color);
        padding: 10px;
        border-radius: 5px;
        background-color: var(--background-color);
    }
    .st-expander p {
        margin: 0;
        color: var(--text-color);
    }
    .search-box {
        width: 100%;
        padding: 10px;
        margin-top: 10px;
        border: 2px solid #4169e1;
        border-radius: 5px;
    }
    .stButton.get-recommendations {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: var(--text-color);
    }
    .stSidebar {
        background-color: #BEBEBE; 
        padding: 20px;
    }
    
    .stSidebar [data-testid="stSidebarNav"] > ul {
        padding-top: 20px;
    }
    
    .stSidebar [data-testid="stSidebarNav"] > ul > li:first-child {
        font-size: 24px;
        font-weight: bold;
        color: black;
    }
    
    .stSidebar [data-testid="stSidebarNav"] ul {
        color: black;
    }
    
    .stSidebar .stRadio > label {
        font-size: 18px !important;
        margin-bottom: 15px !important;
        color: black;
    }
    
    .stSidebar .stRadio > div {
        margin-bottom: 20px !important;
    }
    
    .stSidebar [data-testid="stMarkdownContainer"] > p {
        font-size: 16px !important;
        line-height: 1.5 !important;
        color: black;
    }
    
    .stSidebar * {
        color: black !important;
    }
    .main .stTextInput > div > div > input {
        font-size: 16px !important;
        color: var(--text-color);
    }
    .main .stSelectbox > div > div > div {
        font-size: 16px !important;
        color: var(--text-color);
    }
    .main .stCheckbox > label {
        font-size: 16px !important;
        color: var(--text-color);
    }
    .section {
        background-color: var(--background-color);
        padding: 0px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        color: var(--text-color);
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: var(--background-color);
        color: var(--text-color);
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid var(--secondary-background-color);
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'

st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")
st.sidebar.image(image1_path, use_column_width=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", ('🏢 Home', '📊 Step 1: Explore', '💼 Step 2: Find', '📚 Step 3: Grow'))
st.sidebar.markdown("---")
st.sidebar.markdown("© 2024 TriStep 🚀")
st.sidebar.markdown("Created By M-Tree")

if 'previous_page' not in st.session_state:
    st.session_state.previous_page = None

current_page = page

if current_page != st.session_state.previous_page:
    if 'job_recommendations' in st.session_state:
        st.session_state.job_recommendations = None
        st.session_state.job_page = 0
    if 'course_recommendations' in st.session_state:
        st.session_state.course_recommendations = None
        st.session_state.course_page = 0

st.session_state.previous_page = current_page

if page == '🏢 Home':
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')

    with col2:
        st.image(image2_path)

    with col3:
        st.write(' ')
    st.title("🏢 About TriStep")
   

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Welcome to TriStep")
    st.write(
        "TriStep is your premier platform for professional growth and talent development. "
        "We're dedicated to empowering individuals with the tools and resources needed to "
        "enhance their skills, expand their knowledge, and unlock their full potential in their career journeys."
    )
    
    
    st.write(
        "The name 'TriStep' embodies our core philosophy of growth through three essential steps:"
        "\n\n1. **Explore** current job market trends and opportunities through our interactive dashboard."
        "\n2. **Find** jobs that align with your career goals and identify skill requirements."
        "\n3. **Grow** access tailored courses and resources to develop the skills needed for your desired career path."
        "\n\nBy following these three steps, you can transform into a more talented, skilled, and valuable professional."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Our Mission")
    st.write(
        "At TriStep, we believe in creating a comprehensive talent growth ecosystem. "
        "Our platform allows you to explore current job and company trends, search for "
        "tailored learning opportunities, and access a wide range of courses designed to "
        "help you develop the skills needed for your dream career path."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("How to Use TriStep")
    st.subheader("1. Explore Trends on the Dashboard")
    st.write(
        "- Browse the latest job and company trends on your dashboard.\n"
        "- Stay updated with industry insights and hot topics in the job market.\n"
        "- Use this information to guide your skill development and career planning."
    )

    st.subheader("2. Job Insights")
    st.write(
    "- Navigate to the 'Find' section for insights into current job opportunities.\n"
    "- Input your areas of interest and skills in the search bar.\n"
    "- Use filters like experience level and work type to refine your search.\n"
    "- Explore detailed job descriptions to understand market demands and skill requirements."
    )

    st.subheader("3. Course Recommendations")
    st.write(
    "- Visit the 'Grow' section to find skill-enhancing courses.\n"
    "- Enter your learning interests or desired skills.\n"
    "- Filter courses by site, subtitle language, and more.\n"
    "- Get personalized course recommendations to boost your talents and career prospects."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.header("Contact Us")
    st.write(
        "Have questions or feedback? We'd love to hear from you!\n\n"
        "📧 Email: TriStepcompany@gmail.com"
    )

    st.write(
        "Join TriStep today and take the next step towards becoming a more talented and valuable professional. "
        "Your journey to personal and career growth starts here!"
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif page == '📊 Step 1: Explore':
    st.title("📊 Explore the Latest Job Trends")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image(image2_path)
    with col3:
        st.write(' ')
    html_string = """
        <div class='tableauPlaceholder' id='viz1727850105151' style='position: relative'>
          <noscript>
            <a href='#'>
              <img alt='Dashboard' src='https://public.tableau.com/static/images/Jo/JobMarket_Dashboard/Dashboard/1_rss.png' style='border: none' />
            </a>
          </noscript>
          <object class='tableauViz' style='display:none;'>
            <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
            <param name='embed_code_version' value='3' />
            <param name='site_root' value='' />
            <param name='name' value='JobMarket_Dashboard/Dashboard' />
            <param name='tabs' value='no' />
            <param name='toolbar' value='yes' />
            <param name='static_image' value='https://public.tableau.com/static/images/Jo/JobMarket_Dashboard/Dashboard/1.png' />
            <param name='animate_transition' value='yes' />
            <param name='display_static_image' value='yes' />
            <param name='display_spinner' value='yes' />
            <param name='display_overlay' value='yes' />
            <param name='display_count' value='yes' />
            <param name='language' value='en-US' />
          </object>
        </div>
        
        <script type='text/javascript'>
          var divElement = document.getElementById('viz1727850105151');
          var vizElement = divElement.getElementsByTagName('object')[0];
          
          if (divElement.offsetWidth > 800) {
            vizElement.style.width = '900px';
            vizElement.style.height = '1827px';
          } else if (divElement.offsetWidth > 500) {
            vizElement.style.width = '900px';
            vizElement.style.height = '1827px';
          } else {
            vizElement.style.width = '100%';
            vizElement.style.height = '3877px';
          }
        
          var scriptElement = document.createElement('script');
          scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
          vizElement.parentNode.insertBefore(scriptElement, vizElement);
        </script>

    """
    st.components.v1.html(html_string, width=900, height=1827)

elif page == '💼 Step 2: Find':
    st.title("💼 Find the Perfect Job for You")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image(image2_path)
    with col3:
        st.write(' ')
    st.subheader('🎚️ Experience Level')
    experience_levels = [level for level in df_job['formatted_experience_level'].unique().tolist() if level != "Unknown"]
    selected_experience_levels = []
    cols = st.columns(2)
    for i, exp in enumerate(experience_levels):
        with cols[i % 2]:
            if st.checkbox(exp, key=f"exp_{exp}"):
                selected_experience_levels.append(exp)
                
    st.subheader('🏢 Work Type')
    work_types = [wt for wt in df_job['formatted_work_type'].unique().tolist() if wt != "Unknown" and wt != "Other"]
    selected_work_types = []
    cols = st.columns(2)
    for i, work in enumerate(work_types):
        with cols[i % 2]:
            if st.checkbox(work, key=f"work_{work}"):
                selected_work_types.append(work)

    st.subheader('📍 Location')
    unique_countries = ['All'] + sorted([country for country in df_job['country'].unique().tolist() if country != "Unknown"])
    selected_country = st.selectbox('Choose a country', unique_countries)

    st.subheader('🔍 Company Name')
    unique_companies = ['All'] + sorted([company for company in df_job['name'].unique().tolist() if company != "Unknown"])
    name = st.selectbox('Select a company', unique_companies)

    user_input = st.text_area(
    "🧑‍💼 Prompt your career profile (e.g., education background, key skills, project experience, certifications, and interests)", 
    height=150,
    help="For better recommendations, provide detailed information such as:\n\n 'I am a Programmer with experience in designing, developing, testing, and maintaining complex applications. I have a strong understanding of industry best practices, and I'm proficient in writing clean, efficient code. I have experience collaborating with cross-functional teams, working with external APIs, and ensuring optimal application performance and quality. I am also skilled in identifying and resolving bottlenecks and bugs to maintain high code quality.'")
    
    if st.button("🚀 Get Job Insights", key="get_job_recommendations"):
        recommendations = recommend_job(
            user_input,
            df_job,
            vectorizer_job,
            tfidf_matrix_job,
            selected_experience_levels if selected_experience_levels else None,
            selected_work_types if selected_work_types else None,
            name if name != 'All' else None,
            selected_country if selected_country != 'All' else None
        )
        if recommendations is None or recommendations.empty:
            st.error("😕 No relevant jobs found matching your criteria. Please try adjusting your filters or providing more details in your career profile.")
            st.session_state.job_recommendations = None
            st.session_state.job_page = 0
        else:
            st.session_state.job_recommendations = recommendations
            st.session_state.job_page = 0

    if 'job_recommendations' in st.session_state and st.session_state.job_recommendations is not None:
        recommendations = st.session_state.job_recommendations
        page = st.session_state.job_page
        items_per_page = 5
        start_index = page * items_per_page
        end_index = start_index + items_per_page
    
        st.write("### 🎯 Here Are The Most Suitable Jobs For You:")
        # Adjust the job description display to be justified
        for i, (_, row) in enumerate(recommendations.iloc[start_index:end_index].iterrows(), start=start_index + 1):
            st.markdown(f"#### {i}. {row['title']}")
            st.markdown(f"🏢 Company Name: {row['name']}")
            st.markdown(f"📍 Country: {row['country']}")
            st.markdown(f"📍 City: {row['city']}")
            st.markdown(f"[🔗 View Job Posting]({row['job_posting_url']})")
            with st.expander("📄 More Info"):
                # Applying justify alignment to the description
                st.markdown(
                    f"<p style='text-align: justify;'>{row['description_x']}</p>", 
                    unsafe_allow_html=True
                )
                if row['min_salary'] == 'Unknown':
                    st.markdown(f"💰 Min Salary (Yearly): {row['min_salary']}")
                else:
                    st.markdown(f"💰 Min Salary (Yearly): Rp{row['min_salary']}")
                if row['max_salary'] == 'Unknown':
                    st.markdown(f"💵 Max Salary (Yearly): {row['max_salary']}")
                else:
                    st.markdown(f"💵 Max Salary (Yearly): Rp{row['max_salary']}")
                st.markdown(f"🕒 Work Type: {row['formatted_work_type']}")
                st.markdown(f"🎓 Experience Level: {row['formatted_experience_level']}")
            st.markdown("---")

    
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            if start_index > 0:
                if st.button("⬅️ Prev", key='job_previous'):
                    st.session_state.job_page -= 1
        with col3:
            if end_index < len(recommendations):
                if st.button("Next ➡️", key='job_next'):
                    st.session_state.job_page += 1

     # Add the new button for adding a job
    st.markdown("""
        <a href="https://docs.google.com/forms/d/e/1FAIpQLSfXOzq3CDsGvMu9UXZeq_6z9d1-QrT3KHSW5R3WPFHlRDDqVw/viewform" target="_blank">
            <button style="
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
                transition: background-color 0.3s;
            ">
                ➕ Contribute to Job Data
            </button>
        </a>
    """, unsafe_allow_html=True)
                
elif page == '📚 Step 3: Grow':
    st.title('📚 Grow Through Course Choices')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image(image2_path)
    with col3:
        st.write(' ')

    st.subheader('🌐 Sites')
    sites = [site for site in df_course['Site'].unique().tolist() if site != "Unknown"]
    selected_sites = []
    cols = st.columns(2)
    for i, site in enumerate(sites):
        with cols[i % 2]:
            if st.checkbox(site, key=f"site_{site}"):
                selected_sites.append(site)

    st.subheader('📊 Categories')
    categories = [cat for cat in df_course['Category'].unique().tolist() if cat != "Unknown"]
    selected_categories = []
    cols = st.columns(2)
    for i, cat in enumerate(categories):
        with cols[i % 2]:
            if st.checkbox(cat, key=f"cat_{cat}"):
                selected_categories.append(cat)

    st.subheader('🗣️ Subtitle Language')
    unique_subtitles = ['All'] + sorted(set([lang.strip() for sublist in df_course['Subtitle Languages'].dropna().str.split(',') for lang in sublist if lang.strip() != 'Unknown']))
    selected_subtitle = st.selectbox('Choose a language', unique_subtitles)

    user_input = st.text_area(
        "🔍 Prompt skills or topics you'd like to learn:", 
        height=150,
        help="For better recommendations, provide topic or job desk from the company, such as:\n\n 'The job responsibilities I want to gain experience in include Data Engineering, Big Data Technologies, Data Transformation, and Data Modelling.'"
    )

    if st.button("🚀 Get Course Recommendations", key="get_course_recommendations"):
        recommendations = recommend_course(
            user_input, 
            df_course, 
            vectorizer_course, 
            tfidf_matrix_course,
            selected_sites if selected_sites else None,
            selected_categories if selected_categories else None,
            selected_subtitle if selected_subtitle != 'All' else None
        )
        if recommendations is None or recommendations.empty:
            st.error("😕 No relevant courses found matching your criteria. Please try adjusting your filters or providing more details in your learning interests.")
            st.session_state.course_recommendations = None
            st.session_state.course_page = 0
        else:
            st.session_state.course_recommendations = recommendations
            st.session_state.course_page = 0

    if 'course_recommendations' in st.session_state and st.session_state.course_recommendations is not None:
        recommendations = st.session_state.course_recommendations
        page = st.session_state.course_page
        items_per_page = 5
        start_index = page * items_per_page
        end_index = start_index + items_per_page

        st.write("### 🎯 Here Are The Most Suitable Courses For You:")
        for i, (_, row) in enumerate(recommendations.iloc[start_index:end_index].iterrows(), start=start_index + 1):
            st.markdown(f"#### {i}. {row['Title']}")
            st.markdown(f"📊 Category: {row['Category']}")
            st.markdown(f"📑 Sub-Category: {row['Sub-Category']}")
            st.markdown(f"🌐 Site: {row['Site']}")
            st.markdown(f"[🔗 View Course]({row['URL']})")
            with st.expander("📄 More Info"):
                st.markdown(f"📝 Short Intro: {row['Short Intro']}")
                st.markdown(f"⭐ Rating: {row['Rating']}")
                st.markdown(f"👥 Number of Viewers: {int(row['Number of viewers'])}")
                st.markdown(f"🗣️ Language: {row['Language']}")
                st.markdown(f"🔠 Subtitle Languages: {row['Subtitle Languages']}")
            st.markdown("---")

        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            if start_index > 0:
                if st.button("⬅️ Prev", key='course_previous'):
                    st.session_state.course_page -= 1
        with col3:
            if end_index < len(recommendations):
                if st.button("Next ➡️", key='course_next'):
                    st.session_state.course_page += 1

    # Add the new button for adding your own course
    st.markdown("""
        <a href="https://docs.google.com/forms/d/e/1FAIpQLSedcvWeMfGsXcaoLO7lZ1MKi_EZq8fAimxlH7sxZAKZgNcslQ/viewform" target="_blank">
            <button style="
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
                transition: background-color 0.3s;
            ">
                ➕ Contribute to Course Data
            </button>
        </a>
    """, unsafe_allow_html=True)
if __name__ == "__main__":
    pass

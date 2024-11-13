# TRISTEP: AI-Driven Talent Development Platform Empowering Indonesia Through Three Simple Steps

TRISTEP is an innovative AI-powered platform designed to bridge the gap between the supply and demand of digital talent in Indonesia. The platform helps users explore industry trends, find suitable job opportunities, and grow their skills through relevant online courses. By leveraging advanced vectorization techniques like TF-IDF, WORD2VEC, and BERT, TRISTEP delivers accurate and efficient recommendations tailored to individual user preferences.

## 🌐 Try TRISTEP Yourself
Experience the TRISTEP platform firsthand by visiting: [Access TRISTEP Platform](https://tristep.streamlit.app/)

For a guided tour on how to use the platform, check out our live demo video: [Watch the Demo](https://youtu.be/b5qtDCex9Ts?si=R6DS-VD32_0Zo84I)

> **Note:** An admin website is also available for managing user-generated data. When users add job or course information, admins can verify these submissions through the admin site: [TRISTEP Admin](https://tristep-adm1n.streamlit.app/)

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Results](#results)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [FAQ](#FAQ)
- [Contributors](#contributors)
- [License](#license)

## Introduction

TRISTEP addresses the growing need for digital talent by offering a structured approach to career development. The platform consists of three main steps:

1. **Explore:** Users can explore current industry trends and identify job opportunities that align with their career goals.
2. **Find:** TRISTEP provides personalized job recommendations based on user preferences, including education, skills, and experience.
3. **Grow:** Users receive recommendations for online courses to enhance their skills, ensuring they remain competitive in the job market.

## Datasets

TRISTEP utilizes two primary datasets:

1. **[Job Postings LinkedIn](https://www.kaggle.com/code/enricofindley/linkedin-job-postings-2023-data-analysis), [Job Postings Jobstreet](https://www.kaggle.com/datasets/azizainunnajib/jobs-crawling):** Contains 49,194 job postings from LinkedIn & Jobstreet including attributes such as job title, description, salary, location, and company details.
2. **[Online Courses](https://www.kaggle.com/code/enricofindley/linkedin-job-postings-2023-data-analysis):** Contains 8,092 online courses from platforms like Coursera, Udacity, and others, with detailed information such as course title, description, rating, and more.

These datasets are used to train and evaluate the recommendation models that power the TRISTEP platform.

## Methodology

TRISTEP uses three vectorization methods to generate recommendations:

- **TF-IDF (Term Frequency-Inverse Document Frequency):** Captures the importance of words in job descriptions and course details relative to the entire dataset.
- **WORD2VEC:** Creates high-dimensional vector representations of words, capturing semantic similarities.
- **BERT (Bidirectional Encoder Representations from Transformers):** A sophisticated model that understands the context of words in both directions within a sentence.

### Data Preprocessing

Data preprocessing involves cleaning and structuring both datasets to prepare them for vectorization. This includes tasks such as:

- Removing duplicates
- Imputing missing values
- Normalizing text fields
- Tokenizing and vectorizing text data

### Model Training

The recommendation models were trained using Kaggle Notebooks equipped with two GPU T4s, ensuring efficient computation. The models were evaluated based on `Prec@K`, focusing on how well the top `K` recommendations matched user preferences.

## Results

| Method   | Prec@5 (Jobs) | Prec@5 (Courses) | Processing Time (s) |
|----------|----------------|-------------------|---------------------|
| TF-IDF   | 0.900          | 0.860             | Moderate            |
| WORD2VEC | 0.532          | 0.304             | Fast (Jobs), Slow (Courses) |
| BERT     | 0.440          | 0.668             | Slow                |

TF-IDF emerged as the most balanced method, offering high accuracy with reasonable processing times, making it the preferred choice for the TRISTEP platform.



## System Requirements

- Python 3.8 or higher
- Kaggle Notebook with two GPU T4s (optional for local development)
- [Required libraries](https://github.com/steveee27/TRISTEP/blob/main/requirements.txt): scikit-learn, pandas, numpy, streamlit, gdown, psutil

## Installation

Clone the repository and install the required libraries:
```
git clone https://github.com/steveee27/TRISTEP.git
cd TRISTEP
pip install -r requirements.txt
```

## Usage

To run the Streamlit application locally:
```
cd streamlit_app
streamlit run app.py
```
Access the TRISTEP platform online at: [TRISTEP Live](https://tristep.streamlit.app/)

## FAQ

Why am I getting an error when trying to run my Streamlit app?

Your issue is most likely happening because Streamlit needs to be rebooted regularly. This can happen when the app runs out of memory or goes into sleep mode after being idle for a while, especially since we are not using the premium version. For assistance with rebooting, feel free to contact TriStep Company via email at TriStepcompany@gmail.com or through WhatsApp at 085106378743.

## Contributors
1. [Steve Marcello Liem](https://github.com/steveee27)
2. [Matthew Lefrandt](https://github.com/MatthewLefrandt)
3. [Marvel Martawidjaja](https://github.com/marvelm69)

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/steveee27/TRISTEP/blob/main/LICENSE) file for details.

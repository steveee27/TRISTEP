# TRISTEP: AI-Powered Talent Development Platform

TRISTEP is an innovative AI-powered platform designed to bridge the gap between the supply and demand of digital talent in Indonesia. The platform helps users explore industry trends, find suitable job opportunities, and grow their skills through relevant online courses. By leveraging advanced vectorization techniques like TF-IDF, WORD2VEC, and BERT, TRISTEP delivers accurate and efficient recommendations tailored to individual user preferences.

## 🚀 Live Demo

Check out the live demo of the TRISTEP platform here: [TRISTEP Live](https://tristep.streamlit.app/)

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Results](#results)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

## Introduction

TRISTEP addresses the growing need for digital talent by offering a structured approach to career development. The platform consists of three main steps:

1. **Explore:** Users can explore current industry trends and identify job opportunities that align with their career goals.
2. **Find:** TRISTEP provides personalized job recommendations based on user preferences, including education, skills, and experience.
3. **Grow:** Users receive recommendations for online courses to enhance their skills, ensuring they remain competitive in the job market.

## Datasets

TRISTEP utilizes two primary datasets:

1. **[Job Postings LinkedIn](https://www.kaggle.com/code/enricofindley/linkedin-job-postings-2023-data-analysis):** Contains 15,886 job postings from LinkedIn, including attributes such as job title, description, salary, location, and company details.
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

| Method   | Prec@10 (Jobs) | Prec@10 (Courses) | Processing Time (s) |
|----------|----------------|-------------------|---------------------|
| TF-IDF   | 0.846          | 0.836             | Moderate            |
| WORD2VEC | 0.462          | 0.296             | Fast (Jobs), Slow (Courses) |
| BERT     | 0.502          | 0.654             | Slow                |

TF-IDF emerged as the most balanced method, offering high accuracy with reasonable processing times, making it the preferred choice for the TRISTEP platform.

You can review the detailed evaluation and analysis [here](https://www.kaggle.com/code/enricofindley/linkedin-job-postings-2023-data-analysis).

## System Requirements

- Python 3.8 or higher
- Kaggle Notebook with two GPU T4s (optional for local development)
- Required libraries: TensorFlow, PyTorch, Scikit-learn, Pandas, Numpy, Streamlit

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

## Contributors
1. [Steve Marcello Liem](https://github.com/steveee27)
2. [Matthew Lefrandt](https://github.com/MatthewLefrandt)
3. [Marvel Martawidjaja](https://github.com/marvelm69)

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/steveee27/TRISTEP/blob/main/LICENSE) file for details.

# SkillBridge
## ML-Based Career Skill Gap Analysis and Personalized Learning Recommendation System

**Project Type:** BTech Computer Science and Engineering Microproject  
**Application Type:** Web Application  
**Primary Technology:** Python, Streamlit, Scikit-learn  
**Target Users:** Students, fresh graduates, and aspiring software developers

---

## 1. Project Overview

SkillBridge is a machine-learning-powered web application that helps students identify the technical skills they need to develop for their desired career roles.

Users enter their existing technical skills, and the application analyzes relevant job profiles to identify suitable career paths, missing skills, and learning priorities.

The system uses Natural Language Processing (NLP), TF-IDF vectorization, cosine similarity, and K-means clustering to analyze job descriptions and compare a student's skills with the skills associated with different career opportunities.

An interactive skill network visualizes relationships between technical skills, while a recommendation module suggests which skills users should learn first.

The goal is to make career preparation more data-driven, transparent, and personalized.

## 2. Problem Statement

Students often struggle to understand which technical skills are important for their desired career roles. Job descriptions contain large amounts of information, and manually comparing them with existing skills can be time-consuming.

Many existing career guidance platforms provide generic learning recommendations without clearly explaining the gap between a student's current skills and the requirements of relevant roles.

SkillBridge addresses this problem by analyzing job-related skill data, identifying skill gaps, ranking missing skills, and recommending practical learning resources.

## 3. Objectives

The primary objectives are:

1. Build an interactive web application for career skill analysis.
2. Extract and normalize technical skills from job descriptions.
3. Apply TF-IDF vectorization to represent job descriptions numerically.
4. Use K-means clustering to discover groups of similar job profiles.
5. Match student skill profiles with relevant career opportunities using cosine similarity.
6. Identify skills that students already possess and skills they are missing.
7. Rank missing skills according to their relevance and demand in the selected role.
8. Visualize relationships between technical skills using a network graph.
9. Recommend learning resources for prioritized missing skills.
10. Evaluate the implemented machine-learning components using appropriate metrics.
11. Provide an attractive dashboard with downloadable analysis results.

## 4. Target Users

- BTech and other undergraduate students.
- Fresh graduates preparing for software development jobs.
- Students exploring career paths in technology.
- Learners preparing for internships and entry-level positions.

## 5. Scope of the Project

### Included Features

- Manual entry of technical skills.
- Selection of career interests.
- Job-description dataset loading and preprocessing.
- Technical skill extraction and normalization.
- TF-IDF feature generation.
- K-means clustering of job profiles.
- Career matching using cosine similarity.
- Baseline matching using skill overlap.
- Skill-gap identification.
- Missing-skill prioritization.
- Learning resource recommendations.
- Interactive skill co-occurrence network.
- Model evaluation dashboard.
- CSV export of analysis results.
- Sample dataset for reliable demonstrations.
- Unit tests and project documentation.

### Excluded from the Initial Version

- Real-time scraping of job portals.
- Guaranteed employment predictions.
- Paid external AI APIs.
- Automatic applications to job vacancies.
- Production-grade user authentication.
- Claims that similarity scores represent the probability of getting hired.

These features may be considered future enhancements.

## 6. Functional Requirements

### FR-01: User Skill Profile

The application shall allow users to enter their current technical skills.

Examples include:
- Python
- Java
- SQL
- HTML
- CSS
- JavaScript
- React
- Machine Learning
- Data Analysis
- Git

The application shall normalize spelling, capitalization, and common skill aliases.

### FR-02: Career Interest Selection

Users shall be able to select or explore career categories, such as:

- Python Developer
- Full-Stack Developer
- Data Analyst
- Machine Learning Engineer
- Backend Developer
- Frontend Developer

Available roles must reflect the dataset. The application must not display unsupported roles as if they were discovered in the data.

### FR-03: Dataset Management

The system shall load job descriptions and related skill information from a local dataset.

The data-loading module shall:
- Validate required columns.
- Handle missing values.
- Remove duplicate records where appropriate.
- Normalize text.
- Report invalid or empty datasets clearly.
- Provide sample data for demonstration purposes.

The project must document the dataset's source, license, limitations, and preprocessing steps.

If genuine data cannot be obtained, a small synthetic dataset may be used for development and demonstrations. Synthetic records must be clearly labelled and must not be presented as real job postings.

### FR-04: Skill Extraction

The application shall identify technical skills from job descriptions using a configurable skill dictionary and alias mapping.

Example:

"Machine Learning" and "ML" may be mapped to the same normalized skill where appropriate.

The system shall keep skill extraction logic in a separate, reusable module.

### FR-05: Text Vectorization

The system shall use TF-IDF vectorization to convert job descriptions into numerical feature vectors.

Text preprocessing may include:
- Lowercasing.
- Removal of unnecessary punctuation.
- Whitespace normalization.
- Stop-word handling where appropriate.
- Consistent processing of missing text.

The fitted vectorizer shall be reusable for new inputs.

### FR-06: Job Profile Clustering

The system shall apply K-means clustering to group similar job descriptions.

The implementation shall:
- Allow the number of clusters to be configured.
- Avoid invalid cluster configurations.
- Report cluster sizes.
- Display representative skills or terms for each cluster.
- Use a suitable evaluation measure, such as silhouette score, when the dataset and clustering configuration permit it.

Clusters shall not automatically be treated as verified career labels. Their meaning must be interpreted from the data.

### FR-07: Career Matching

The system shall compare the user's skill profile with job profiles or career-role profiles.

The primary matching method shall use cosine similarity on suitable TF-IDF representations.

A transparent skill-overlap baseline shall also be implemented so the results can be compared with the primary method.

The interface shall explain why a role was recommended by showing relevant matching skills and missing skills.

Similarity scores shall be labelled clearly and shall not be described as employment probabilities.

### FR-08: Skill Gap Analysis

For a selected career role, the application shall identify:

- Skills the user already possesses.
- Skills required by the selected role according to the available data.
- Missing skills.
- Skills that appear frequently in relevant job descriptions.
- Suggested priorities for learning.

The analysis shall be based on the dataset rather than hard-coded claims about the entire job market.

### FR-09: Learning Recommendations

The application shall recommend learning resources for missing skills.

Recommendations may include documentation, tutorials, and reputable learning websites.

The recommendation logic shall be transparent. It should explain why a skill is prioritized, for example because it appears frequently in relevant job descriptions or is associated with multiple related skills.

The application shall not claim that a course is the objectively best resource unless that conclusion is supported by evidence.

Resource links shall be configurable and checked during project development.

### FR-10: Skill Network Visualization

The application shall visualize relationships between technical skills using an interactive graph.

Skills may be represented as nodes, with connections representing co-occurrence within job descriptions.

The graph shall support:
- Interactive exploration.
- Skill labels.
- Meaningful connections.
- Filtering or threshold controls where practical.
- A clear explanation of what connections represent.

A connection shall not be interpreted as proof that one skill causes the need for another.

### FR-11: Model Evaluation

The application shall provide an evaluation page showing appropriate metrics for implemented components.

Possible metrics include:

- Silhouette score for valid clustering configurations.
- Skill extraction precision, recall, and F1-score using manually reviewed examples.
- Precision@K or Recall@K for career recommendations when a defensible evaluation set is available.
- Comparison between cosine-similarity matching and the skill-overlap baseline.
- Dataset size and preprocessing statistics.

Evaluation results must be calculated from actual data and tests. The application must never fabricate accuracy scores or claim that an unvalidated system is highly accurate.

### FR-12: Result Export

Users shall be able to download relevant analysis results as CSV files.

Exports may include:
- Career match rankings.
- Existing skills.
- Missing skills.
- Skill priority rankings.
- Matching explanations.

The application shall handle empty results and export errors gracefully.

## 7. Non-Functional Requirements

### Usability
The interface should be simple, responsive, and understandable to students with basic technical knowledge.

### Performance
The application should run comfortably on a typical student laptop using a moderate-sized dataset.

### Maintainability
Code shall be divided into logical modules rather than implemented entirely in one file.

### Reliability
Invalid user inputs and missing dataset columns shall produce helpful messages rather than unhandled crashes.

### Reproducibility
Random states shall be configured where appropriate so experiments can be repeated.

### Privacy
The initial version should process user-entered skills locally and should not transmit personal information to external services.

### Portability
The application shall support a documented local setup on a compatible Python environment.

### Transparency
Recommendations and similarity scores shall be explained in understandable terms.

## 8. Proposed Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web interface | Streamlit |
| Data manipulation | Pandas, NumPy |
| Natural Language Processing | Scikit-learn text utilities |
| Feature extraction | TF-IDF Vectorizer |
| Machine learning | K-means clustering |
| Similarity calculation | Cosine similarity |
| Visualizations | Plotly |
| Network graph | PyVis or a compatible alternative |
| Model persistence | Joblib |
| Testing | Pytest |
| Dataset format | CSV |

Use compatible, documented dependency versions. The application should not require paid APIs.

## 9. System Architecture

The application shall follow a modular architecture.

**Layer 1 — Data Input**
- Local job-description dataset.
- User-entered skill profile.

**Layer 2 — Data Processing**
- Data validation.
- Text cleaning.
- Skill extraction.
- Skill normalization.

**Layer 3 — Feature Engineering**
- TF-IDF vectorization.
- Numerical skill representations.
- Skill co-occurrence calculation.

**Layer 4 — Machine Learning**
- K-means clustering.
- Cosine-similarity matching.
- Skill-overlap baseline.

**Layer 5 — Recommendation Engine**
- Career ranking.
- Missing-skill identification.
- Skill priority calculation.
- Learning resource mapping.

**Layer 6 — Presentation**
- Streamlit dashboard.
- Career match charts.
- Skill-gap visualization.
- Skill network.
- Evaluation metrics.
- CSV exports.

## 10. Proposed Application Pages

### Page 1: Overview Dashboard
Display:
- Project introduction.
- Dataset summary.
- Number of available job profiles.
- Number of recognized skills.
- Brief workflow explanation.

### Page 2: My Skill Profile
Allow users to:
- Enter existing skills.
- Select career interests.
- Review recognized skills.
- Correct unsupported or unrecognized entries.

### Page 3: Career Matches
Display:
- Ranked career matches.
- Similarity scores.
- Matching skills.
- Missing skills.
- Comparison with the baseline method.

### Page 4: Skill Gap Analysis
Display:
- Existing versus missing skills.
- Skill demand within the selected role's dataset records.
- Learning priorities.
- Suggested resources.
- Downloadable results.

### Page 5: Skill Network
Display an interactive network showing relationships between technical skills.

### Page 6: Model Evaluation
Display:
- Clustering metrics where valid.
- Skill extraction evaluation.
- Matching baseline comparison.
- Dataset statistics.
- Evaluation limitations.

### Page 7: About
Display:
- Project objectives.
- Technology stack.
- Dataset attribution.
- Methodology.
- Limitations.
- Instructions for running the project.

## 11. Proposed Project Structure

```text
skillbridge/
├── app.py
├── requirements.txt
├── README.md
├── PRD.md
├── .gitignore
│
├── data/
│   ├── README.md
│   └── sample_jobs.csv
│
├── models/
│   └── .gitkeep
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── skill_extractor.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── matching.py
│   ├── skill_gap.py
│   ├── recommendations.py
│   ├── skill_network.py
│   ├── evaluation.py
│   └── reporting.py
│
├── pages/
│   ├── 1_Profile.py
│   ├── 2_Career_Matches.py
│   ├── 3_Skill_Gap.py
│   ├── 4_Skill_Network.py
│   ├── 5_Model_Evaluation.py
│   └── 6_About.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_skill_extractor.py
│   ├── test_matching.py
│   ├── test_skill_gap.py
│   └── test_evaluation.py
│
└── reports/
    └── .gitkeep
```

The structure may be adjusted if necessary, but the code should remain modular and easy to understand.

## 12. Dataset Requirements

The dataset should contain job descriptions or role profiles with enough information to identify technical skills.

Potential fields include:
- Job title.
- Job description.
- Skills.
- Career category.
- Source identifier, if available.

The implementation team shall verify the chosen dataset's actual schema before writing the data-loading logic.

Dataset selection must consider:
- Availability.
- Licensing and permitted use.
- Data quality.
- Skill coverage.
- Relevance to the intended career categories.

The application should work with the included sample dataset even if the external dataset is unavailable.

The README must explain how to obtain, prepare, and place any external dataset in the expected directory.

## 13. Machine-Learning Methodology

### Step 1: Data Preparation
Load the dataset, validate its structure, handle missing values, remove inappropriate duplicates, and normalize text.

### Step 2: Skill Extraction
Extract recognized technical skills using a configurable dictionary and alias mapping.

### Step 3: Feature Engineering
Convert job-description text into TF-IDF feature vectors.

### Step 4: Clustering
Use K-means to group similar job descriptions and examine representative terms and skills.

### Step 5: Career Matching
Represent the student's skill profile and career profiles in a compatible feature space. Calculate cosine similarity to rank relevant profiles.

### Step 6: Baseline Comparison
Implement a simple skill-overlap measure to compare against the primary matching method.

### Step 7: Gap Identification
Compare the student's recognized skills with the skills associated with the selected career profile.

### Step 8: Prioritization
Rank missing skills using transparent measures, such as their frequency in relevant dataset records and their relationships with other skills.

### Step 9: Evaluation
Run appropriate tests and calculate metrics supported by the available evaluation data.

The project documentation must clearly distinguish unsupervised clustering, similarity-based ranking, rule-based processing, and recommendation logic.

## 14. Testing Requirements

The project shall include tests for:

- Dataset loading and schema validation.
- Handling of missing values.
- Text normalization.
- Skill alias mapping.
- Skill extraction.
- TF-IDF feature generation.
- Invalid clustering configurations.
- Career ranking.
- Skill-gap calculations.
- Recommendation ranking.
- Evaluation metric calculations.
- Empty or invalid user inputs.

The application shall be launched and manually checked before the final demonstration.

## 15. Success Criteria

The project will be considered successful when:

1. The application launches using documented instructions.
2. Users can enter technical skills and obtain career recommendations.
3. The system produces understandable matching explanations.
4. Missing skills are identified correctly against the selected profile data.
5. Skill recommendations have transparent priority explanations.
6. The skill network works with the available dataset.
7. Evaluation metrics are computed from real test data.
8. The user can export analysis results.
9. Automated tests pass, with any remaining limitations documented.
10. The README and project report accurately explain the implementation.

No unsupported performance or accuracy target shall be claimed.

## 16. Development Plan

### Phase 1: Project Setup
- Create the project structure.
- Set up a Python environment.
- Install dependencies.
- Inspect and document the dataset.

### Phase 2: Data Processing
- Implement data loading.
- Implement text preprocessing.
- Build skill extraction and alias mapping.
- Add unit tests.

### Phase 3: Machine-Learning Pipeline
- Implement TF-IDF feature engineering.
- Implement K-means clustering.
- Implement cosine-similarity matching.
- Implement the skill-overlap baseline.

### Phase 4: Recommendation Engine
- Implement skill-gap analysis.
- Implement transparent skill ranking.
- Map missing skills to learning resources.
- Build the skill co-occurrence network.

### Phase 5: Web Interface
- Build the dashboard and application pages.
- Add interactive visualizations.
- Add downloadable reports.
- Improve error handling and layout.

### Phase 6: Evaluation and Testing
- Calculate valid evaluation metrics.
- Compare matching approaches.
- Run automated tests.
- Test the application with different inputs.

### Phase 7: Documentation and Demonstration
- Finalize the README.
- Prepare the project report.
- Create an architecture diagram.
- Prepare a demonstration script.
- Prepare likely viva questions and answers.

## 17. Risks and Mitigation

**Risk: Poor or inconsistent dataset quality.**  
Mitigation: Validate the dataset, normalize skills, document limitations, and provide a clearly labelled sample dataset.

**Risk: Inaccurate skill extraction.**  
Mitigation: Use alias mapping, test representative examples, and report precision and recall when a labelled evaluation set is available.

**Risk: Misleading career matches.**  
Mitigation: Display explanations, compare against a baseline, and clearly label similarity scores.

**Risk: Unstable clustering results.**  
Mitigation: Configure random states, validate cluster counts, and report appropriate evaluation metrics.

**Risk: Missing external resources.**  
Mitigation: Keep learning resource mappings configurable and provide sensible fallback messages.

**Risk: Excessive project complexity.**  
Mitigation: Complete the core pipeline before adding optional features such as résumé parsing or advanced semantic embeddings.

## 18. Future Enhancements

Possible future improvements include:
- Résumé upload and skill extraction.
- Semantic embeddings for better understanding of related skills.
- Personalized learning roadmaps.
- Skill progression tracking.
- Additional career categories.
- Updated job datasets.
- Integration with external course catalogues.
- Improved evaluation using manually reviewed career profiles.

These enhancements are optional and should not delay the working core application.

## 19. Expected Deliverables

The completed project shall include:

1. Working SkillBridge web application.
2. Modular Python source code.
3. Dataset and data-source documentation.
4. Sample dataset for demonstrations.
5. Machine-learning pipeline.
6. Interactive skill network.
7. Evaluation dashboard.
8. Unit tests.
9. Requirements file.
10. README with setup and execution instructions.
11. Project report.
12. System architecture diagram.
13. Demonstration script.
14. Viva questions and answers.
15. Example exported analysis results.

## 20. Final Project Goal

The final application should demonstrate how machine learning and natural language processing can support career planning by turning job-related skill information into personalized, understandable, and actionable recommendations.

SkillBridge must prioritize a working, well-tested, explainable application over unnecessary complexity. Every claimed machine-learning result must be supported by the actual implementation and evaluation.
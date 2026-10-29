"""
config.py — Central configuration for SkillBridge.

All paths, ML hyperparameters, skill dictionary, alias mapping,
and learning resources are defined here so every other module
reads from a single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

SAMPLE_CSV = DATA_DIR / "sample_jobs.csv"
JOBS_CSV = DATA_DIR / "jobs.csv"          # Downloaded HuggingFace dataset
HF_PARQUET_URL = (
    "https://huggingface.co/datasets/batuhanmtl/job-skill-set/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)

# ---------------------------------------------------------------------------
# Required dataset columns
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["job_id", "category", "job_title", "job_description", "job_skill_set"]

# ---------------------------------------------------------------------------
# ML Hyperparameters
# ---------------------------------------------------------------------------
N_CLUSTERS = 6
TFIDF_MAX_FEATURES = 500
TFIDF_NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42
TOP_K_MATCHES = 10
TOP_K_TERMS = 10          # Top TF-IDF terms shown per cluster
COOCCURRENCE_THRESHOLD = 3  # Minimum co-occurrences to draw a network edge
MAX_NETWORK_NODES = 40    # Cap for performance

# ---------------------------------------------------------------------------
# Skill alias mapping (abbreviation → canonical form)
# Only unambiguous, context-independent aliases are included.
# ---------------------------------------------------------------------------
SKILL_ALIASES: dict[str, str] = {
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "k8s": "kubernetes",
    "tf": "tensorflow",
    "pt": "pytorch",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "sci-kit learn": "scikit-learn",
    "node": "node.js",
    "nodejs": "node.js",
    "node js": "node.js",
    "postgres": "postgresql",
    "pg": "postgresql",
    "mongo": "mongodb",
    "es": "elasticsearch",
    "gcp": "google cloud platform",
    "amazon web services": "aws",
    "restful": "restful api",
    "rest api": "restful api",
    "rest": "restful api",
    "oop": "object-oriented programming",
    "ci cd": "ci/cd",
    "cicd": "ci/cd",
    "powerbi": "power bi",
    "power-bi": "power bi",
    "tableau public": "tableau",
    "react.js": "react",
    "reactjs": "react",
    "react js": "react",
    "vue.js": "vue",
    "vuejs": "vue",
    "angular.js": "angular",
    "angularjs": "angular",
    "hadoop mapreduce": "hadoop",
    "apache spark": "spark",
    "apache kafka": "kafka",
    "apache airflow": "airflow",
    "c sharp": "c#",
    "c plus plus": "c++",
    "golang": "go",
    "rustlang": "rust",
}

# ---------------------------------------------------------------------------
# Master skill dictionary (canonical lowercase forms)
# ---------------------------------------------------------------------------
SKILL_DICTIONARY: set[str] = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "scala", "r", "swift", "kotlin", "php", "ruby", "perl", "matlab",
    # Web front-end
    "html", "css", "react", "angular", "vue", "next.js", "nuxt.js",
    "svelte", "sass", "less", "webpack", "vite", "tailwind css", "bootstrap",
    # Web back-end
    "node.js", "django", "flask", "fastapi", "spring boot", "express.js",
    "laravel", "rails", "asp.net", "graphql", "restful api",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "sqlite", "oracle", "cassandra", "dynamodb", "firebase",
    # Cloud & DevOps
    "aws", "azure", "google cloud platform", "docker",
    "kubernetes", "terraform", "ansible", "ci/cd", "jenkins", "github actions",
    "linux", "bash", "git",
    # Data & ML
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "scikit-learn", "keras",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    "data analysis", "data visualization", "data wrangling",
    "feature engineering", "model deployment", "mlops",
    # Big Data
    "spark", "hadoop", "kafka", "airflow", "hive", "databricks",
    # BI Tools
    "tableau", "power bi", "looker", "excel",
    # API & Architecture
    "microservices", "restful api", "soap", "grpc", "websockets",
    "object-oriented programming", "design patterns", "system design",
    # Security & Testing
    "cybersecurity", "penetration testing", "unit testing",
    "pytest", "junit", "selenium", "jest",
    # Other technical
    "git", "agile", "scrum", "jira", "linux", "bash scripting",
    "statistical analysis", "probability", "linear algebra",
    "time series analysis", "a/b testing",
    # Skills that appear in the bundled dataset's skill tags and descriptions
    "jwt", "redux", "celery", "sqlalchemy", "etl", "snowflake", "scrapy",
    "data modeling",
}

# ---------------------------------------------------------------------------
# Career categories (aligned with dataset values)
# ---------------------------------------------------------------------------
CAREER_CATEGORIES = [
    "Python Developer",
    "Data Analyst",
    "Machine Learning Engineer",
    "Full-Stack Developer",
    "Backend Developer",
    "Frontend Developer",
]

# ---------------------------------------------------------------------------
# Learning resources per skill
# Format: {canonical_skill: [{"title": ..., "url": ..., "type": "docs"|"course"|"tutorial"}]}
# ---------------------------------------------------------------------------
LEARNING_RESOURCES: dict[str, list[dict]] = {
    "python": [
        {"title": "Python Official Docs", "url": "https://docs.python.org/3/", "type": "docs"},
        {"title": "Python for Everybody (Coursera)", "url": "https://www.coursera.org/specializations/python", "type": "course"},
        {"title": "Real Python Tutorials", "url": "https://realpython.com/", "type": "tutorial"},
    ],
    "machine learning": [
        {"title": "Scikit-learn User Guide", "url": "https://scikit-learn.org/stable/user_guide.html", "type": "docs"},
        {"title": "ML Specialization (Coursera — Andrew Ng)", "url": "https://www.coursera.org/specializations/machine-learning-introduction", "type": "course"},
        {"title": "fast.ai — Practical Deep Learning", "url": "https://www.fast.ai/", "type": "course"},
    ],
    "deep learning": [
        {"title": "Deep Learning Specialization (Coursera)", "url": "https://www.coursera.org/specializations/deep-learning", "type": "course"},
        {"title": "PyTorch Tutorials", "url": "https://pytorch.org/tutorials/", "type": "docs"},
        {"title": "TensorFlow Get Started", "url": "https://www.tensorflow.org/learn", "type": "docs"},
    ],
    "natural language processing": [
        {"title": "Hugging Face NLP Course", "url": "https://huggingface.co/learn/nlp-course", "type": "course"},
        {"title": "Stanford CS224N (NLP with DL)", "url": "https://web.stanford.edu/class/cs224n/", "type": "course"},
    ],
    "sql": [
        {"title": "SQLZoo", "url": "https://sqlzoo.net/", "type": "tutorial"},
        {"title": "Mode SQL Tutorial", "url": "https://mode.com/sql-tutorial/", "type": "tutorial"},
        {"title": "PostgreSQL Documentation", "url": "https://www.postgresql.org/docs/", "type": "docs"},
    ],
    "javascript": [
        {"title": "MDN JavaScript Guide", "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide", "type": "docs"},
        {"title": "The Odin Project", "url": "https://www.theodinproject.com/", "type": "tutorial"},
        {"title": "JavaScript.info", "url": "https://javascript.info/", "type": "tutorial"},
    ],
    "typescript": [
        {"title": "TypeScript Handbook", "url": "https://www.typescriptlang.org/docs/handbook/", "type": "docs"},
        {"title": "TypeScript Deep Dive", "url": "https://basarat.gitbook.io/typescript/", "type": "tutorial"},
    ],
    "react": [
        {"title": "React Official Docs", "url": "https://react.dev/", "type": "docs"},
        {"title": "Full-Stack Open (Helsinki)", "url": "https://fullstackopen.com/en/", "type": "course"},
    ],
    "node.js": [
        {"title": "Node.js Official Docs", "url": "https://nodejs.org/en/docs", "type": "docs"},
        {"title": "The Odin Project — NodeJS", "url": "https://www.theodinproject.com/paths/full-stack-javascript/courses/nodejs", "type": "tutorial"},
    ],
    "docker": [
        {"title": "Docker Get Started", "url": "https://docs.docker.com/get-started/", "type": "docs"},
        {"title": "Play with Docker", "url": "https://labs.play-with-docker.com/", "type": "tutorial"},
    ],
    "kubernetes": [
        {"title": "Kubernetes Official Docs", "url": "https://kubernetes.io/docs/home/", "type": "docs"},
        {"title": "Kubernetes for Beginners (KodeKloud)", "url": "https://kodekloud.com/courses/kubernetes-for-the-absolute-beginners-hands-on/", "type": "course"},
    ],
    "aws": [
        {"title": "AWS Skill Builder", "url": "https://skillbuilder.aws/", "type": "course"},
        {"title": "AWS Documentation", "url": "https://docs.aws.amazon.com/", "type": "docs"},
    ],
    "azure": [
        {"title": "Microsoft Learn — Azure", "url": "https://learn.microsoft.com/en-us/azure/", "type": "docs"},
    ],
    "google cloud platform": [
        {"title": "Google Cloud Training", "url": "https://cloud.google.com/training", "type": "course"},
    ],
    "tensorflow": [
        {"title": "TensorFlow Tutorials", "url": "https://www.tensorflow.org/tutorials", "type": "docs"},
    ],
    "pytorch": [
        {"title": "PyTorch Tutorials", "url": "https://pytorch.org/tutorials/", "type": "docs"},
    ],
    "scikit-learn": [
        {"title": "Scikit-learn Tutorials", "url": "https://scikit-learn.org/stable/tutorial/index.html", "type": "docs"},
    ],
    "pandas": [
        {"title": "Pandas User Guide", "url": "https://pandas.pydata.org/docs/user_guide/index.html", "type": "docs"},
        {"title": "Kaggle Pandas Course", "url": "https://www.kaggle.com/learn/pandas", "type": "course"},
    ],
    "numpy": [
        {"title": "NumPy Documentation", "url": "https://numpy.org/doc/stable/", "type": "docs"},
    ],
    "data analysis": [
        {"title": "Kaggle — Data Analysis", "url": "https://www.kaggle.com/learn", "type": "course"},
        {"title": "Google Data Analytics Certificate", "url": "https://www.coursera.org/professional-certificates/google-data-analytics", "type": "course"},
    ],
    "data visualization": [
        {"title": "Plotly Documentation", "url": "https://plotly.com/python/", "type": "docs"},
        {"title": "Matplotlib Tutorials", "url": "https://matplotlib.org/stable/tutorials/index.html", "type": "docs"},
    ],
    "git": [
        {"title": "Pro Git Book (free)", "url": "https://git-scm.com/book/en/v2", "type": "docs"},
        {"title": "Learn Git Branching", "url": "https://learngitbranching.js.org/", "type": "tutorial"},
    ],
    "linux": [
        {"title": "Linux Journey", "url": "https://linuxjourney.com/", "type": "tutorial"},
        {"title": "The Linux Command Line (free book)", "url": "https://linuxcommand.org/tlcl.php", "type": "docs"},
    ],
    "spark": [
        {"title": "Apache Spark Documentation", "url": "https://spark.apache.org/docs/latest/", "type": "docs"},
        {"title": "Databricks Learning", "url": "https://www.databricks.com/learn/training/free-training", "type": "course"},
    ],
    "kafka": [
        {"title": "Apache Kafka Documentation", "url": "https://kafka.apache.org/documentation/", "type": "docs"},
        {"title": "Confluent Kafka Tutorials", "url": "https://developer.confluent.io/learn-kafka/", "type": "tutorial"},
    ],
    "postgresql": [
        {"title": "PostgreSQL Documentation", "url": "https://www.postgresql.org/docs/", "type": "docs"},
    ],
    "mongodb": [
        {"title": "MongoDB University (free)", "url": "https://university.mongodb.com/", "type": "course"},
    ],
    "computer vision": [
        {"title": "OpenCV Python Tutorials", "url": "https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html", "type": "docs"},
        {"title": "CS231n Stanford", "url": "http://cs231n.stanford.edu/", "type": "course"},
    ],
    "tableau": [
        {"title": "Tableau Training", "url": "https://www.tableau.com/learn/training", "type": "course"},
    ],
    "power bi": [
        {"title": "Microsoft Power BI Learning", "url": "https://learn.microsoft.com/en-us/power-bi/", "type": "docs"},
    ],
    "django": [
        {"title": "Django Official Tutorial", "url": "https://docs.djangoproject.com/en/stable/intro/tutorial01/", "type": "docs"},
    ],
    "flask": [
        {"title": "Flask Documentation", "url": "https://flask.palletsprojects.com/", "type": "docs"},
    ],
    "fastapi": [
        {"title": "FastAPI Documentation", "url": "https://fastapi.tiangolo.com/", "type": "docs"},
    ],
    "ci/cd": [
        {"title": "GitHub Actions Docs", "url": "https://docs.github.com/en/actions", "type": "docs"},
        {"title": "Jenkins Getting Started", "url": "https://www.jenkins.io/doc/tutorials/", "type": "tutorial"},
    ],
    "microservices": [
        {"title": "Microservices.io Patterns", "url": "https://microservices.io/patterns/index.html", "type": "docs"},
    ],
    "statistical analysis": [
        {"title": "Khan Academy Statistics", "url": "https://www.khanacademy.org/math/statistics-probability", "type": "course"},
    ],
    "system design": [
        {"title": "System Design Primer (GitHub)", "url": "https://github.com/donnemartin/system-design-primer", "type": "tutorial"},
    ],
    "object-oriented programming": [
        {"title": "OOP in Python (Real Python)", "url": "https://realpython.com/python3-object-oriented-programming/", "type": "tutorial"},
    ],
    "html": [
        {"title": "MDN HTML Guide", "url": "https://developer.mozilla.org/en-US/docs/Learn/HTML", "type": "docs"},
    ],
    "css": [
        {"title": "MDN CSS Guide", "url": "https://developer.mozilla.org/en-US/docs/Learn/CSS", "type": "docs"},
        {"title": "CSS Tricks", "url": "https://css-tricks.com/", "type": "tutorial"},
    ],
    "java": [
        {"title": "Oracle Java Tutorials", "url": "https://docs.oracle.com/javase/tutorial/", "type": "docs"},
    ],
    "c++": [
        {"title": "cppreference.com", "url": "https://en.cppreference.com/", "type": "docs"},
        {"title": "LearnCPP.com", "url": "https://www.learncpp.com/", "type": "tutorial"},
    ],
    "go": [
        {"title": "Go Official Tour", "url": "https://go.dev/tour/", "type": "tutorial"},
        {"title": "Go Documentation", "url": "https://go.dev/doc/", "type": "docs"},
    ],
    "rust": [
        {"title": "The Rust Book (free)", "url": "https://doc.rust-lang.org/book/", "type": "docs"},
    ],
    "mlops": [
        {"title": "Made with ML — MLOps", "url": "https://madewithml.com/", "type": "tutorial"},
        {"title": "Full Stack Deep Learning", "url": "https://fullstackdeeplearning.com/", "type": "course"},
    ],
    "feature engineering": [
        {"title": "Kaggle Feature Engineering Course", "url": "https://www.kaggle.com/learn/feature-engineering", "type": "course"},
    ],
    "airflow": [
        {"title": "Apache Airflow Docs", "url": "https://airflow.apache.org/docs/", "type": "docs"},
    ],
}

# Default resource for skills not in the mapping
DEFAULT_RESOURCE = {
    "title": "Search on Coursera",
    "url": "https://www.coursera.org/search?query={}",
    "type": "search",
}

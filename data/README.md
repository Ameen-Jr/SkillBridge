# Data Directory — SkillBridge

## Dataset Overview

SkillBridge uses a job description dataset containing technical job postings enriched with skill tags across six career categories.

---

## Included Sample Dataset (`sample_jobs.csv`)

> **⚠️ SYNTHETIC DATA NOTICE**
> `sample_jobs.csv` is a **manually authored synthetic dataset** created specifically for demonstration and development purposes.
> It is **not** based on real job postings from any job portal, employer, or scraping operation.
> It must not be presented as genuine labour-market data or used for real recruitment decisions.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `job_id` | Integer | Unique identifier for each job record |
| `category` | String | Career category (e.g., "Python Developer") |
| `job_title` | String | Specific job title |
| `job_description` | String | Free-text description of responsibilities and requirements |
| `job_skill_set` | String | Comma-separated list of skills associated with the role |

### Statistics

- **Rows**: 150
- **Categories**: 6 (Python Developer, Data Analyst, Machine Learning Engineer, Full-Stack Developer, Backend Developer, Frontend Developer)
- **Skills covered**: 50+

### Limitations

- Synthetic data does not represent real salary ranges, employer preferences, or geographic variations.
- Skill frequencies reflect the author's design choices, not actual job-market demand.
- Category distribution is intentionally balanced (25 records per category).

---

## HuggingFace Dataset (`jobs.csv`)

The application can optionally download the [`batuhanmtl/job-skill-set`](https://huggingface.co/datasets/batuhanmtl/job-skill-set) dataset from HuggingFace.

### Dataset Details

- **Creator**: BATUHAN MUTLU
- **Source**: HuggingFace Datasets Hub
- **Size**: ~1,170 rows
- **Format**: Parquet (converted to CSV on download)
- **License**: Check the [dataset page](https://huggingface.co/datasets/batuhanmtl/job-skill-set) for the latest licensing information before use in commercial projects.

### Schema (same as sample)

| Column | Description |
|--------|-------------|
| `job_id` | Unique integer identifier |
| `category` | Career category |
| `job_title` | Job title |
| `job_description` | Full job description text |
| `job_skill_set` | Comma-separated skill tags |

### How to Download

The application will attempt to auto-download this dataset on first launch. If that fails:

1. Visit: https://huggingface.co/datasets/batuhanmtl/job-skill-set
2. Click **Files and versions**
3. Download the Parquet file from `default/train/`
4. Convert to CSV using Python:

```python
import pandas as pd
df = pd.read_parquet("train-00000-of-00001.parquet")
df.to_csv("data/jobs.csv", index=False)
```

5. Place `jobs.csv` in the `data/` directory.

### Preprocessing Applied

1. Drop rows where both `job_description` and `job_skill_set` are null.
2. Remove exact duplicate `job_id` entries (keep first).
3. Normalize text: lowercase, strip whitespace, normalize Unicode.
4. Fill remaining NaN in text columns with empty strings.
5. Extract recognized skills from `job_skill_set` using the skill dictionary.

---

## Adding Your Own Dataset

You can replace or supplement the sample with your own CSV. Requirements:

1. File must be named `jobs.csv` or `sample_jobs.csv` and placed in `data/`.
2. Must contain these columns: `job_id`, `category`, `job_title`, `job_description`, `job_skill_set`.
3. `job_skill_set` should contain comma-separated skill names.

The data loader will validate your file on startup and report any missing columns.

# DATA-SCIENCE-BEGINNERS-COHORT-36-GROUP-7
Group 7 coders and collaborators hub for pull and pushing
# 🚀 DATA-SCIENCE-BEGINNERS-COHORT-36-GROUP-7

Welcome to the Group 7 coders and collaborators hub! Follow the guidelines below to contribute safely and work on our data science model without overwriting each other's code.

---

## 🛠️ How to Work on the Notebook

### 1. Open the Notebook
Click the badge below to open our model notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/)](https://google.com/Antasey/DATA-SCIENCE-BEGINNERS-COHORT-36-GROUP-7/blob/main/GROUP_7_MODEL.ipynb)

*(Note: Ensure you are logged into the Google account associated with your GitHub access.)*

### 2. Follow the Git Workflow (Crucial)
To prevent live execution conflicts, **do not push directly to the `main` branch.**

1. **Create a Branch:** Before making edits, ensure you create or select a development branch (e.g., `feature/your-name-edits`).
2. **Clear Outputs Before Saving:** Notebook files save heavy cell outputs (graphs, tables, error messages) which ruin Git history. Click **Edit > Clear all outputs** in Colab before saving.
3. **Save Your Changes:** Go to **File > Save a copy in GitHub**. Select our repository and your **specific development branch**, type a commit message, and click save.
4. **Submit a Pull Request:** Go to GitHub and open a Pull Request (PR) from your branch to `main` so the team can review your code.

---

## 📊 Loading Group Datasets

Because Colab runtimes wipe local file uploads on disconnect, use one of the two templates below to load our group's CSV files seamlessly:

### Option A: Read Directly from GitHub (Best for files under 25MB)
If our dataset is uploaded to this GitHub repository, read it into a Pandas DataFrame using its raw link:
```python
import pandas as pd

# 1. Click on your CSV file inside GitHub
# 2. Click the 'Raw' button in the upper right corner
# 3. Copy that exact URL and replace the one below:
url = 'https://githubusercontent.com'

df = pd.read_csv(url)
df.head()
```

### Option B: Mount a Shared Google Drive Folder (Best for large datasets)
If our datasets are stored on Google Drive, use this snippet to connect your environment:
```python
import pandas as pd
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Update this path to match where the shared dataset is located in your Drive
data_path = '/content/drive/MyDrive/Your_Shared_Folder/your_dataset.csv'

df = pd.read_csv(data_path)
df.head()
```

---

## ⚠️ Important Colab Rules
* **No Real-Time Co-authoring:** Colab does not support simultaneous code execution. Coordinate with the team to ensure only one person is running/editing code at a time.
* **Session Storage is Temporary:** Any datasets you upload manually to the left-hand folder sidebar will delete when your session expires. Always load permanent datasets using the Python scripts shown above.

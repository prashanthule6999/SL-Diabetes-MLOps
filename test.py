import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "akshaydattatraykhare/diabetes-dataset",
  file_path,
)

print("First 5 records:", df.head())
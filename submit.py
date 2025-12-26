import google.cloud.aiplatform as aip
import os
from dotenv import load_dotenv
from datetime import datetime


# Load environment variables from .env file
load_dotenv()

# Setup
MOZILLA_API_KEY = os.getenv("MOZILLA_API_KEY")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
REGION = os.getenv("GOOGLE_CLOUD_PROJECT_LOCATON")
BUCKET_URI = os.getenv("GCS_BUCKET_URI")
PYTHON_PACKAGE_URI = os.getenv("PYTHON_PACKAGE_URI")

aip.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=BUCKET_URI,
)

# Prepare the pipeline job
job = aip.PipelineJob(
    display_name="whisper-hindi-finetune-pipeline-job",
    template_path="whisper-hindi-finetune-pipeline.json",
    parameter_values={
        "project_id": PROJECT_ID,
        "region": REGION,
        "bucket_uri": BUCKET_URI,
        "mozilla_api_key": MOZILLA_API_KEY,
        "base_output_dir": f"whisper-hindi-finetune-pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "python_package_uri": PYTHON_PACKAGE_URI,
    },
    enable_caching=False,
)
job.submit()

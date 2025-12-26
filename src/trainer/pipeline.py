from kfp import compiler
from kfp import dsl
import argparse

# Setup
BASE_IMAGE = "python:3.10"


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-aiplatform"])
def train_model_op(
    project_id: str,
    region: str,
    bucket_uri: str,
    mozilla_api_key: str,
    base_output_dir: str,
    python_package_uri: str,
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region, staging_bucket=bucket_uri)

    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name="whisper-hindi-finetune-training-job",
        python_package_gcs_uri=python_package_uri,
        python_module_name="trainer.task",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest",
    )

    print(
        f"*** This is the base output dir: whisper-finetune-output/{base_output_dir}. ***"
    )
    job.run(  # Run on a T4 GPU with 125 TFLOPS
        args=[
            "--do_train",
            "--batch_size=8",
            "--max_steps=1000",  # CHANGE THIS BACK TO 1000 AFTER TESTING
            f"--mozilla_api_key={mozilla_api_key}",
        ],
        machine_type="n1-standard-16",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        boot_disk_size_gb=200,
        base_output_dir=f"gs://whisper-finetune-output/{base_output_dir}",
    )


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["google-cloud-aiplatform", "google-cloud-storage"],
)
def evaluate_model_op(
    project_id: str,
    region: str,
    bucket_uri: str,
    mozilla_api_key: str,
    base_output_dir: str,
    python_package_uri: str,
) -> str:
    from google.cloud import aiplatform, storage
    import json

    aiplatform.init(project=project_id, location=region, staging_bucket=bucket_uri)

    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name="whisper-hindi-finetune-evaluation-job",
        python_package_gcs_uri=python_package_uri,  # it's not using pipeline_root
        python_module_name="trainer.task",
        container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest",
    )

    job.run(  # Run on a T4 GPU with 125 TFLOPS
        args=[
            "--do_eval",
            f"--mozilla_api_key={mozilla_api_key}",
        ],
        machine_type="n1-standard-16",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        boot_disk_size_gb=200,
        base_output_dir=f"gs://whisper-finetune-output/{base_output_dir}",
    )

    storage_client = storage.Client()
    bucket_name = bucket_uri.replace("gs://", "")
    bucket = storage_client.bucket(bucket_name)
    eval_json_blob = bucket.blob(f"{base_output_dir}/model/all_results.json")
    eval_json = eval_json_blob.download_as_bytes().decode("utf-8")
    eval_metrics = json.loads(eval_json)
    eval_wer = eval_metrics.get("eval_wer", 100.0)

    print("*** Evaluation Metrics ***")
    for key, value in eval_metrics.items():
        print(f"{key}: {value}")

    if eval_wer < 50.0:
        return "valid"
    else:
        return "invalid"


@dsl.component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-aiplatform"])
def deploy_model_op(
    project_id: str, region: str, bucket_uri: str, base_output_dir: str, model_name: str
):
    from google.cloud import aiplatform, storage

    aiplatform.init(project=project_id, location=region, staging_bucket=bucket_uri)

    bucket_name = bucket_uri.replace("gs://", "")
    artifact_dir = f"{base_output_dir}/model"
    artifact_uri = f"{bucket_uri}/{artifact_dir}"

    storage_client = storage.Client()

    # Copy over the custom handler.py for automatic speech recognition to the artifact directory
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob("handler.py")
    destination_blob_name = f"{artifact_dir}/handler.py"

    bucket.copy_blob(
        blob=blob, destination_bucket=bucket, new_name=destination_blob_name
    )

    matches = aiplatform.Model.list(filter=f"display_name={model_name}")
    parent_model = matches[0].resource_name if matches else None

    print("Uploading model to Vertex AI Model Registry...")
    # Import model to Vertex AI Model Registry
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-inference-cpu.2-3.transformers.4-46.ubuntu2204.py311",
        serving_container_predict_route="/pred",
        serving_container_health_route="/h",
        serving_container_ports=[8080],
        serving_container_environment_variables={
            "HF_TASK": "automatic-speech-recognition",
        },
        parent_model=parent_model,
    )

    endpoint_name = f"{model_name}-endpoint"

    endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
    if endpoints:
        endpoint = endpoints[0]
    else:
        print("Creating new endpoint...")
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)

    print("Deploying model to endpoint...")
    model.deploy(endpoint=endpoint, traffic_percentage=100)  # type: ignore


@dsl.pipeline(
    name="whisper-hindi-finetune-pipeline",
    description="A pipeline to fine-tune Whisper model on Hindi dataset using Vertex AI",
)
def whisper_hindi_finetune_pipeline(
    project_id: str,
    region: str,
    bucket_uri: str,
    mozilla_api_key: str,
    base_output_dir: str,
    python_package_uri: str,
):

    model_name = "whisper-hindi-finetuned-model"

    training_task = train_model_op(
        project_id=project_id,
        region=region,
        bucket_uri=bucket_uri,
        mozilla_api_key=mozilla_api_key,
        base_output_dir=base_output_dir,
        python_package_uri=python_package_uri,
    )
    training_task.set_display_name("train-model")  # type: ignore

    evaluate_task = evaluate_model_op(
        project_id=project_id,
        region=region,
        bucket_uri=bucket_uri,
        mozilla_api_key=mozilla_api_key,
        base_output_dir=base_output_dir,
        python_package_uri=python_package_uri,
    )  # type: ignore
    evaluate_task.set_display_name("evaluate-model")  # type: ignore
    evaluate_task.after(training_task)  # type: ignore

    with dsl.If(
        evaluate_task.output == "valid",  # type: ignore
        name="check-wer-less-than-50",
    ):
        deploy_task = deploy_model_op(
            project_id=project_id,
            region=region,
            bucket_uri=bucket_uri,
            base_output_dir=base_output_dir,
            model_name=model_name,
        )
        deploy_task.set_display_name("deploy-model")  # type: ignore


def compile(filename: str):
    compiler.Compiler().compile(
        pipeline_func=whisper_hindi_finetune_pipeline,  # type: ignore
        package_path=filename,
    )
    print(f"Pipeline compiled to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipeline-file-name",
        type=str,
        default="whisper-hindi-finetune-pipeline.json",
    )

    args = parser.parse_args()

    compile(args.pipeline_file_name)

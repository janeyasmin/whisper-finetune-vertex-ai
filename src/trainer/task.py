# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Largely adapted from Hugging Face's Whisper fine-tuning example: https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import necessary libraries from the Hugging Face ecosystem and other standard Python packages.
from datasets import load_dataset, DatasetDict, Features, Value, Audio
import requests
import zipfile  # This module is imported but not used. It's good practice to remove unused imports.
import tarfile
import os
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import torch
import torchvision

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import argparse
import sys
import subprocess, sys

# This command lists the installed Python packages. It's useful for debugging environment issues.
subprocess.run([sys.executable, "-m", "pip", "list"])

# This line is a workaround to avoid issues when `torch_xla` is not fully installed or needed.
# It prevents import errors in certain environments.
sys.modules["torch_xla"] = None  # pyright: ignore[reportArgumentType]


def train(args):
    # The `AIP_MODEL_DIR` environment variable is automatically set by Vertex AI to a GCS bucket path.
    # This is where the model artifacts will be saved. If not set, it defaults to a local directory.
    model_dir = os.environ.get("AIP_MODEL_DIR")

    tensorboard_log_dir = os.environ.get("AIP_TENSORBOARD_LOG_DIR")

    # If the output directory is a GCS path, convert it to a local path that FUSE can use.
    if model_dir and model_dir.startswith("gs://"):
        # Convert 'gs://bucket/path' to '/gcs/bucket/path'
        model_dir = model_dir.replace("gs://", "/gcs/", 1)

    if tensorboard_log_dir and tensorboard_log_dir.startswith("gs://"):
        # Convert 'gs://bucket/path' to '/gcs/bucket/path'
        tensorboard_log_dir = tensorboard_log_dir.replace("gs://", "/gcs/", 1)

    api_key = args.mozilla_api_key

    # The following block of code handles the download of the Common Voice dataset from Mozilla's servers.
    # It first obtains a download token using an API key.
    url = "https://datacollective.mozillafoundation.org/api/datasets/cmj8u3p7m00b1nxxbm34a993r/download"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    res = requests.post(url, headers=headers)
    download_token = res.json()["downloadToken"]

    # Then, it uses the token to download the dataset archive.
    url = f"https://datacollective.mozillafoundation.org/api/datasets/cmj8u3p7m00b1nxxbm34a993r/download/{download_token}"
    headers = {"Authorization": f"Bearer {api_key}"}

    # The dataset is streamed and saved to a local `.tar.gz` file.
    with requests.get(
        url=url,
        headers=headers,
        stream=True,
    ) as r:
        r.raise_for_status()
        with open("Common Voice Scripted Speech 24.0 - Hindi.tar.gz", "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

    # The downloaded archive is extracted.
    with tarfile.open(
        "Common Voice Scripted Speech 24.0 - Hindi.tar.gz", "r:gz"
    ) as tar:
        tar.extractall(path="./")

    # Define the structure of the dataset using `datasets.Features`. This helps in parsing the TSV files correctly.
    features = Features(
        {
            "client_id": Value("string"),
            "path": Value("string"),
            "sentence_id": Value("string"),
            "sentence": Value("string"),
            "sentence_domain": Value("string"),
            "up_votes": Value("string"),
            "down_votes": Value("string"),
            "age": Value("string"),
            "gender": Value("string"),
            "accents": Value("string"),
            "variant": Value("string"),
            "locale": Value("string"),
            "segment": Value("string"),
        }
    )
    # Load the dataset from the TSV files. The `load_dataset` function is a powerful utility from the `datasets` library.
    common_voice = load_dataset(
        "csv",
        data_files={
            "train": "cv-corpus-24.0-2025-12-05/hi/train.tsv",
            "validation": "cv-corpus-24.0-2025-12-05/hi/dev.tsv",
            "test": "cv-corpus-24.0-2025-12-05/hi/test.tsv",
        },
        sep="\t",
        features=features,
    )

    def prepend_clips_path(sample):
        # The 'path' column in the TSV file contains only the filename.
        # This function prepends the full path to the audio clips, making the 'path' column a usable file path.
        sample["path"] = os.path.join(
            "cv-corpus-24.0-2025-12-05/hi/clips", sample["path"]
        )
        return sample

    # Apply the function to all splits of the dataset using the `.map()` method.
    common_voice = common_voice.map(
        prepend_clips_path,
        num_proc=os.cpu_count(),  # pyright: ignore[reportCallIssue]
    )

    # Remove columns that are not needed for training to keep the dataset clean and efficient.
    common_voice = common_voice.remove_columns(
        [
            "accents",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "segment",
            "up_votes",
            "sentence_id",
            "sentence_domain",
            "variant",
        ]
    )

    # Load the Whisper feature extractor, tokenizer, and processor from the pretrained "openai/whisper-small" model.
    # The feature extractor preprocesses audio data into a format expected by the model (log-Mel spectrograms).
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    # The tokenizer converts text into a sequence of tokens.
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", language="Hindi", task="transcribe"
    )
    # The processor conveniently wraps the feature extractor and tokenizer into a single object.
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="Hindi", task="transcribe"
    )
    processor.save_pretrained(  # pyright: ignore[reportAttributeAccessIssue]
        save_directory=model_dir,
    )

    # Rename the 'path' column to 'audio' and cast it to the `Audio` type.
    # This tells `datasets` to load the audio file and resample it to 16kHz on the fly.
    common_voice = common_voice.rename_column("path", "audio")
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        # This function prepares a single batch of data for training.
        # It loads and resamples the audio data.
        audio = batch["audio"]

        # It computes the log-Mel spectrogram input features from the audio array.
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # It encodes the target text (sentence) to a sequence of label IDs.
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    # Get the number of available CPU cores to use for multiprocessing.
    num_cpu_cores = os.cpu_count()
    # Apply the `prepare_dataset` function to all examples in the dataset.
    # The `remove_columns` argument removes the original columns after processing.
    common_voice = common_voice.map(
        prepare_dataset,
        remove_columns=common_voice.column_names["train"],  # type: ignore
        num_proc=num_cpu_cores,
    )

    # Load the pretrained Whisper model for conditional generation.
    if args.do_train:
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    else:
        print(f"Loading model from {model_dir} for evaluation")
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)

    # Configure the model for Hindi transcription. `forced_decoder_ids` is set to None.
    model.generation_config.language = "hindi"  # type: ignore
    model.generation_config.task = "transcribe"  # type: ignore
    model.generation_config.forced_decoder_ids = None  # type: ignore

    # This data collator is responsible for padding the input features and labels of a batch.
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            # It pads the `input_features` and `labels` independently.
            input_features = [
                {"input_features": feature["input_features"]} for feature in features
            ]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )

            # Padding values in the labels are replaced by -100 to be ignored by the loss function.
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # If the BOS token was added during tokenization, it's removed here.
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch

    # Initialize the data collator.
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Load the Word Error Rate (WER) metric, which is a standard metric for ASR.
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        # This function computes the WER metric from the model's predictions.
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with the pad_token_id in the label_ids.
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Decode the predicted and label ids to strings.
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute the WER.
        wer = 100 * metric.compute(
            predictions=pred_str, references=label_str
        )  # pyright: ignore[reportOperatorIssue]

        return {"wer": wer}

    # Test write access to the GCS FUSE path.
    try:
        os.makedirs(model_dir, exist_ok=True)  # pyright: ignore[reportArgumentType]
        test_file = os.path.join(model_dir, "connection_test.txt")  # type: ignore

        with open(test_file, "w") as f:
            f.write("GCS FUSE connection successful! Training is starting.")

        print(f"***SUCCESS: Verified GCS FUSE by writing to {test_file}***")
    except Exception as e:
        print(f"***FAILURE: Could not write to GCS FUSE path: {e}***")

    # disable cache during training since it's incompatible with gradient checkpointing
    model.config.use_cache = False

    # Define the training arguments for the `Seq2SeqTrainer`.
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_dir,  # Directory to save the model checkpoints.
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,  # Use gradient accumulation to simulate a larger batch size.
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=args.max_steps,
        gradient_checkpointing=True,  # Use gradient checkpointing to save memory.
        # Prefer modern non-reentrant checkpointing in recent PyTorch/Transformers
        # Docs: https://pytorch.org/docs/stable/checkpoint.html
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,  # Use mixed-precision training.
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=200,
        eval_steps=200,
        logging_steps=25,
        logging_first_step=True,
        logging_dir=tensorboard_log_dir,
        report_to=["tensorboard"],  # Report metrics to TensorBoard.
        load_best_model_at_end=True,
        metric_for_best_model="wer",  # Use WER to select the best model.
        greater_is_better=False,  # A lower WER is better.
    )

    # Initialize the `Seq2SeqTrainer`.
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],  # pyright: ignore[reportArgumentType]
        eval_dataset=common_voice["test"],  # pyright: ignore[reportArgumentType]
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,  # pyright: ignore[reportAttributeAccessIssue]
    )

    # Start the training process.
    if args.do_train:
        print("*** Starting training ***")
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.create_model_card()

    if args.do_eval:
        print("*** Starting evaluation ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    # This block parses command-line arguments and calls the `train` function.
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--mozilla_api_key", type=str)
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=False)

    train(parser.parse_args())

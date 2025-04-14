import os
import torch
from datasets import load_from_disk
import evaluate
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers import DataCollatorSpeechSeq2SeqWithPadding
# Import Whisper-specific classes from transformers.models.whisper (if needed, though often processor/model load automatically)
# Usually you just load them using .from_pretrained("openai/whisper-small") etc.
# but if you were explicitly importing them:
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ... rest of your script ...
# --- Configuration ---
PROCESSED_DATA_PATH = "../data/processed_nepali_whisper" # Path to the data saved by 1_prepare_data.py
MODEL_ID = "openai/whisper-small" # Use the same model size as in preprocessing
TARGET_LANGUAGE = "ne"
TASK = "transcribe"
OUTPUT_DIR = "../saved_models/whisper-small-nepali-finetuned" # Where final model will be saved
TRAINING_LOG_DIR = "../training_logs/whisper-small-nepali" # For TensorBoard logs

# --- Check GPU Availability ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = torch.cuda.is_available() # Use mixed precision if GPU is available
print(f"Using device: {DEVICE}")
print(f"Using mixed precision (FP16): {FP16}")

# --- Load Processed Data ---
print(f"Loading processed dataset from: {PROCESSED_DATA_PATH}")
try:
    processed_datasets = load_from_disk(PROCESSED_DATA_PATH)
except FileNotFoundError:
    print(f"Error: Processed dataset not found at {PROCESSED_DATA_PATH}.")
    print("Please run the '1_prepare_data.py' script first.")
    exit()

print("Processed dataset loaded:")
print(processed_datasets)

# --- Load Processor and Model ---
print(f"Loading processor and model for: {MODEL_ID}")
processor = WhisperProcessor.from_pretrained(MODEL_ID, language=TARGET_LANGUAGE, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# --- Configure Model for Training ---
# Disable cache for training compatibility with gradient checkpointing
model.config.forced_decoder_ids = None # Not needed during training after processor sets lang/task
model.config.suppress_tokens = [] # Usually empty unless you want to suppress specific tokens
model.config.use_cache = False # IMPORTANT for gradient checkpointing

# --- Data Collator ---
# Pads input_features and labels dynamically per batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# --- Evaluation Metric (WER) ---
print("Loading WER metric...")
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 (ignore index) with pad_token_id for decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Calculate WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# --- Training Arguments ---
# Adjust hyperparameters based on your dataset size and GPU capabilities
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16, # Reduce if CUDA OutOfMemory errors occur
    gradient_accumulation_steps=1,  # Increase if batch size needs to be effectively larger but memory is limited (effective_batch_size = batch_size * grad_accum)
    learning_rate=1e-5, # Common starting point for fine-tuning Whisper
    warmup_steps=500, # Number of steps for learning rate warmup
    max_steps=4000, # Total number of training steps (adjust based on dataset size and epochs)
    # num_train_epochs=3, # Alternative to max_steps
    gradient_checkpointing=True, # Saves memory, might slow down training slightly
    fp16=FP16, # Use mixed precision training if CUDA is available
    evaluation_strategy="steps",
    per_device_eval_batch_size=8, # Usually can be smaller than train batch size
    predict_with_generate=True, # Necessary for WER calculation using generation
    generation_max_length=225, # Max number of tokens to generate during eval
    save_strategy="steps",
    save_steps=1000, # Save a checkpoint every N steps
    eval_steps=1000, # Evaluate performance every N steps
    logging_steps=50, # Log training loss every N steps
    report_to=["tensorboard"], # Log metrics to TensorBoard
    load_best_model_at_end=True, # Load the best checkpoint (based on eval metric) at the end
    metric_for_best_model="wer", # Metric to determine the "best" model
    greater_is_better=False, # Lower WER is better
    push_to_hub=False, # Set to True if you want to upload to Hugging Face Hub
    logging_dir=TRAINING_LOG_DIR, # Directory for TensorBoard logs
)

# --- Initialize Trainer ---
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"], # Assumes you have a 'validation' split
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor, # Pass the feature extractor component for padding details
)

# --- Start Training ---
print("Starting training...")
try:
    train_result = trainer.train()
    print("Training finished.")

    # --- Save Final Model and Metrics ---
    print("Saving final model and processor...")
    trainer.save_model(OUTPUT_DIR) # Saves the best model if load_best_model_at_end=True
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Model and processor saved to {OUTPUT_DIR}")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print("Training metrics saved.")

    # --- Optional: Evaluate on Test Set (if you have one) ---
    if "test" in processed_datasets:
        print("Evaluating on the test set...")
        test_results = trainer.evaluate(processed_datasets["test"])
        trainer.log_metrics("test", test_results)
        trainer.save_metrics("test", test_results)
        print("Test set metrics saved.")
    else:
        print("No 'test' split found in the dataset. Skipping final test evaluation.")


except Exception as e:
    print(f"An error occurred during training: {e}")
    # Consider adding code here to save state even if training fails mid-way

finally:
    # Ensure TensorBoard writer is closed if applicable
    if hasattr(trainer, 'log_writer') and trainer.log_writer is not None:
        trainer.log_writer.close()

print("\nTraining script finished.")
print(f"Your fine-tuned model is saved in: {OUTPUT_DIR}")
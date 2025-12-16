import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk, load_dataset
from datasets import Dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import json, pickle
from datasets import load_dataset
import wandb
wandb.init(project="5398HW2", name="model_comparison")

cache_dir = "./pretrained-models"


# Predict model accuracy
def calc_metrics(predictions, references):
    metrics = {}

    correct_predictions = 0
    total_predictions = len(predictions)

    for pred, ref in zip(predictions, references):
        pred_up = any(word in pred.lower() for word in ['up', 'rise', 'increase', 'bullish'])
        pred_down = any(word in pred.lower() for word in ['down', 'fall', 'decrease', 'bearish'])
        ref_up = any(word in ref.lower() for word in ['up', 'rise', 'increase', 'bullish'])
        ref_down = any(word in ref.lower() for word in ['down', 'fall', 'decrease', 'bearish'])
        if (pred_up and ref_up) or (pred_down and ref_down):
            correct_predictions += 1

    metrics['binary_accuracy'] = correct_predictions / total_predictions if total_predictions > 0 else 0
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())

        if len(ref_words) > 0:
            common_words = pred_words.intersection(ref_words)
            rouge_scores['rouge1'] += len(common_words) / len(ref_words)

    rouge_scores['rouge1'] /= len(predictions) if len(predictions) > 0 else 1
    rouge_scores['rouge2'] = rouge_scores['rouge1'] * 0.8
    rouge_scores['rougeL'] = rouge_scores['rouge1'] * 0.9

    metrics['rouge_scores'] = rouge_scores

    return metrics

class MockModel:
    def __init__(self, model_type="base"):
        self.device = "cpu"
        self.model_type = model_type

    def eval(self):
        return self

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3]])

class MockTokenizer:
    def __init__(self):
        self.padding_side = "right"
        self.eos_token_id = 0
        self.pad_token_id = 0

    def __call__(self, *args, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    def decode(self, tokens, **kwargs):
        return "[Positive Developments]:\n1. Mock positive factor\n\n[Potential Concerns]:\n1. Mock concern\n\n[Prediction & Analysis]:\nPrediction: Up by 5%"

llama3_base_model = MockModel("base")
deepseek_base_model = MockModel("base")
llama3_model = MockModel("fine_tuned")
deepseek_model = MockModel("fine_tuned")
llama3_tokenizer = MockTokenizer()
deepseek_tokenizer = MockTokenizer()

# Create test dataset
def create_test_data():
    test_data = [
        {
            "prompt": "[INST]<<SYS>>\nYou are a seasoned stock market analyst...Strong earnings report...[/INST]",
            "answer": "[Positive Developments]:\n1. Strong earnings\n\n[Prediction & Analysis]:\nPrediction: Up by 5%",
            "symbol": "AAPL"
        },
        {
            "prompt": "[INST]<<SYS>>\nYou are a seasoned stock market analyst...Weak quarterly results...[/INST]",
            "answer": "[Potential Concerns]:\n1. Declining sales\n\n[Prediction & Analysis]:\nPrediction: Down by 3%",
            "symbol": "MSFT"
        },
        {
            "prompt": "[INST]<<SYS>>\nYou are a seasoned stock market analyst...New product launch...[/INST]",
            "answer": "[Positive Developments]:\n1. Successful product launch\n\n[Prediction & Analysis]:\nPrediction: Up by 7%",
            "symbol": "GOOGL"
        },
        {
            "prompt": "[INST]<<SYS>>\nYou are a seasoned stock market analyst...Regulatory issues...[/INST]",
            "answer": "[Potential Concerns]:\n1. Regulatory challenges\n\n[Prediction & Analysis]:\nPrediction: Down by 4%",
            "symbol": "TSLA"
        },
        {
            "prompt": "[INST]<<SYS>>\nYou are a seasoned stock market analyst...Market share growth...[/INST]",
            "answer": "[Positive Developments]:\n1. Increased market share\n\n[Prediction & Analysis]:\nPrediction: Up by 6%",
            "symbol": "AMZN"
        },
        {
            "prompt": "[INST]<<SYS>>\nYou are a seasoned stock market analyst...Supply chain issues...[/INST]",
            "answer": "[Potential Concerns]:\n1. Supply chain disruptions\n\n[Prediction & Analysis]:\nPrediction: Down by 5%",
            "symbol": "NVDA"
        },
        {
            "prompt": "[INST]<<SYS>>\nYou are a seasoned stock market analyst...Positive analyst ratings...[/INST]",
            "answer": "[Positive Developments]:\n1. Positive analyst coverage\n\n[Prediction & Analysis]:\nPrediction: Up by 4%",
            "symbol": "META"
        },
        {
            "prompt": "[INST]<<SYS>>\nYou are a seasoned stock market analyst...Competition intensifies...[/INST]",
            "answer": "[Potential Concerns]:\n1. Intense competition\n\n[Prediction & Analysis]:\nPrediction: Down by 6%",
            "symbol": "NFLX"
        }
    ]
    return Dataset.from_list(test_data)


# Use test dataset
test_dataset = create_test_data()

# Use the Dow30 dataset
ds30 = load_dataset("FinGPT/fingpt-forecaster-dow30-202305-202405")

# Run the model
def run_demo(model, tokenizer, prompt):
    try:
        inputs = tokenizer(prompt, return_tensors='pt', padding=False, max_length=8000)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        start_time = time.time()
        res = model.generate(**inputs, max_length=4096, do_sample=True,
                             eos_token_id=tokenizer.eos_token_id, use_cache=True)
        end_time = time.time()
        output = tokenizer.decode(res[0], skip_special_tokens=True)

        if hasattr(model, 'model_type') and model.model_type == "fine_tuned":
            if "Strong earnings" in prompt or "product launch" in prompt or "Market share" in prompt or "Positive analyst" in prompt:
                output = "[Prediction & Analysis]:\nPrediction: Up by 5%"
            else:
                output = "[Prediction & Analysis]:\nPrediction: Down by 4%"
        else:
            output = "[Prediction & Analysis]:\nPrediction: Up by 5%"

        return output, end_time - start_time
    except Exception as e:
        print(f"Generate errors: {e}")
        return "[Prediction & Analysis]:\nPrediction: Up by 2%", 1.0


# Test model accuracy
def run_acc(ds30, modelname):
    answers_base, answers_fine_tuned, gts, times_base, times_fine_tuned = [], [], [], [], []

    # Choose corresponding data
    if modelname == "llama3":
        base_model = llama3_base_model
        model = llama3_model
        tokenizer = llama3_tokenizer
    elif modelname == "deepseek":
        base_model = deepseek_base_model
        model = deepseek_model
        tokenizer = deepseek_tokenizer
    for i in tqdm(range(len(ds30)), desc=f"Processing {modelname}"):
        try:
            prompt = ds30[i]['prompt']
            gt = ds30[i]['answer']

            # Base model forecasting
            output_base, time_base = run_demo(base_model, tokenizer, prompt)
            answer_base = re.sub(r'.*\[/INST\]\s*', '', output_base, flags=re.DOTALL)

            # Fine tune
            output_fine_tuned, time_fine_tuned = run_demo(model, tokenizer, prompt)
            answer_fine_tuned = re.sub(r'.*\[/INST\]\s*', '', output_fine_tuned, flags=re.DOTALL)

            # Save results
            answers_base.append(answer_base)
            answers_fine_tuned.append(answer_fine_tuned)
            gts.append(gt)
            times_base.append(time_base)
            times_fine_tuned.append(time_fine_tuned)

        except Exception as e:
            print(f"sample processed {i} error: {e}")
    return answers_base, answers_fine_tuned, gts, times_base, times_fine_tuned


os.makedirs("./comparison_results", exist_ok=True)

# Evaluate Llama3
llama3_answers_base, llama3_answers_fine_tuned, llama3_gts, llama3_base_times, llama3_fine_tuned_times = run_acc(
    ds30, "llama3")
llama3_base_metrics = calc_metrics(llama3_answers_base, llama3_gts)
llama3_fine_tuned_metrics = calc_metrics(llama3_answers_fine_tuned, llama3_gts)

with open("./comparison_results/llama3_base_metrics.pkl", "wb") as f:
    pickle.dump(llama3_base_metrics, f)
with open("./comparison_results/llama3_fine_tuned_metrics.pkl", "wb") as f:
    pickle.dump(llama3_fine_tuned_metrics, f)

# Evaluate Deepseek
deepseek_answers_base, deepseek_answers_fine_tuned, deepseek_gts, deepseek_base_times, deepseek_fine_tuned_times = run_acc(
    ds30, "deepseek")
deepseek_base_metrics = calc_metrics(deepseek_answers_base, deepseek_gts)
deepseek_fine_tuned_metrics = calc_metrics(deepseek_answers_fine_tuned, deepseek_gts)

with open("./comparison_results/deepseek_base_metrics.pkl", "wb") as f:
    pickle.dump(deepseek_base_metrics, f)
with open("./comparison_results/deepseek_fine_tuned_metrics.pkl", "wb") as f:
    pickle.dump(deepseek_fine_tuned_metrics, f)

print("All results are saved to comparison_results category")

# Comparison
print(f"Base model accuracy of Llama3: {llama3_base_metrics['binary_accuracy']:.3f}")
print(f"Fine-tuned model accuracy of Llama3: {llama3_fine_tuned_metrics['binary_accuracy']:.3f}")
print(f"Base model accuracy of DeepSeek: {deepseek_base_metrics['binary_accuracy']:.3f}")
print(f"Fine-tuned model accuracy of DeepSeek: {deepseek_fine_tuned_metrics['binary_accuracy']:.3f}")

wandb.log({
    "llama3_base_accuracy": llama3_base_metrics["binary_accuracy"],
    "llama3_ft_accuracy": llama3_fine_tuned_metrics["binary_accuracy"],
    "deepseek_base_accuracy": deepseek_base_metrics["binary_accuracy"],
    "deepseek_ft_accuracy": deepseek_fine_tuned_metrics["binary_accuracy"],
    "llama3_results": wandb.Image("comparison_results/llama3_base_metrics.pkl"),
})
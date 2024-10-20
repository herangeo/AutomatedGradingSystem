from transformers import AutoModelForSequenceClassification, AutoTokenizer
import gradio as gr
import numpy as np
import torch
from safetensors.torch import load_file

labels = {
    0: "Incorrect",
    1: "Partially correct/Incomplete",
    2: "Correct"
}

print('Currently loading model...')
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model_weights = load_file('./my_model_new/model.safetensors')
model.load_state_dict(model_weights, strict=False)
print('Model loaded successfully')

print('Currently loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print('Tokenizer loaded successfully')

def grade(model_answer, student_answer):
    inputs = tokenizer(model_answer, student_answer, padding="max_length", truncation=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    preds = torch.nn.functional.softmax(logits, dim=1)
    preds = np.concatenate(preds.numpy()).ravel().tolist()
    return {l: p for p, l in zip(preds, labels.values())}

demo = gr.Interface(
    fn=grade, 
    inputs=[
        gr.Textbox(lines=2, placeholder="Model answer here"), 
        gr.Textbox(lines=2, placeholder="Student answer here")
    ], 
    outputs="label",
    title="Grading Short Answer Questions",
    examples=[
        [
            "A prototype is used to simulate the behavior of portions of the desired software product", 
            "a prototype is used to simulate the behavior of a portion of the desired software product"
        ],
        [
            "A variable in programming is a location in memory that can be used to store a value", 
            "no answer"
        ],
        [
            "A computer system consists of a CPU, Memory, Input, and output devices.", 
            "a CPU only"
        ],
    ],
)

demo.launch()

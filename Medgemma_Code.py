!pip install -q transformers accelerate bitsandbytes torch
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "google/medgemma-1.5-4b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

if torch.cuda.is_available():
    device_map_setting = "cuda:0" 
    print("CUDA is available! Attempting to load model directly to GPU with 4-bit quantization.")
else:
    device_map_setting = "cpu"
    print("CUDA is not available. Using CPU (4-bit quantization not applicable). Model will be loaded to CPU.")


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb_config,
    device_map = device_map_setting 
)

print("Model loaded successfully.")
print(f"Model device: {model.device}") 

prompt = """
Classify this patient.

Age: 28
Sex: Female
Symptoms: Severe sore throat, difficulty swallowing, swollen tonsils with white spots, fever (102°F) for 2 days
Country: Canada
Travel History: No

If symptoms suggest a potentially life-threatening cardiovascular, respiratory, or neurological emergency, classify as High risk and Emergency.
If symptoms indicate a moderate concern requiring timely attention but not immediate life-saving intervention, classify as Medium risk and Urgent.
If symptoms are mild and not indicative of a serious condition, classify as Low risk and Routine.

Respond STRICTLY in this format:

Risk Level:
Urgency:
Top 3 Conditions:
1.
2.
3.
Treatment:
1.
2.
"""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
inputs = {k: v.to(model.device) for k, v in inputs.items()}


outputs = model.generate(
    **inputs,
    max_new_tokens = 240,
    do_sample = False,
    pad_token_id = tokenizer.eos_token_id,
    use_cache = True
)

input_length = inputs["input_ids"].shape[1]
generated_tokens = outputs[0][input_length:]

decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

decoded_text = decoded.strip()

start_marker_for_structured = "Risk Level:"
start_index = decoded_text.find(start_marker_for_structured)

if start_index != -1:
    structured_output = decoded_text[start_index:].strip()
    print("Found 'Risk Level:' as starting marker for parsing.")
else:
    structured_output = decoded_text
    print("Warning: 'Risk Level:' marker not found at the expected start. Attempting to parse entire output.")


print("Diagnosis:")

risk_level_match = re.search(r"Risk Level:\s*(?:[^a-zA-Z0-9\s]*\s*)?(Low|Medium|High)(?:\s*[^a-zA-Z0-9\s]*)?", structured_output, re.IGNORECASE)
if risk_level_match:
    print(f"Risk Level: {risk_level_match.group(1).capitalize()}")
else:
    print("Risk Level: Not Found")

urgency_match = re.search(r"Urgency:\s*(?:[^a-zA-Z0-9\s]*\s*)?(Routine|Urgent|Emergency)(?:\s*[^a-zA-Z0-9\s]*)?", structured_output, re.IGNORECASE)
if urgency_match:
    print(f"Urgency: {urgency_match.group(1).capitalize()}")
else:
    print("Urgency: Not Found")

conditions_block_start_index = structured_output.find("Top 3 Conditions:")
treatment_block_start_index = structured_output.find("Treatment:")

if conditions_block_start_index != -1:
    end_of_conditions = treatment_block_start_index if treatment_block_start_index != -1 else len(structured_output)
    conditions_section = structured_output[conditions_block_start_index + len("Top 3 Conditions:"):end_of_conditions].strip()

    conditions = []
    for line in conditions_section.split('\n'):
        match = re.search(r"^\s*\d+\.\s*(.+)", line)
        if match and match.group(1).strip():
            conditions.append(match.group(1).strip())

    print("Highly probable conditions:")
    if conditions:
        for i, condition in enumerate(conditions[:3]):
            print(f"{i+1}. {condition}")
    else:
        print("Conditions: Not Found or incomplete.")
else:
    print("Top 3 Conditions: Not Found")

if treatment_block_start_index != -1:
    treatment_section_raw = structured_output[treatment_block_start_index + len("Treatment:"):].strip()
  
    end_markers = ["Next Steps:", "Follow-up:", "Patient Education:", "Medication:", "Referral:", "**My thinking process:**"]
    end_of_treatment_index = len(treatment_section_raw)
    for marker in end_markers:
        marker_index = treatment_section_raw.find(marker)
        if marker_index != -1 and marker_index < end_of_treatment_index:
            end_of_treatment_index = marker_index

    treatment_section_clean = treatment_section_raw[:end_of_treatment_index].strip()

    treatments = []
    for line in treatment_section_clean.split('\n'):
        match = re.search(r"^\s*\d+\.\s*(.+)", line)
        if match and match.group(1).strip():
            treatments.append(match.group(1).strip())

    print("Recommended course of action:")
    if treatments:
        for i, treatment in enumerate(treatments[:2]):
            print(f"{i+1}. {treatment}")
    else:
        print("Treatment: Not Found or incomplete.")
else:
    print("Treatment: Not Found") 

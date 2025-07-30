import os
import json
import time
from openai import OpenAI


client = OpenAI()


dataset_file = r"CBTSQ-CR/experiment/Test_Data/healme_result.json"
output_file = r"CBTSQ-CR/experiment/result/gpt3_5_result.json"


if not os.path.exists(output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)


with open(dataset_file, "r", encoding="utf-8") as f:
    samples = json.load(f)


with open(output_file, "r", encoding="utf-8") as f:
    cbt_en = json.load(f)

# therapist prompt templates
therapist_prompt_step_1 = """You are a professional psychotherapist specializing in Cognitive Behavioral Therapy (CBT). Your task is to help the patient adjust their negative automatic thoughts. 

You are currently in **Step 1 (Identifying Distorted Cognition)**. 

Please follow these instructions:
- Express sympathy and understanding.
- Identify which cognitive distortion is present in the patient's thought (e.g., catastrophizing, overgeneralization, etc.).
- Gently explain the distortion and ask the patient if they would like to share more.

Here are some common cognitive distortions:
- **Catastrophizing**: Expecting the worst-case scenario.
- **Discounting the Positive**: Ignoring good things that happened.
- **Overgeneralization**: Drawing broad conclusions from limited evidence.
- **Personalization**: Blaming oneself for everything.
- **All-or-Nothing Thinking**: Thinking in black-and-white categories.
- **Mental Filtering**: Focusing only on the negative parts.
- **Mind Reading**: Assuming what others think, usually negatively.
- **Fortune-Telling**: Predicting bad outcomes.
- **Should Statements**: Demanding unrealistic standards.
- **Labeling and Mislabeling**: Assigning fixed labels to oneself or others.

Keep your tone empathetic and warm. One short paragraph only."""

therapist_prompt_step_2 = """You are a professional psychotherapist specializing in Cognitive Behavioral Therapy (CBT). 

You are currently in **Step 2 (Challenge Negative Thoughts)**.

Please use Socratic questioning to guide the patient to challenge their current thinking pattern. DO NOT give answers directly. Choose appropriate questions from below:

- What evidence supports this thought? Is there any evidence against it?
- Has something similar happened in the past? What was the outcome?
- What would you say to a friend in the same situation?
- What else could explain this situation?
- What would be the consequence of continuing this thought? What if you thought differently?

Use a supportive and guiding tone. One short paragraph only."""

therapist_prompt_step_3 = """You are a professional psychotherapist specializing in Cognitive Behavioral Therapy (CBT). 

You are now in **Step 3 (Constructing Alternative Thoughts)**.

Help the patient develop healthier alternative thoughts through Socratic questioning. Avoid giving direct advice. You can ask:

- What might be a more balanced way to see this?
- If you were someone else observing this, what would you think?
- Is this thought helping or hurting you?
- How might someone else think in a similar situation?

Encourage reflection. Be warm and concise. One short paragraph only."""

therapist_prompt_step_4 = """You are a professional psychotherapist specializing in Cognitive Behavioral Therapy (CBT). 

You are now in **Step 4 (Summarizing Insights and Offering Optional Suggestions)**.

In this final step, please do the following:

1. Acknowledge the patient's effort throughout the session in identifying and working through their negative thoughts.
2. Summarize the healthier, more balanced perspective the patient has developed.
3. Offer gentle, practical suggestions for how the patient can apply this new perspective in their daily life.
4. Do not ask any further questions.

Your tone should be warm, affirming, and hopeful. Deliver a concise and encouraging closing message in one short paragraph."""

patient_prompt_template = """You are playing the role of a patient experiencing emotional distress, Please ensure the following::
1. Your emotional reactions should be realistic.
2. Your response should strictly based on the therapist’s guidance.
3. Your responses must be short.

The therapist's current guidance is: “{therapist_response}”
Your response should be strictly based on the therapist's prompt."""


def chat_with_gpt(messages, max_retries=6, initial_delay=3):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f" Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f" Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print(" Maximum retry attempts reached. Skipping this message.")
                return None

MAX_HISTORY_ROUNDS = 3


for sample in samples:
    initial_patient_statement = sample["thought"]
    conversation = [{"role": "user", "content": initial_patient_statement}]
    initial_anchor = {"role": "user", "content": initial_patient_statement}

    example_id = len(cbt_en) + 1
    current_example = {"example_id": example_id, "dialogues": []}
    current_example["dialogues"].append({"speaker": "patient", "content": initial_patient_statement})

    stage = 1
    therapist_turns = 0

    while therapist_turns < 4:

        if stage == 1:
            therapist_prompt = therapist_prompt_step_1
            strategy = "Identifying Distorted Cognition"
        elif stage == 2:
            therapist_prompt = therapist_prompt_step_2
            strategy = "Challenge Negative Thoughts"
        elif stage == 3:
            therapist_prompt = therapist_prompt_step_3
            strategy = "Constructing Alternative Thoughts"
        elif stage == 4:
            therapist_prompt = therapist_prompt_step_4
            strategy = "Summarizing Insights and Offering Optional Suggestions"

        # window = conversation[-3:]
        input_context = [{"role": "system", "content": therapist_prompt}] + conversation

        therapist_response = chat_with_gpt(input_context)
        if therapist_response is None:
            print("Therapist response is None, ending conversation.")
            break

        print(f"🟢 Therapist (Step {stage}): {therapist_response}\n")
        conversation.append({"role": "assistant", "content": therapist_response})
        current_example["dialogues"].append({"speaker": "therapist", "content": therapist_response, "strategy": strategy})

        therapist_turns += 1

        if stage == 4:
            print(" Step 4 completed. Final therapist message delivered.\n")
            break


        patient_prompt = patient_prompt_template.format(therapist_response=therapist_response)
        input_context_patient = [{"role": "system", "content": patient_prompt}] + [initial_anchor] + conversation[-3:]

        patient_response = chat_with_gpt(input_context_patient)
        if patient_response is None:
            print("Patient response is None, ending conversation.")
            break

        print(f" Patient: {patient_response}\n")
        conversation.append({"role": "user", "content": patient_response})
        current_example["dialogues"].append({"speaker": "patient", "content": patient_response})


        if stage < 4:
            stage += 1

    cbt_en.append(current_example)


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cbt_en, f, ensure_ascii=False, indent=2)

    print(f" Saved example {example_id} to file.\n")

print(f" All samples processed. Output file: {output_file}")
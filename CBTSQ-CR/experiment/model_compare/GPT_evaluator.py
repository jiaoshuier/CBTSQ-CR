import os
import json
from openai import OpenAI
from tqdm import tqdm

# Initialize client
client = OpenAI()

# Load baseline samples
with open(r"CBTSQ-CR/Data/SoCBTtalk_2000.json", "r",
          encoding="utf-8") as f:
    baseline_samples = json.load(f)


def format_dialogue(dialogue, speaker_map):
    text = ""
    for turn in dialogue:
        speaker = speaker_map.get(turn["speaker"].lower(), turn["speaker"])
        content = turn.get("content") or turn.get("text")
        text += f"{speaker}: {content}\n"
    return text


# Output file path
output_file = r"...\SoCBT_scores.json"
summary_file = r"...\SoCBT_dimension_avg.json"

# Initialize output files
if not os.path.exists(output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False)

# Load existing results
try:
    with open(output_file, "r", encoding="utf-8") as f:
        all_scores = json.load(f)
    processed_ids = {item["example_id"] for item in all_scores}
except:
    all_scores = []
    processed_ids = set()

dimension_scores = {
    "Emotional Support": [],
    "Dialogue Naturalness": [],
    "Restructuring Effectiveness": [],
    "Therapist Adaptability": [],
    "Guidance Quality": []
}

for idx, base_data in tqdm(enumerate(baseline_samples), total=len(baseline_samples)):
    example_id = base_data.get("example_id", idx)

    if example_id in processed_ids:
        continue

    dialogue = format_dialogue(base_data["dialogues"], {"patient": "Patient", "therapist": "Therapist"})

    prompt = f"""
    You are an expert in evaluating counseling dialogues. When scoring, please focus primarily on the **therapist’s responses**, as the evaluation is intended to assess the quality of the therapist’s intervention. Please rate the following counseling dialogues based on the five dimensions below. Each dimension should be scored on a scale from 1 to 7 according to the provided criteria. Please provide **actual scores**.

    (1). Emotional Support

    1 points: Almost no response to the patient's emotions; language is cold and neglectful.
    2 points: Some emotional acknowledgment, but appears formal or mismatched, lacking empathy.
    3 points: Shows some emotional understanding and comfort, but not deep or somewhat repetitive.
    4 points: Demonstrates clear emotional empathy and reassurance; responses align with the patient's feelings.
    5 points: Fully empathetic, with nuanced and warm responses that alleviate negative emotions.
    6 points: Exceptionally sensitive to emotional nuance, offers comforting, compassionate responses that feel profoundly supportive.
    7 points: Provides exceptional emotional depth, deeply validating and comforting, helping to normalize and process the patient's emotional experience.

    (2). Dialogue Naturalness

    1 points: Language is rigid, formulaic, or logically inconsistent, with noticeable AI traits or incoherence.
    2 points: Sentences are coherent but stiff, lacking smooth transitions, not resembling a real conversation.
    3 points: Mostly natural, but some transitions are awkward or language feels somewhat scripted.
    4 points: Fluent and natural dialogue, with human-like expressions and appropriate emotional tone.
    5 points: Indistinguishable from a real human conversation, with smooth flow and a warm, engaging tone.
    6 points: The dialogue feels deeply personalized, with seamless transitions and an engaging, highly fluid conversation style.
    7 points: The conversation has the nuance and spontaneity of a real human interaction, creating a sense of true connection.

    (3). Restructuring Effectiveness

    1 points: No cognitive intervention; remains at emotional support or generic advice.
    2 points: Attempts guidance but is confusing; fails to identify core beliefs or negative thoughts.
    3 points: Some cognitive intervention, identifying some distortions, but restructuring is incomplete.
    4 points: Clearly points out irrational thoughts and guides toward alternative interpretations.
    5 points: Deeply identifies irrational cognitions and achieves clear restructuring, fostering positive change.
    6 points: Applies advanced cognitive restructuring techniques, guiding the patient through nuanced thought patterns and challenging deeply rooted beliefs.
    7 points: Promotes profound cognitive shifts, fostering long-term changes in perspective and behavior by deeply challenging and reframing core beliefs.

    (4). Therapist Adaptability

    1 points: Entirely template-based responses, ignoring the patient's input; irrelevant replies.
    2 points: Some responses are relevant, but mostly generic or overly templated.
    3 points: Replies are contextually appropriate but lack depth or specific understanding.
    4 points: Fairly personalized, addressing specific situations with some adaptability.
    5 points: Highly tailored to the patient's issues, with individualized language and deep comprehension.
    6 points: Exceptionally tailored responses, with fine-tuned adaptability to the patient's needs, fostering a sense of being truly understood.
    7 points: The therapist's responses feel bespoke, addressing not just the content but also the emotional nuances of the patient's concerns.

    (5). Guidance Quality

    1 points: No questions or guidance; conversation lacks structure or direction.
    2 points: Attempts guidance, but questions are vague or fail to provoke thought.
    3 points: Some guiding questions, providing basic direction but not systematic.
    4 points: Clear and purposeful guidance, eliciting some reflection or expression.
    5 points: Systematically promoting self-awareness and active engagement.
    6 points: Provides skillful guidance, helping the patient deeply reflect and make meaningful connections to their beliefs and emotions.
    7 points: Offers transformative guidance, facilitating a deep, introspective journey and encouraging self-discovery and lasting insight.

    Please provide the output in the following JSON format:  
    {{
      "Emotional Support": x,
      "Dialogue Naturalness": x,
      "Restructuring Effectiveness": x,
      "Therapist Adaptability": x,
      "Guidance Quality": x
    }}

    The dialogues is as follows:
    {dialogue}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating psychotherapy dialogue."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        score = json.loads(response_text)

        values = [
            score["Emotional Support"],
            score["Dialogue Naturalness"],
            score["Restructuring Effectiveness"],
            score["Therapist Adaptability"],
            score["Guidance Quality"]
        ]
        score["average_score"] = round(sum(values) / len(values), 2)

        dimension_scores["Emotional Support"].append(score["Emotional Support"])
        dimension_scores["Dialogue Naturalness"].append(score["Dialogue Naturalness"])
        dimension_scores["Restructuring Effectiveness"].append(score["Restructuring Effectiveness"])
        dimension_scores["Therapist Adaptability"].append(score["Therapist Adaptability"])
        dimension_scores["Guidance Quality"].append(score["Guidance Quality"])

    except Exception as e:
        score = {"error": str(e), "raw": response_text if 'response_text' in locals() else None}

    result = {
        "example_id": example_id,
        "dialogue": dialogue,
        "evaluation": score
    }

    all_scores.append(result)
    processed_ids.add(example_id)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2, ensure_ascii=False)

dimension_avg = {
    "Emotional Support_avg": round(
        sum(dimension_scores["Emotional Support"]) / len(dimension_scores["Emotional Support"]), 2),
    "dialogue_naturalness_avg": round(
        sum(dimension_scores["Dialogue Naturalness"]) / len(dimension_scores["Dialogue Naturalness"]), 2),
    "Restructuring Effectiveness_avg": round(
        sum(dimension_scores["Restructuring Effectiveness"]) / len(dimension_scores["Restructuring Effectiveness"]), 2),
    "Therapist Adaptability_avg": round(
        sum(dimension_scores["Therapist Adaptability"]) / len(dimension_scores["Therapist Adaptability"]), 2),
    "Guidance Quality_avg": round(
        sum(dimension_scores["Guidance Quality"]) / len(dimension_scores["Guidance Quality"]), 2),
    "total_samples": len(all_scores)
}

with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(dimension_avg, f, indent=2, ensure_ascii=False)

print(json.dumps(dimension_avg, indent=2, ensure_ascii=False))
INFERENCE_FINAL_PROMPT = (
    "{task_description}\n\n"
    # "You are a professional image analysis assistant, "
    # "highly skilled in reasoning step by step to refine and correct descriptions of traffic scenes based on actual images and task objectives."
    "<Note>\n\n"
    "Below is a rough description of the image, which may have inaccuracies or omissions. Please reason through the problem"
    # " step by step"
    " and refine the answer.\n\n"
    # "Rough Description:\n"
    "<Inaccurate or incomplete description>\n"
    "{rough_description}\n\n"
    # "Your refined description:\n"
    # "Your description:\n"
    # "Your refined description:\n"
    # "Steps to Follow:\n"
    # " - Identify inaccuracies or incomplete details in the rough description.\n"
    # " - Cross-check the described objects, attributes, and relationships against the actual image (imaginary reasoning for the task).\n"
    # " - Add missing information and correct errors while considering the task objective.\n"
    # " - Organize the refined description in a logical and concise manner.\n"
    # " - Output the final refined description starting with <answer>:\n"
)

INFERENCE_FEW_SHOT_PROMPT = [
    {
        "role": "user",
        "prompt": (
            "<image>\nYou are a professional image analysis assistant give me a rough description of the image."
        ),
    },
    {"role": "assistant", "prompt": "{rough_description}"},
    {
        "role": "user",
        "prompt": "Now answer the below questions based on the rough description. Please remove all bbox information.\n\n{task_description}",
    },
]


def fillin_fewshot(rough_description, task_description):
    template = [{**x} for x in INFERENCE_FEW_SHOT_PROMPT]
    template[1]["prompt"] = template[1]["prompt"].format(
        rough_description=rough_description
    )
    template[2]["prompt"] = template[2]["prompt"].format(
        task_description=task_description
    )

    return template

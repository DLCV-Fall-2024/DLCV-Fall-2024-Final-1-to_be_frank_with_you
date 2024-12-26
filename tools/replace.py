import random
import re

REPLACEMENT_SETS = {
    "color": [
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "white",
        "black",
        "gray",
        "silver",
        "brown",
        "purple",
    ],
    "vehicle": [
        "car",
        "truck",
        "bus",
        "motorcycle",
        "scooter",
        "bicycle",
        "ambulance",
        "firetruck",
        "van",
        "pickup truck",
    ],
    "vulnerable_road_user": [
        "pedestrian",
        "cyclist",
        "motorcyclist",
        "child crossing",
        "elderly pedestrian",
        "person with a stroller",
    ],
    "traffic_sign": [
        "stop sign",
        "yield sign",
        "no parking sign",
        "speed limit sign",
        "directional sign",
        "warning sign",
        "pedestrian crossing sign",
        "construction warning sign",
    ],
    "traffic_light": [
        "red light",
        "green light",
        "yellow light",
        "flashing red light",
        "flashing yellow light",
        "broken traffic light",
    ],
    "miscellaneous": [
        "debris",
        "trash bin",
        "roadkill",
        "stray animal",
        "cone",
        "barrier",
        "dustbin",
        "shopping cart",
        "construction equipment",
        "fallen tree branch",
    ],
    "direction": [
        "stationary",
        "moving forward",
        "reversing",
        "turning left",
        "turning right",
        "approaching",
        "moving away",
        "crossing",
    ],
    "location": [
        "left",
        "right",
        "center",
        "center-left",
        "center-right",
        "ahead",
        "behind",
    ],
    "placement": [
        "on the sidewalk",
        "on the crosswalk",
        "in the lane",
        "on the shoulder",
    ],
}


def random_replace_and_highlight(
    text, replacement_sets=REPLACEMENT_SETS, replace_prob=0.3
):
    """
    Randomly replace original attributes in the text and highlight the replacements.

    Args:
        text (str): Ground truth text containing original attributes.
        replacement_sets (dict): A dictionary of attribute types and their possible replacement values.
                                 Example: {
                                     "color": ["red", "blue", "green"],
                                     "vehicle": ["car", "truck", "bike"],
                                     "location": ["left", "right", "center"]
                                 }
        replace_prob (float): Probability (0-1) of replacing an attribute in the text.

    Returns:
        str: Modified text with replacements highlighted.
    """

    def replace_match(match, color=False):
        attribute_type = match.lastgroup  # Get the named group (e.g., "color")
        original_value = match.group(0)  # The original matched value
        if attribute_type in replacement_sets and random.random() < replace_prob:
            new_value = random.choice(replacement_sets[attribute_type])
            # Highlight the replacement
            if color:
                return f"\033[1;31m{new_value}\033[0m"
            return new_value
        return original_value  # Keep the original if not replacing

    # Regular expressions for identifying attributes
    patterns = {
        "color": r"\b(red|green|blue|yellow|orange|white|black|gray|silver|brown|purple)\b",
        "vehicle": r"\b(car|truck|bus|motorcycle|scooter|bicycle|ambulance|firetruck|van|pickup truck)\b",
        "vulnerable_road_user": r"\b(pedestrian|cyclist|motorcyclist|child crossing|elderly pedestrian|person with a stroller)\b",
        "traffic_sign": r"\b(stop sign|yield sign|no parking sign|speed limit sign|directional sign|warning sign|pedestrian crossing sign|construction warning sign)\b",
        "traffic_light": r"\b(red light|green light|yellow light|flashing red light|flashing yellow light|broken traffic light)\b",
        "miscellaneous": r"\b(debris|trash bin|roadkill|stray animal|cone|barrier|dustbin|shopping cart|construction equipment|fallen tree branch)\b",
        "direction": r"\b(stationary|moving forward|reversing|turning left|turning right|approaching|moving away|crossing)\b",
        "location": r"\b(left|right|center|center-right|center-left|ahead|behind)\b",
        "placement": r"\b(on the sidewalk|on the crosswalk|in the lane|on the shoulder)\b",
    }

    # Combine patterns into a single regex with named groups
    combined_pattern = re.compile(
        "|".join(f"(?P<{key}>{pattern})" for key, pattern in patterns.items())
    )

    # Replace matches using the combined pattern
    modified_text = combined_pattern.sub(replace_match, text)

    return modified_text


if __name__ == "__main__":
    import json
    import pickle

    task = "train"
    config_path = f"data/{task}/config.json"
    store_path = f"data/{task}/config_aug.json"

    with open(config_path, "r") as f:
        original = json.load(f)

    for item in original["data"]:
        gt = item["gt"]
        modified_gt = random_replace_and_highlight(gt, replace_prob=0.7)
        item["features"]["object_info"] = modified_gt

    with open(store_path, "w") as f:
        json.dump(original, f)

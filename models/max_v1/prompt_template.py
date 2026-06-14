MAX_SYSTEM_PREFIX = "<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"
MAX_PROMPT = "<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"
MAX_CHAT_SEP = "<|im_end|>\n"
MAX_SUFFIX = "<|im_end|>"

# CARLA command index → B2DVL-style navigation intent text.
# Index 0 ("void") defaults to "follow the road".
COMMAND_TO_TEXT = [
    "follow the road",                # 0: void
    "turn left at the intersection",  # 1: turn left
    "turns right at the intersection",# 2: turn right
    "drive straight at the intersection",  # 3: go straight
    "follow the road",                # 4: follow lane
    "do a lane change to the left",   # 5: change lane to left
    "do a lane change to the right",  # 6: change lane to right
]

MAX_DEFAULT_SYSTEM = (
    "You are a responsible driver, you need to follow the rules of the road and stay safe as efficiently as possible."
    "Every 0.5s, the coordinates are represented by [x, y], where x is the front and y is the left and right direction,"
    "and the trajectory of the future 4s is output in the format [x1, y1], [x2, y2],..., [x8, y8]]."
)

B2DVL_IMAGE_DESC = (
    "The two concatenated images below are from "
    "all cameras attached to the ego vehicle on current frame."
)

B2DVL_WAYPOINT_QUESTION = (
    "Please predict the waypoint tokens for the next 4 seconds, "
    "with one set every 0.5 seconds, "
    "for a total of 8 sets of relative displacements."
)

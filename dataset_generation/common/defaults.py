DEFAULT_IMAGE_RELPATH = "patch_tif/0.tif"
DEFAULT_MASK_RELPATH = "patch_tif/0_edit_poly.tif"
DEFAULT_LANE_RELPATH = "label_check_crop/Lane.geojson"
DEFAULT_INTERSECTION_RELPATH = "label_check_crop/Intersection.geojson"

DEFAULT_STAGE_A_PROMPT_TEMPLATE = """<image>
Please construct the complete road-structure line map in the current satellite patch."""

DEFAULT_STAGE_A_SYSTEM_PROMPT = (
    "You are a road-structure reconstruction assistant for satellite-image patches.\n"
    "Predict the complete patch-local line map from the current image.\n"
    "The output JSON schema is {\"lines\": [...]} .\n"
    "Each line must stay in patch-local UV coordinates.\n"
    "Use patch-local integer UV coordinates where one pixel equals one unit.\n"
    "Use category lane_line for roads and intersection_polygon for intersections.\n"
    "Return only valid JSON and no extra text."
)

DEFAULT_STAGE_B_PROMPT_TEMPLATE = """<image>
Please construct the road-structure line map in the current patch.
The previous state contains cut traces passed from already processed neighboring patches.
Continue those traces when appropriate and also predict all owned line segments for the current patch.
Previous state:
{state_json}"""

DEFAULT_STAGE_B_SYSTEM_PROMPT = (
    "You are a road-structure reconstruction assistant for satellite-image patches.\n"
    "Use the image and the previous line-map state to predict the current patch.\n"
    "The previous state contains cut traces from already processed neighboring patches.\n"
    "Preserve cross-patch continuity whenever those traces enter the current patch.\n"
    "The output JSON schema is {\"lines\": [...]} .\n"
    "Each line must stay in patch-local UV coordinates.\n"
    "Use patch-local integer UV coordinates where one pixel equals one unit.\n"
    "Use category lane_line for roads and intersection_polygon for intersections.\n"
    "Return only valid JSON and no markdown fences."
)
"""Static list of the 90 Food-101 class names the vision checkpoint was trained on.

Generated once via `list(load_food101_splits(DATA_DIR)[0]["class"])` (vision/dataset.py)
and committed as a plain list, so `agent/` never needs the raw (gitignored) `data/`
directory present at runtime -- only the checkpoint file. Order matters: it must match
the label indices `vision/checkpoints/config4_full.pt` was trained with (alphabetical,
as produced by `make_class_to_idx`).
"""

CLASS_NAMES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "club_sandwich",
    "creme_brulee", "croque_madame", "cup_cakes", "deviled_eggs", "donuts",
    "dumplings", "edamame", "eggs_benedict", "falafel", "fish_and_chips",
    "foie_gras", "french_fries", "french_onion_soup", "french_toast", "fried_calamari",
    "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad",
    "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
    "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream",
    "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons",
    "miso_soup", "nachos", "omelette", "onion_rings", "pad_thai",
    "paella", "pancakes", "panna_cotta", "peking_duck", "pho",
    "pizza", "poutine", "prime_rib", "ravioli", "red_velvet_cake",
    "risotto", "samosa", "sashimi", "seaweed_salad", "spaghetti_bolognese",
    "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake", "sushi",
    "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles",
]

assert len(CLASS_NAMES) == 90

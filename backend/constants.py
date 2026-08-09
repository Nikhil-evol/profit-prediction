SUB_CATEGORIES = {
    "Technology": [
        "Accessories",
        "Phones",
        "Copiers",
        "Machines"
    ],
    "Furniture": [
        "Tables",
        "Bookcases",
        "Chairs",
        "Furnishings"
    ],
    "Office Supplies": [
        "Paper",
        "Art",
        "Storage",
        "Appliances",
        "Supplies",
        "Envelopes",
        "Fasteners",
        "Labels",
        "Binders"
    ]
}

CATEGORIES = ["Technology", "Furniture", "Office Supplies"]

SEGMENTS = [
    "Consumer",
    "Corporate",
    "Home Office"
]



REGIONS = ['West',
           'East',
           'South',
           'Central', 
           'Africa', 
           'Central Asia',
           'North Asia',
           'Caribbean',
           'North',
           'EMEA',
           'Oceania',
           'Southeast Asia',
           'Canada']

ALL_SUB_CATEGORIES = [
    item
    for subcats in SUB_CATEGORIES.values()
    for item in subcats
]
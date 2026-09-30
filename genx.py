import random


rune_values = {
    "Opulent": { "LootMult" : 1.35, "Avoid" :False },
    "Bond":    { "LootMult" : 1.25, "Avoid" :False },
    "Power":   { "LootMult" : 1.30, "Avoid" :False },
    "Time":    { "LootMult" : 1.18, "Avoid" :False },
    "Death":   { "LootMult" : 1.15, "Avoid" :False },
    "Rebirth": { "LootMult" : 1.10, "Avoid" :False },
    "Wisdom":  { "LootMult" : 0.95, "Avoid" :True },
    "Oath":    { "LootMult" : 0.75, "Avoid" :True },
    "Bait":    { "LootMult" : 1.00, "Avoid" :True },

}

def get_rune_val(rune_type:str):
    if rune_type not in rune_values:
        return 1
    else:
        return rune_values[rune_type]["LootMult"]


rune_names = ["Opulent",
"Bond",
"Power",
"Time",
"Death",
"Rebirth",
"Wisdom",
"Oath",
"Bait"    ]

"""
    "num_slots": 3,
    "crown_runes": 1,
    "crown_slots": [0],
    "rune_type":
    {
        0: "Time",
        1: "",
        2: 1,
        3: 1,
        4: 1,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0
    },
"rune_values":
{
    0: 1.0,
    1: 1.1,
    2: 1.0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0
}"""


def main():
    ret = []
    for i in range(6):
        num_slots = random.randint(3,7)
        crown_runes = random.randint(1,2)
        crown_slots = [random.randint(0,num_slots) for item in range(crown_runes)]
        crown_slots.sort()
        rune_type = {item : rune_names[random.randint(0,len(rune_names) - 1)]for item in range(num_slots)}
        num_runes = len(rune_type)
        if num_runes < 10 :
            for j in range(num_runes,10):
                #j = num_runes
                rune_type[j] = "Lightning"
        rune_value = {}
        for a in rune_type:
            #print(rune_type[a])
            rune_value[a] = get_rune_val(rune_type[a])

        #print(rune_value)
        txt = {}
        txt["num_slots"] = num_slots
        txt["crown_runes"] = crown_runes
        txt["crown_slots"] = crown_slots
        txt["rune_type"] = rune_type
        txt["rune_values"] = rune_value
        #txt["num_slots"] = num_slots

        print(txt)


if __name__ == "__main__":
    main()


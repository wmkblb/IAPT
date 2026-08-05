from statistics import mean

results = {
    "imagenet": [
        {"base": 78.6100, "new": 72.5500, "hm": 75.4585},
        {"base": 78.9300, "new": 72.2600, "hm": 75.4479},
        {"base": 79.0200, "new": 72.1000, "hm": 75.4016},
    ],

    "caltech101": [
        {"base": 98.3900, "new": 95.7400, "hm": 97.0469},
        {"base": 98.7100, "new": 95.3100, "hm": 96.9802},
        {"base": 98.8400, "new": 95.8500, "hm": 97.3220},
    ],

    "oxford_pets": [
        {"base": 95.7500, "new": 97.5400, "hm": 96.6367},
        {"base": 95.1600, "new": 97.3200, "hm": 96.2279},
        {"base": 95.8500, "new": 97.3200, "hm": 96.5794},
    ],

    "stanford_cars": [
        {"base": 82.6100, "new": 74.8900, "hm": 78.5608},
        {"base": 83.2100, "new": 74.8900, "hm": 78.8311},
        {"base": 81.6600, "new": 73.8300, "hm": 77.5479},
    ],

    "oxford_flowers": [
        {"base": 98.7700, "new": 75.6700, "hm": 85.6905},
        {"base": 99.0500, "new": 75.8900, "hm": 85.9369},
        {"base": 98.4800, "new": 75.8900, "hm": 85.7217},
    ],

    "food101": [
        {"base": 90.6500, "new": 91.6800, "hm": 91.1621},
        {"base": 90.6300, "new": 91.6900, "hm": 91.1569},
        {"base": 90.8100, "new": 92.0700, "hm": 91.4357},
    ],

    "fgvc_aircraft": [
        {"base": 51.4400, "new": 40.0100, "hm": 45.0107},
        {"base": 48.5600, "new": 39.1700, "hm": 43.3625},
        {"base": 47.0600, "new": 38.6300, "hm": 42.4303},
    ],

    "sun397": [
        {"base": 82.9900, "new": 80.0800, "hm": 81.5090},
        {"base": 82.9800, "new": 79.9000, "hm": 81.4109},
        {"base": 83.2600, "new": 79.7100, "hm": 81.4463},
    ],

    "dtd": [
        {"base": 85.0700, "new": 67.0300, "hm": 74.9802},
        {"base": 85.4200, "new": 68.4800, "hm": 76.0177},
        {"base": 84.0300, "new": 66.6700, "hm": 74.3501},
    ],

    "eurosat": [
        {"base": 96.0500, "new": 70.1500, "hm": 81.0819},
        {"base": 95.1900, "new": 70.3300, "hm": 80.8931},
        {"base": 94.6400, "new": 77.6900, "hm": 85.3314},
    ],

    "ucf101": [
        {"base": 88.0000, "new": 82.6400, "hm": 85.2358},
        {"base": 88.7800, "new": 82.7500, "hm": 85.6590},
        {"base": 87.4900, "new": 83.2900, "hm": 85.3384},
    ],
}

dataset_avgs = {}

for dataset, values in results.items():
    dataset_avgs[dataset] = {
        "base": mean([x["base"] for x in values]),
        "new": mean([x["new"] for x in values]),
        "hm": mean([x["hm"] for x in values]),
    }

overall_base = mean([v["base"] for v in dataset_avgs.values()])
overall_new = mean([v["new"] for v in dataset_avgs.values()])
overall_hm = mean([v["hm"] for v in dataset_avgs.values()])

print("| Dataset | Avg Base | Avg New | Avg HM |")
print("|---|---:|---:|---:|")
print(f"| Average | {overall_base:.2f} | {overall_new:.2f} | {overall_hm:.2f} |")

for dataset, avg in dataset_avgs.items():
    print(f"| {dataset} | {avg['base']:.2f} | {avg['new']:.2f} | {avg['hm']:.2f} |")
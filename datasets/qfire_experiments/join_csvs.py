import pandas as pd

with open("C:\Q-Fire_experiments\Q-FIRE_dataset\Q-FIRE-info-extracted.csv", "w") as excsv:
    with open("C:\Q-Fire_experiments\Q-FIRE_dataset\Q-FIRE-info-windows.csv", 'r') as wincsv:
        for line in wincsv:
            line = line.replace('\\', '/')
            excsv.write(line)
        
dataset1 = pd.read_csv("C:\Q-Fire_experiments\Q-FIRE_dataset\Q-FIRE-info-mbark.csv")
dataset2 = pd.read_csv("C:\Q-Fire_experiments\Q-FIRE_dataset\Q-FIRE-info-extracted.csv")

full_dataset = pd.concat([dataset1, dataset2])
full_dataset.to_csv('C:\Q-Fire_experiments\Q-FIRE_dataset\Q-FIRE-info-full.csv')
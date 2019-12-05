import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
d = pd.DataFrame()
for x in soft:
    d = d.append({"task": "DEP","val":x*100,"label":"soft"},ignore_index=True)
for x in soft:
    d = d.append({"task": "DEP","val":x*100,"label":"hard"},ignore_index=True)
for x in ner_soft:
    d = d.append({"task": "NER","val":x,"label": "soft"},ignore_index=True)
    d = d.append({"task": "NER","val":x,"label": "hard"},ignore_index=True)
for x in ner_soft:
    d = d.append({"task": "NER","val":x-3,"label": "warmup"},ignore_index=True)
    d = d.append({"task": "NER","val":x-7,"label": "nowarmup"},ignore_index=True)

def draw_violinplot(dataframe):
    plt.title("Variations")
    ax = sns.violinplot(x = "task", y= "val",vertical=True,data=d,inner="quartile",hue="label");
    ax.set_ylabel("F-1 score");
    ax.set_xlabel("");
draw_violinplot(d)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dunlin.utils as ut
import dunlin.utils_plot.keywords as upk

plt.close('all')
plt.ion()

def plot_bar(ax, groups, ):
    x     = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    
    
    div = len(groups)
    
    rects1 = ax.bar(x - width/div, men_means, width, label='Men')
    rects2 = ax.bar(x + width/div, women_means, width, label='Women')

    
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

fig = plt.figure()
AX_ = [fig.add_subplot(2, 2, i+1) for i in range(4)]
ax = AX_[0]


labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

groups = {'men': dict(zip(labels, men_means)),
          'women': dict(zip(labels, women_means)),
          }
df = pd.DataFrame(groups)

print(df)
# for g, m, w in zip(labels, men_means, women_means):
#     groups[g] = {'men': m, 'women': w}
    


plot_bar(ax, groups)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')


fig.tight_layout()




from matplotlib_venn import venn2, venn2_circles
from matplotlib import pyplot as plt

plt.figure(figsize=(4, 3))

# the subset sizes and set labels
venn = venn2(subsets=(0.3, 0.4, 0.1), set_labels=('X', 'Y'))
venn2_circles(subsets=(0.3, 0.4, 0.1), linestyle='solid', linewidth=1, color='black')

# circle color to white and remove labels
venn.get_patch_by_id('10').set_color('white')
venn.get_patch_by_id('01').set_color('white')
venn.get_patch_by_id('11').set_color('white')
#venn.get_label_by_id('10').set_text('')
#venn.get_label_by_id('01').set_text('')
#venn.get_label_by_id('11').set_text('')

# diplay U
plt.text(-0.6, 0.58, 'U', ha='center', va='center', color='black', fontsize=14)

# change font size of labels and position
#for label in venn.set_labels:
#    label.set_fontsize(14)
#    label.set_y(label.get_position()[1] + 0.965)

# set individually
venn.set_labels[0].set_y(venn.set_labels[0].get_position()[1] + 0.95)
venn.set_labels[1].set_y(venn.set_labels[1].get_position()[1] + 1.005)
    
# set linewidth
plt.gca().spines['top'].set_linewidth(1)
plt.gca().spines['right'].set_linewidth(1) 
plt.gca().spines['bottom'].set_linewidth(1)
plt.gca().spines['left'].set_linewidth(1)

# display figure
plt.tight_layout()
plt.axis('on')
plt.show()
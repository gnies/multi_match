import numpy as np
import multi_match 
import matplotlib.pyplot as plt

np.random.seed(2)
ns = [11, 23, 13, 9]
max_dist = 0.1
colors = ["red", "green", "blue", "yellow"]
point_lst = [np.random.random(size= (n_j, 2)) for n_j in ns]
match = multi_match.Multi_Matching(point_lst, max_dist=max_dist, method="pairwise")
# And count the number of different objects in the image:
num_obj = match.count_objects()
for key, value in num_obj.items():
    print(key, ' : ', value)
# the matches can be plotted
match.plot_match(channel_colors=colors, circle_alpha=0.5)
plt.axis()
plt.show()

##### object count over matching range
maxdist = 0.2
# set a range of maximal matching distance values
maxdist_range = np.linspace(0, maxdist, num=21)
range_match = multi_match.Multi_Matching_Over_Range(point_lst, maxdist_range, method="pairwise")
range_match.plot_match()
W = range_match.count_objects()
print(W)
range_match.plot_number_of_objects()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

for m in ['d', '+', '|']:

    for i in range(5):
        a1, a2  = np.random.random(2)
        angle = np.random.choice([180, 45, 90, 35])

        # make a markerstyle class instance and modify its transform prop
        t = mpl.markers.MarkerStyle(marker=m)
        t._transform = t.get_transform().rotate_deg(angle)
        plt.scatter((a1), (a2), marker=t, s=100)

plt.show()
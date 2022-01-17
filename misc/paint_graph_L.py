import numpy as np
import matplotlib.pyplot as plt

def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)


fig = plt.figure()
adjustFigAspect(fig,aspect=1.75)
ax = fig.add_subplot(111)


y =  [32.51, 72.04, 74.69, 87.01, 80.04, 72.01 ,80.25,83.35]
y2 = [39.05, 76.86, 84.19, 85.35, 78.46, 88.06, 78.41, 79.74 ]
y3 = [10.39, 71.4,  87.56,89.23, 73.6, 85.75, 81.66,74.02 ]
x=np.arange(1,len(y)+1)

ax.plot(x,y, marker="o", label= 'S1')
ax.plot(x,y2, marker="o", label = 'S2')
ax.plot(x,y3, marker="o", label = 'S3')
# plt.xlim(1,len(y))
# plt.ylim(0,1.5)

# ax.xticks(x,  [str(i) for i in x])  # Set text labels.
#
plt.xlabel("L")
plt.ylabel("V-measure")
# plt.title("CDF for discrete distribution")
plt.legend()

# ax.show()
plt.savefig('L_ablation_study.pdf')
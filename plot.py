# importing the required module
import matplotlib.pyplot as plt
import numpy as np
 
# x axis values
# corresponding y axis values
y = [0.52,0.68,0.56,0.56,0.66,0.68,0.7,0.62, 0.76,0.68, 0.6, 0.55, 0.7, 0.65, 0.9, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.65, 0.85, 0.8, 0.65, 0.6, 0.8]
x = [6*i for i in range(len(y))]
 
# plotting the points
plt.plot(x, y)

ax=plt.gca()
# adjust the y axis scale.
plt.ylim(ymin=0, ymax=1)
plt.xlim(xmin=0, xmax=500)
ax.locator_params('y', nbins=20)
plt.axhline(y=0.5, color='r', linestyle='-', label='baseline')
def running_mean(N):
    return np.convolve(y, np.ones(N) / float(N), 'valid')
y_ = [0.5]

for i in range(1,len(y)):
    #RUNNING AVERAGE
    y_.append(running_mean(i)[0])
plt.plot(x,y_, label = 'running average')
plt.legend(loc="upper right", borderaxespad=0)
 
# naming the x axis
plt.xlabel('MCTS games simulated')
# naming the y axis
plt.ylabel('winrate')
#print(np.convolve(y, np.ones(len(y))/len(y), mode='valid'))
# giving a title to my graph
plt.title('Winrate vs random')

# function to show the plot
#print(y_)
plt.show()
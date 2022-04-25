# importing the required module
import matplotlib.pyplot as plt
import numpy as np
import pickle


# with open('losses_per_epoch_iter2.pkl', 'rb') as f:
#     data = pickle.load(f)
#     plt.scatter([i for i in range(len(data))],data)
# x axis values
# corresponding y axis values
y = [0.5, 0.64, 0.78, 0.78, 0.82, 0.82, 0.8, 0.86, 0.82, 0.8, 0.84, 0.86, 0.78, 0.78, 0.74, 0.86, 0.86, 0.82, 0.88, 0.98, 0.84, 0.84, 
    0.82, 0.8, 0.9, 0.76, 0.9, 0.84, 0.84, 0.8]
x = [252*i for i in range(len(y))]

y1 = [0.5, 0.8, 0.78, 0.92, 0.86]
x1 = [252*i for i in range(len(y1))]

 
# plotting the points
plt.plot(x, y)
plt.plot(x1,y1)
#plt.scatter(x,y)

ax=plt.gca()
# adjust the y axis scale.
plt.ylim(ymin=0, ymax=1)
plt.xlim(xmin=0, xmax=10000)
ax.locator_params('y', nbins=20)
plt.axhline(y=0.5, color='r', linestyle='-', label='baseline')
def running_mean(N, x):
    return np.convolve(x, np.ones(N) / float(N), 'valid')

y_ = [0.5]


for i in range(1,len(y)):
    #RUNNING AVERAGE
    y_.append(running_mean(i,y)[0])
plt.plot(x,y_, label = 'trailing average', color = 'green')
plt.legend(bbox_to_anchor=(0.9,0.95), loc="lower center", borderaxespad=0)
#plt.subplots_adjust(right=0.75)   

y1_ = [0.5]

for i in range(1, len(y1)):
    y1_.append(running_mean(i,y1)[0])
plt.plot(x1,y1_, label = 'trailing average', color = 'black')
plt.legend(bbox_to_anchor=(0.9,0.95), loc="lower center", borderaxespad=0)

 
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
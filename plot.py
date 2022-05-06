# importing the required module
import matplotlib.pyplot as plt
import numpy as np
import pickle


# with open('losses_per_epoch_iter2.pkl', 'rb') as f:
#     data = pickle.load(f)
#     plt.scatter([i for i in range(len(data))],data)
# x axis values
# corresponding y axis values
#y = [0.64, 0.78, 0.78, 0.82, 0.82, 0.8, 0.86, 0.82, 0.8, 0.84, 0.86, 0.78, 0.78, 0.74, 0.86, 0.86, 0.82, 0.88, 0.98, 0.84, 0.84, 
#   0.82, 0.8, 0.9, 0.76, 0.9, 0.84, 0.84, 0.8]
#x = [252*i for i in range(len(y))]

#y1 = [0.9, 0.95, 0.65, 0.75, 0.9, 0.85,0.65, 0.8, 0.7, 0.9, 0.9, 1, 0.9, 0.7, 0.95, 0.8, 0.65, 0.65, 0.8, 0.95, 0.85, 0.85, 0.8, 0.75, 1, 0.85]
#x1 = [252*i for i in range(len(y1))]
#y1_add = [0.8125, 0.927083333334, 0.916666666666]
#x1_add = [672*(i+1)+(252*len(x1)) for i in range(len(y1_add))]
#x1.extend(x1_add)
#y1.extend(y1_add)
it3 = [0.5, 0.84375,0.84375,0.84375, 0.885416666, 0.9375, 0.91666666, 0.90625, 0.95833333, 0.947916666666, 0.95833333333, 0.875, 0.8645833333334, 0.90625, 0.8541666666666666]
it3 = [i*100 for i in it3]
x1 = [672/1000*i for i in range(len(it3))]

it2 = [0.5, 0.854166666, 0.80208333333334, 0.8854166666, 0.8958333333, 0.8229166666666666, 0.84375]
it1 = [0.5, 0.739583333334, 0.6770833333333334, 0.7604166666666666, 0.7916666666666666, 0.7916666666666666]
it2 = [i*100 for i in it2]
x2 = [672/1000*i for i in range(len(it2))]
it1 = [i*100 for i in it1]
x = [672/1000*i for i in range(len(it1))]
#cumsum_vec_1 = np.cumsum(np.insert(y1, 0, 0)) 
##width = 3
#ma_vec_1= (cumsum_vec_1[width:] - cumsum_vec_1[:-width]) / width
#cumsum_vec_2 = np.cumsum(np.insert(y, 0, 0)) 
#ma_vec_2= (cumsum_vec_2[width:] - cumsum_vec_2[:-width]) / width
#print(len(y1))

#winrate_y = [0.05]
#winrate_x = [20]
 
# plotting the points
#plt.plot(x, y, label = "3")
#plt.plot(x1,y1, label = "4")
plt.plot(x,it1, label="Iteration 1")
plt.plot(x2,it2, label = "Iteration 2")
plt.plot(x1,it3, label="Iteration 3")

#ma_vec_1 = np.insert(ma_vec_1,0,0.5)
#ma_vec_1 = np.insert(ma_vec_1,0,0.5)
#plt.plot(x1,ma_vec_1)

#plt.scatter(x,y)

ax=plt.gca()
# adjust the y axis scale.
plt.ylim(ymin=50, ymax=100)
plt.xlim(xmin=0, xmax=15)
ax.locator_params('y', nbins=5)
#ax.locator_params('x',nbins=10)

y_ = [0]

#plt.plot(x,y_, label = 'trailing average, 3x3', color = 'green')
plt.legend(bbox_to_anchor=(0.9,0.95), loc="lower center", borderaxespad=0)
#plt.subplots_adjust(right=0.75)

y1_ = [0]

#plt.plot(x1,y1_, label = 'trailing average, 4x4', color = 'black')
plt.legend(bbox_to_anchor=(0.9,0.95), loc="lower center", borderaxespad=0)

 
# naming the x axis
plt.xlabel("1000's of MCTS games simulated")
# naming the y axis
plt.ylabel('winrate per 96 games')
#print(np.convolve(y, np.ones(len(y))/len(y), mode='valid'))
# giving a title to my graph
#plt.title('Winrate vs random')

# function to show the plot
#print(y_)
plt.show()
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
#x9 = [252*i for i in range(len(y1))]
#y1_add = [0.8125, 0.927083333334, 0.916666666666]
#x9_add = [672*(i+1)+(252*len(x9)) for i in range(len(y1_add))]
#x9.extend(x9_add)
#y1.extend(y1_add)
it9 = [0.5, 0.84375,0.84375,0.84375, 0.885416666, 0.9375, 0.91666666, 0.90625, 0.95833333, 0.947916666666, 
       0.95833333333, 0.875, 0.8645833333334, 0.90625, 0.8541666666666666, 0.875, 0.9375,
       0.8958333333333334, 0.8854166666666666, 0.8854166666666666, 0.9375
       ]


it9_y = [(0.90625,7), (0.8541666666666666,4), (0.8020833333333334,0)]

trailing = [13440]
def trail(i):
       trailing.append(trailing[len(trailing)-1] + i)
       return trailing[len(trailing)-1]
trail_ = [(trail(32*10*(i[1]+1)))/1000 for i in it9_y]
it9_yy = [i[0] for i in it9_y]
x9 = [672/1000*i for i in range(len(it9))]
x9 = np.insert(x9, len(x9), trail_)
it9 = np.insert(it9, len(it9), it9_yy)
it9 = [i*100 for i in it9]

plt.axvline(x=13440/1000, color = 'orange', linestyle = '--')
iteration = 9
for i in trail_:
       plt.axvline(x=i, color='red', linestyle='--')
       
it8 = [0.5, 0.854166666, 0.80208333333334, 0.8854166666, 0.8958333333, 0.8229166666666666, 0.84375, 
       0.8541666666666666, 0.84375, 0.8854166666666666, 0.9166666666666666, 0.90625, 0.9375, 0.8645833333334,
       0.9270833333333334, 0.8541666666666666, 0.90625, 0.9375, 0.9270833333333334
       ]

it10 = [0.5, 0.739583333334, 0.6770833333333334, 0.7604166666666666, 0.7916666666666666, 0.7916666666666666,
        0.8541666666666666, 0.84375, 0.90625, 0.875, 0.8541666666666666, 0.8125, 0.875, 0.9270833333333, 0.88541666666,
        0.90625, 0.8645833333333334, 0.8854166666666666, 0.9270833333333334, 0.8854166666666666, 0.8541666666666666
        ]
print(len(it8)*672)
it8 = [i*100 for i in it8]
x8 = [672/1000*i for i in range(len(it8))]
it10 = [i*100 for i in it10]
x10 = [672/1000*i for i in range(len(it10))]
cumsum_vec_1 = np.cumsum(np.insert(it9, 0, 0)) 
width = 3
ma_vec_1= (cumsum_vec_1[width:] - cumsum_vec_1[:-width]) / width
ma_vec_1 = np.insert(ma_vec_1, 0,0.5)
ma_vec_1 = np.insert(ma_vec_1, 0,0.5)
#ma_vec_1 = np.insert(ma_vec_1, 0,0.5)

cumsum_vec_2 = np.cumsum(np.insert(it8, 0, 0)) 
width = 3
ma_vec_2= (cumsum_vec_2[width:] - cumsum_vec_2[:-width]) / width
ma_vec_2 = np.insert(ma_vec_2, 0,0.5)
ma_vec_2 = np.insert(ma_vec_2, 0,0.5)
#ma_vec_2 = np.insert(ma_vec_2, 0,0.5)

cumsum_vec_3 = np.cumsum(np.insert(it10, 0, 0)) 
width = 3
ma_vec_3= (cumsum_vec_3[width:] - cumsum_vec_3[:-width]) / width
ma_vec_3 = np.insert(ma_vec_3, 0,0.5)
ma_vec_3 = np.insert(ma_vec_3, 0,0.5)
#ma_vec_3 = np.insert(ma_vec_3, 0,0.5)
#cumsum_vec_2 = np.cumsum(np.insert(y, 0, 0)) 
#ma_vec_2= (cumsum_vec_2[width:] - cumsum_vec_2[:-width]) / width
#print(len(y1))

#winrate_y = [0.05]
#winrate_x = [20]
 
# plotting the points
#plt.plot(x, y, label = "3")
#plt.plot(x9,y1, label = "4")
plt.plot(x10,it10, label="Iteration 1, 2x2")
plt.plot(x8,it8, label = "Iteration 2, 3x3")
plt.plot(x9,it9, label="Iteration 3, 4x4")
#plt.plot(x9,ma_vec_1, label="Moving Average, 4x4")
#plt.plot(x8,ma_vec_2, label="Moving Average, 3x3")
#plt.plot(x10,ma_vec_3, label="Moving Average, 2x2")

#ma_vec_1 = np.insert(ma_vec_1,0,0.5)
#ma_vec_1 = np.insert(ma_vec_1,0,0.5)
#plt.plot(x9,ma_vec_1)

#plt.scatter(x,y)

ax=plt.gca()
# adjust the y axis scale.
plt.ylim(ymin=50, ymax=100)
plt.xlim(xmin=0, xmax=20)
ax.locator_params('y', nbins=20)
#ax.locator_params('x',nbins=10)

y_ = [0]

#plt.plot(x,y_, label = 'trailing average, 3x3', color = 'green')
plt.legend(bbox_to_anchor=(0.9,0.95), loc="lower center", borderaxespad=0)
#plt.subplots_adjust(right=0.75)

y1_ = [0]

#plt.plot(x9,y1_, label = 'trailing average, 4x4', color = 'black')
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
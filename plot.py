# importing the required module
import matplotlib.pyplot as plt
 
# x axis values
x = [6*i for i in range(9)]
# corresponding y axis values
y = [0.52,0.68,0.56,0.56,0.66,0.68,0.7,0.62, 0.76]
 
# plotting the points
plt.plot(x, y)

ax=plt.gca()
# adjust the y axis scale.
plt.ylim(ymin=0, ymax=1)
plt.xlim(xmin=0, xmax=200)
ax.locator_params('y', nbins=20)
 
# naming the x axis
plt.xlabel('MCTS games simulated')
# naming the y axis
plt.ylabel('winrate')
 
# giving a title to my graph
plt.title('Winrate vs random')
 
# function to show the plot
plt.show()
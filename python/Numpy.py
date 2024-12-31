import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im

# x = np.arange(0.1,6,0.01)
# y1 = np.sin(x)
# y2 = np.cos(x)
# y3 = np.log(x)
# y4 = x
# y5 = np.tan(x)

# plt.plot(x,y1,label="sin")
# plt.plot(x,y2,linestyle="--",label="cos")
# plt.plot(x,y3,label="ln")
# plt.plot(x,y4,label="x")

# #plt.plot(x,y5,label="tan")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("test")
# plt.legend()
# plt.show()

img = im.imread('python\img.png')
plt.imshow(img)
plt.show()
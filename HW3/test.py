import numpy as np
import heapq
ac = np.array([10,5.5,7,6,8])
acc = [0,1,2,3]
acc.remove(2)
# ad = map(ac.index, heapq.nlargest(3, ac))
# for i in range(10, 0, -1):
#     ac.insert(0,i)
dd = len(ac)
ad = ac.argsort()[::1][0:3]
print(acc)
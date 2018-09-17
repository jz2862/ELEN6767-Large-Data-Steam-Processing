import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline


result = [(0, 1.0, 5.38183007338448), (50, 0.979324988383067, 12.58431357792474), (100, 0.9587711408824195, 19.81494291782563), (150, 0.9315952736047688, 26.363827151439164), (200, 0.9426164631667028, 31.03973312204558), (250, 0.911208861604707, 36.90613877330241), (300, 0.9086407951753414, 40.7560492734138), (350, 0.8819270006066314, 45.57072324752942), (400, 0.864621717926078, 48.928753959303144), (450, 0.8775570055389982, 51.35734455978534), (500, 0.8820381152799267, 51.90689991204683), (550, 0.86698538631325, 53.32618526624449), (600, 0.8416056626463431, 55.56783756289763), (650, 0.8192330932214358, 56.186252097370925), (700, 0.7537181172624282, 56.314741230656445), (750, 0.6963158425473435, 57.55650284279562), (800, 0.6971225701127048, 57.346392979855494), (850, 0.6822977180973195, 57.76281708975139), (900, 0.6874018928142498, 58.021451440989864), (950, 0.682735125944831, 58.64001515339798)]

x = np.array([i[0] for i in result])
y1 = np.array([i[1] for i in result])
y2 = np.array([i[2]/5.382 for i in result])
# xnew = np.linspace(x.min(), x.max(), 300)
# y1 = spline(x, y1, xnew)
# y2 = spline(x, y2, xnew)

fig = plt.figure()
ax1 = fig.add_subplot(111)
l1, = ax1.plot(x, y1, label='Accuracy')
ax1.set_ylabel('Accuracy compared to no loadshedding')
ax1.set_title("Load Shedding Accuracy and Throughput Trade-off")
ax1.set_xlabel("MSE Threshold")

ax2 = ax1.twinx()  # this is the important function
l2, = ax2.plot(x, y2, 'r', label='Throughput')
# ax2.set_xlim([0, np.e])
ax2.set_ylabel('Throughput compared to no loadshedding')

plt.legend(handles = [l1, l2,], labels = ['Accuracy', 'Throughput'], loc = 'best')
plt.show()

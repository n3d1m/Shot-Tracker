from sympy import *
import numpy as np
import matplotlib.pyplot as plt

# HomeCourt values

velocity_h = [
    0.805,
    0.134,
    0.492,
    0.849,
    0.179,
    0.313,
    0.536,
    0.357,
    0.671,
    0.581,
]  # this is in MPH - has to be converted

print(np.mean(velocity_h), np.std(velocity_h))

release_time_h = [
    1.2,
    1.2,
    1.0,
    1.1,
    1.1,
    1.1,
    1.1,
    1.0,
    1.2,
    1.1,
]

print(np.mean(release_time_h), np.std(release_time_h))

release_angle_h = [
    50,
    53,
    52,
    50,
    51,
    52,
    50,
    51,
    53,
    50,
]

print(np.mean(release_angle_h), np.std(release_angle_h))

# Experimental values

velocity_e = [
    7.66,
    6.93,
    6.58,
    7.82,
    8.0,
    6.71,
    8.23,
    7.54,
    6.86,
    7.37,
]
print(np.mean(velocity_e), np.std(velocity_e))

release_time_e = [
    1.17,
    1.15,
    1.38,
    0.9,
    1.44,
    0.54,
    1.15,
    1.12,
    1.26,
    1.17,
]
print(np.mean(release_time_e), np.std(release_time_e))

release_angle_e = [
    50.57,
    49.3,
    46.4,
    49.87,
    53.58,
    49.36,
    50.56,
    48.21,
    52.64,
    48.77,
]

print(np.mean(release_angle_e), np.std(release_angle_e))

shot_made = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0]

counter = []

for i in range(10):
    diff = abs(release_angle_h[i] - release_angle_e[i])
    counter.append(diff)

print('MAE: ', sum(counter) / 10)

L_h = 0.219
R_b = 0.12
theta_b = 75
theta_h = 135
ang_v = []

for i in range(10):
    v = velocity_e[i]
    A = R_b ** 2
    B = 2 * L_h * R_b * cos(theta_b - theta_h)
    C = L_h ** 2 * rad(theta_h) ** 2 - v ** 2

    w = var('w')

    ans = solve(A * (w ** 2) + B * w + C)
    ans = sum([abs(x) for x in ans]) / 2
    ang_v.append(ans)

print(ang_v)

made_v = [7.66, 6.93, 6.58, 7.82, 6.71, 8.23, 7.54, 6.86]
made_angles = [50.57, 49.3, 46.4, 49.87, 49.36, 50.56, 48.21, 52.64]
made_rad = [63.71, 57.62, 54.69, 65.04, 55.78, 68.47, 62.71, 57.03]

missed_v = [8.0, 7.37]
missed_angles = [53.58, 48.77]
missed_rad = [66.55, 61.29]

N = 2
ind = np.arange(N)  # the x locations for the groups
width = 0.20  # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

means_v = [np.mean(made_v), np.mean(missed_v)]
rects1 = ax.bar(ind, means_v, width, color='r')

means_angle = [np.mean(made_angles), np.mean(missed_angles)]
rects2 = ax.bar(ind + width, means_angle, width, color='g')

means_rad = [np.mean(made_rad), np.mean(missed_rad)]
rects3 = ax.bar(ind + width * 2, means_rad, width, color='b')

ax.set_ylabel('test')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Made Shots', 'Missed Shots'))
ax.legend((rects1[0], rects2[0], rects3[0]),
          ('Average Release Velocity (m/s)', 'Average Release Angle (Degrees)', 'Average Angular Velocity (rad/s)'),
          loc='upper '
              'center',
          bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)


def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        print(h)
        ax.text(rect.get_x() + rect.get_width() / 2, 1.05 * h, '%.2f' % float(h),
                ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.show()


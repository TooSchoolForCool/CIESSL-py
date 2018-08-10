import numpy as np

# (x, y)
x0 = np.array([0, 0])
x1 = np.array([0, -7.27])
x2 = np.array([-5.94, -7.27])
x3 = np.array([-5.94, 0])
x4 = x2 + np.array([0, 0.2])
x5 = x4 + np.array([0, 7.54])
x6 = x5 + np.array([-7.53, 0])
x7 = x6 + np.array([0, -7.54])
x8 = x2 + np.array([1.5, 0])
x9 = x8 + np.array([4.12, 0])
x10 = x9 + np.array([0, -3.66])
x11 = x10 + np.array([-4.12, 0])
x12 = x4 + np.array([-3.51, 0])
x13 = x12 + np.array([0, -5.4])
x14 = x13 + np.array([3.63, 0])
x15 = x14 + np.array([0, -2.65])
x16 = x15 + np.array([1.93, 0])
x17 = x16 + np.array([0, 7.86])

px = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17]


def feet2meters(feet, inch):
    return feet * 0.3048 + inch * 0.0254


if __name__ == '__main__':
    while True:
        line = raw_input("x_feet x_inch y_feet y_inch x# idx:\n")
        line = line.split(' ')

        dx = feet2meters(int(line[0]), int(line[1]))
        dy = feet2meters(int(line[2]), int(line[3]))
        room = int(line[4])
        idx = int(line[5])

        x = px[room][0] + dx
        y = px[room][1] + dy

        print('{"x" : %.2lf,\t"y" : %.2lf,\t"idx" : %d}' % (x, y, idx))
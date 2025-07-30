import numpy as np
import math
import matplotlib.pyplot as plt

# === Angle Modulo Utility ===
def angle_mod(x, zero_2_2pi=False, degree=False):
    if degree:
        x = np.deg2rad(x)
    result = np.arctan2(np.sin(x), np.cos(x))  # wrap to [-π, π)
    if zero_2_2pi and result < 0:
        result += 2 * np.pi
    return np.rad2deg(result) if degree else result


# === Tuning Parameters ===
k = 0.1
Lfc = 2.0
Kp = 1.0
dt = 0.1
WB = 2.9
LENGTH = WB + 1.0
WIDTH = 2.0
WHEEL_LEN = 0.6
WHEEL_WIDTH = 0.2
MAX_STEER = math.pi / 4


class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - (WB / 2) * math.cos(self.yaw)
        self.rear_y = self.y - (WB / 2) * math.sin(self.yaw)

    def update(self, a, delta):
        delta = np.clip(delta, -MAX_STEER, MAX_STEER)
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.v += a * dt
        self.rear_x = self.x - (WB / 2) * math.cos(self.yaw)
        self.rear_y = self.y - (WB / 2) * math.sin(self.yaw)

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)


class States:
    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)


def proportional_control(target, current):
    return Kp * (target - current)


class TargetCourse:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        if self.old_nearest_point_index is None:
            d = [math.hypot(state.rear_x - icx, state.rear_y - icy) for icx, icy in zip(self.cx, self.cy)]
            ind = int(np.argmin(d))
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance = state.calc_distance(self.cx[ind], self.cy[ind])
            while True:
                if (ind + 1) >= len(self.cx):
                    break
                next_dist = state.calc_distance(self.cx[ind + 1], self.cy[ind + 1])
                if distance < next_dist:
                    break
                ind += 1
                distance = next_dist
            self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc
        while (ind + 1) < len(self.cx) and state.calc_distance(self.cx[ind], self.cy[ind]) < Lf:
            ind += 1
        return ind, Lf


def pure_pursuit_steer_control(state, trajectory, prevind):
    ind, Lf = trajectory.search_target_index(state)
    if ind < len(trajectory.cx):
        tx, ty = trajectory.cx[ind], trajectory.cy[ind]
    else:
        tx, ty = trajectory.cx[-1], trajectory.cy[-1]

    alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw
    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)
    return delta, ind


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width / 2, fc=fc, ec=ec)


def plot_vehicle(x, y, yaw, steer=0.0, color='blue'):
    def plot_wheel(wheel_x, wheel_y, wheel_yaw, steer=0.0, color=color):
        wheel = np.array([
            [-WHEEL_LEN / 2, WHEEL_WIDTH / 2],
            [WHEEL_LEN / 2, WHEEL_WIDTH / 2],
            [WHEEL_LEN / 2, -WHEEL_WIDTH / 2],
            [-WHEEL_LEN / 2, -WHEEL_WIDTH / 2],
            [-WHEEL_LEN / 2, WHEEL_WIDTH / 2]
        ])
        if steer != 0:
            c, s = np.cos(steer), np.sin(steer)
            rot_steer = np.array([[c, -s], [s, c]])
            wheel = wheel @ rot_steer.T
        c, s = np.cos(wheel_yaw), np.sin(wheel_yaw)
        rot_yaw = np.array([[c, -s], [s, c]])
        wheel = wheel @ rot_yaw.T
        wheel[:, 0] += wheel_x
        wheel[:, 1] += wheel_y
        plt.plot(wheel[:, 0], wheel[:, 1], color=color)

    corners = np.array([
        [-LENGTH / 2, WIDTH / 2],
        [LENGTH / 2, WIDTH / 2],
        [LENGTH / 2, -WIDTH / 2],
        [-LENGTH / 2, -WIDTH / 2],
        [-LENGTH / 2, WIDTH / 2]
    ])

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    body = corners @ rot.T
    body[:, 0] += x
    body[:, 1] += y
    plt.plot(body[:, 0], body[:, 1], color=color)

    front_x_offset = LENGTH / 4
    rear_x_offset = -LENGTH / 4
    half_width = WIDTH / 2

    plot_wheel(x + front_x_offset * c - half_width * s, y + front_x_offset * s + half_width * c, yaw, steer, 'black')
    plot_wheel(x + front_x_offset * c + half_width * s, y + front_x_offset * s - half_width * c, yaw, steer, 'black')
    plot_wheel(x + rear_x_offset * c - half_width * s, y + rear_x_offset * s + half_width * c, yaw, 0.0, 'black')
    plot_wheel(x + rear_x_offset * c + half_width * s, y + rear_x_offset * s - half_width * c, yaw, 0.0, 'black')
    plot_arrow(x, y, yaw)


def main():
    cx = np.arange(0, 50, 0.5)
    cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]

    target_speed = 10.0 / 3.6
    T = 100.0

    state = State(x=0.0, y=-3.0, yaw=0.0, v=0.0)
    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state)

    target_course = TargetCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(state)

    while T >= time and lastIndex > target_ind:
        ai = proportional_control(target_speed, state.v)
        di, target_ind = pure_pursuit_steer_control(state, target_course, target_ind)

        state.update(ai, di)
        time += dt
        states.append(time, state)

        plt.cla()
        plot_vehicle(state.x, state.y, state.yaw, di)
        plt.plot(cx, cy, "-r", label="Course")
        plt.plot(states.x, states.y, "-b", label="Trajectory")
        plt.plot(cx[target_ind], cy[target_ind], "xg", label="Target")
        plt.axis("equal")
        plt.grid(True)
        plt.title("Speed [km/h]: {:.2f}".format(state.v * 3.6))
        plt.legend()
        plt.pause(0.001)

    assert lastIndex >= target_ind, "Did not reach goal!"

    plt.cla()
    plt.plot(cx, cy, ".r", label="Course")
    plt.plot(states.x, states.y, "-b", label="Trajectory")
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)

    plt.figure()
    plt.plot(states.t, [v * 3.6 for v in states.v], "-r")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [km/h]")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    print("=== Pure Pursuit + PID Speed Control Simulation ===")
    main()

import numpy as np
import matplotlib.pyplot as plt


class Grid:
    def __init__(self):
        # unmutable, do not change over time
        self.dimension = 16
        self.switch = (12, 7.5)
        self.goal = (7.5, 12)
        self.max_steps_per_trial = 200
        self.radius = 1.5
        self.state_size = 7
        self.action_size = 2

        self.reset()
        # for non blocking figure display
        # plt.ion()
        # plt.show()

    def reset(self):
        # mutable, changes over time
        self.current_location = (0, 0)
        self.done = False
        self.hit_wall = False
        self.t = 0
        self.has_reached_switch = False
        self.move_trace = [self.current_location]
        return self.get_state()

    def render(self):
        fig = self.create_fig()
        plt.show()

        # plt.draw()
        # plt.pause(0.001)
        #plt.close(fig)

    def create_fig(self):
        # get the list of x and y coordinates in the trace
        move_trace = np.array(self.move_trace)
        x = move_trace[:, 0]
        y = move_trace[:, 1]

        fig, ax = plt.subplots()
        plt.title('Simulation field')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylim((0, self.dimension))
        ax.set_xlim((0, self.dimension))
        ax.grid(linestyle='dotted')
        ax.set_xticks(np.arange(0, self.dimension, 2))
        ax.set_yticks(np.arange(0, self.dimension, 2))
        ax.plot(x, y, color='red', linewidth=0.5, marker='o', markersize=3)

        # draw a circle at the goal area
        g_center = plt.Circle(self.goal, 0.1, color='blue', alpha=0.8)
        g_area = plt.Circle(self.goal, self.radius, color='blue', alpha=0.1)
        ax.add_artist(g_area)
        ax.add_artist(g_center)

        # draw a circle at the switch area
        s_center = plt.Circle(self.switch, 0.1, color='green', alpha=0.8)
        s_area = plt.Circle(self.switch, self.radius, color='green', alpha=0.1)
        ax.add_artist(s_area)
        ax.add_artist(s_center)

        ax.legend([g_area, s_area], ['goal', 'switch'], loc='upper right')

        return fig

    def get_dist_sin_and_cos(self, target):
        h, x, y = self.get_distance(target)
        sin = y / h
        cos = x / h
        return h, sin, cos

    def save_figure(self, filename):
        fig = self.create_fig()
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    def get_state(self):
        # information from goal
        dg, sin_g, cos_g = self.get_dist_sin_and_cos(self.goal)

        # information from switch
        ds, sin_s, cos_s = self.get_dist_sin_and_cos(self.switch)

        if self.switch_reached():
            signal = 10
        else:
            signal = 0

        return np.array([
            dg,
            sin_g,
            cos_g,
            ds,
            sin_s,
            cos_s,
            signal
        ])

    def get_distance(self, target):
        x = (target[0] - self.current_location[0])
        y = (target[1] - self.current_location[1])
        h = np.linalg.norm((x, y))
        return h, x, y

    def goal_reached(self):
        h, _, _ = self.get_distance(self.goal)
        return h <= self.radius

    def switch_reached(self):
        h, _, _ = self.get_distance(self.switch)
        return h <= self.radius

    def max_steps_per_trial_reached(self):
        return self.t == self.max_steps_per_trial

    def move(self, x, y):
        # ensure x and y are within -1 and 1
        x = np.clip(x, -1, 1)
        y = np.clip(y, -1, 1)

        # update current location
        x = x + self.current_location[0]
        y = y + self.current_location[1]

        self.hit_wall = False

        # if contact left and right wall, set reward to -0.1
        if x <= 0 or x >= self.dimension:
            self.hit_wall = True
            if x <= 0:
                x = 0
            else:
                x = self.dimension

        # contact top and bottom wall
        if y <= 0 or y >= self.dimension:
            self.hit_wall = True
            if y <= 0:
                y = 0
            else:
                y = self.dimension

        if self.switch_reached():
            self.has_reached_switch = True

        self.current_location = (x, y)
        self.move_trace.append(self.current_location)

    def step(self, action):
        action = np.reshape(action, -1)
        x = action[0]
        y = action[1]
        # check if maximum steps have been reached
        if self.done:
            raise Exception("Over ", self.max_steps_per_trial, " steps has been reached, reset the environment")
            return

        # move the agent in the env according to the x and y direction with value min -1, max 1
        self.move(x, y)
        state = self.get_state()

        reward = -0.01

        # the agent hits the wall after move
        if self.hit_wall:
            reward = - 0.1

        # reached the goal
        if self.goal_reached():
            if self.has_reached_switch:
                reward = 0.8
            else :
                reward = -0.5

        # check if reached maximum steps
        if self.max_steps_per_trial_reached() or self.goal_reached():
            self.done = True

        self.t += 1

        return state, reward, self.done


if __name__ == "__main__":
    def test_grid():
        grid = Grid()
        state = grid.reset()
        print("reset state:", state, "\n")
        straight_to_goal = True
        total_rewards = 0

        for i in range(50):
            if straight_to_goal:
                # if i < 25:
                #     x = np.random.uniform(-0.55, 1)
                #     y = np.random.uniform(-0.55, 1)
                # else:
                #     x = -0.2
                #     y = 0.2

                if i < 25:
                    x = 0.35
                    y = 0.5
                else:
                    x = -0.2
                    y = 0.2
            else:
                if i < 25:
                    x = 0.25
                    y = 0.45
                else:
                    x = -0.2
                    y = 0.2
                # if i < 25:
                #     x = 0.45
                #     y = 0.3
                # else:
                #     x = -0.2
                #     y = 0.2

            print("step:", i, ",action x:", x,", y:", y)

            state, reward, done = grid.step([x, y])
            total_rewards += reward
            print("current location:", np.round(grid.current_location, 3), "new_state=", np.round(state, 3), ", reward=", reward, ", done=", done, "\n")

            if done:
                break;

        grid.render()
        print("total rewards:", total_rewards)
        plt.show()

    test_grid()

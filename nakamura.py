import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

# Random seed
np.random.seed(42)

# Robot Parameters
N_ROBOTS = 5
DOMAIN = np.array([[0, 10], [0, 10]])  # 2D Domain
BBOX = [0, 0, 10, 10]  # Bounding Box
K = 1  # Control gain
dT = 0.5  # Time step

# Generate random positions for robots
robot_positions = np.random.uniform(DOMAIN[0][0], DOMAIN[0][1], size=(N_ROBOTS, 2))

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_aspect('equal')
ax.set_xlim(DOMAIN[0])
ax.set_ylim(DOMAIN[1])

def plot_figure(t, ax2, ax, robot_history, robot_positions, robot_voronois, Z, DOMAIN, e, mean):
    # Take the axes from the figure
    ax.clear()
    ax.imshow(Z, extent=(DOMAIN[0][0], DOMAIN[0][1], DOMAIN[1][0], DOMAIN[1][1]), origin='lower', alpha=0.5)
    ax.scatter(e[:, 0], e[:, 1], c='red', marker='x')

    for i in range(N_ROBOTS):
        # Plot Voronoi regions
        ax.plot(*robot_voronois[i].exterior.xy, c='black', alpha=0.5)
        ax.scatter(robot_positions[i, 0], robot_positions[i, 1], c=f'C{i}', label=f'Robot {i}')
        ax.plot(robot_history[i][:, 0], robot_history[i][:, 1], c=f'C{i}', alpha=0.5)
    # Legend
    ax.legend(loc='upper right')
    ax2.clear()
    ax2.contourf(np.linspace(DOMAIN[0][0], DOMAIN[0][1], 100), np.linspace(DOMAIN[1][0], DOMAIN[1][1], 100), mean.reshape(100, 100))
    plt.pause(0.1)


def voronoi_algorithm(robots_positions, BBOX):
    """ Decentralized Bounded Voronoi Computation """
    points_left = robots_positions.copy()
    points_right = robots_positions.copy()
    points_down = robots_positions.copy()
    points_up = robots_positions.copy()

    points_left[:, 0] = 2 * BBOX[0] - robots_positions[:, 0]
    points_right[:, 0] = 2 * BBOX[2] - robots_positions[:, 0]
    points_down[:, 1] = 2 * BBOX[1] - robots_positions[:, 1]
    points_up[:, 1] = 2 * BBOX[3] - robots_positions[:, 1]

    points = np.vstack([robots_positions, points_left, points_right, points_down, points_up])

    # Voronoi diagram
    vor = Voronoi(points)
    vor.filtered_points = robots_positions
    vor.filtered_regions = [vor.regions[i] for i in vor.point_region[:len(robots_positions)]]
    
    return vor

# Example spatial field (true function)
def true_field(X):
    # 2D Gaussian field
    mean = np.array([8, 8])
    cov = 1
    return np.exp(-np.sum((X - mean)**2, axis=1) / (2 * cov**2))

# Sample the field (robot sensors)
def sample_field(robot_positions):
    return true_field(robot_positions) + np.random.normal(0, 0.01, N_ROBOTS)

# Gaussian Process Regression model
def initialize_gpr():
    kernel = C(1.0) * RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel)
    return gpr

# Update GPR model with local and neighboring samples
def update_gpr(gpr, X_train, y_train):
    gpr.fit(X_train, y_train)
    return gpr

# Estimate spatial field and uncertainty
def estimate_field(gpr, X_pred):
    y_pred, sigma = gpr.predict(X_pred, return_std=True)
    return y_pred, sigma

# Simulation Loop
def simulate_dvec(steps=20):
    global robot_positions
    gpr_models = [initialize_gpr() for _ in range(N_ROBOTS)]

    X = np.linspace(DOMAIN[0][0], DOMAIN[0][1], 100)
    Y = np.linspace(DOMAIN[1][0], DOMAIN[1][1], 100)
    delta = DOMAIN[0][1] / 100
    dA = delta**2
    X, Y = np.meshgrid(X, Y)
    XY_mesh = np.c_[X.ravel(), Y.ravel()]
    Z = true_field(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    gamma = 0.5
    # Decreasing from 1 to 0 over the steps
    gamma_dec = np.linspace(0.1, 0, steps)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    robot_voronois = [np.array([]) for _ in range(N_ROBOTS)]
    X_datasets = [np.empty((0, 2)) for _ in range(N_ROBOTS)]
    y_datasets = [np.empty(0) for _ in range(N_ROBOTS)]
    for step in range(1, steps+1):
        print(f'Step {step}/{steps}')
        print("Dataset size: ", [len(x) for x in X_datasets])
        # Update the beta parameter
        beta = 2 * np.log( len(X[0])*len(Y[0]) * np.pi**2/(6*gamma) * step**2) #len(X[0])*len(Y[0])

        vor = voronoi_algorithm(robot_positions, BBOX)
        for i, region in enumerate(vor.filtered_regions):
            polygon = Polygon(vor.vertices[region])
            robot_voronois[i] = polygon

        # Sampling
        local_samples = sample_field(robot_positions)

        e = np.empty((N_ROBOTS, 2))
        d_func = [[] for _ in range(N_ROBOTS)]
        masks = [[] for _ in range(N_ROBOTS)]
        y_preds = [[] for _ in range(N_ROBOTS)]

        for i in range(N_ROBOTS):
            # Data exchange
            neighbors = [j for j in range(N_ROBOTS) if j != i and np.any(np.isin(vor.regions[vor.point_region[i]], vor.regions[vor.point_region[j]]))]
            X_train = np.vstack([robot_positions[i], robot_positions[neighbors]])
            y_train = np.concatenate([np.array([local_samples[i]]), local_samples[neighbors]])
            X_datasets[i] = np.concatenate([X_datasets[i], X_train])
            y_datasets[i] = np.concatenate([y_datasets[i], y_train])
            # Update the GPR model
            gpr_models[i] = update_gpr(gpr_models[i], X_datasets[i], y_datasets[i])
            # Estimate field and uncertainty
            y_pred, sigma = estimate_field(gpr_models[i], XY_mesh)
            y_preds[i] = y_pred
            d_func[i] = y_pred#- np.sqrt(beta)*sigma

            # Find the points of the mesh that are inside the polygon
            masks[i] = np.array([robot_voronois[i].contains(Point(p)) for p in XY_mesh])

            # Take the max uncertainty point inside the polygon
            e[i] = XY_mesh[masks[i]][np.argmax(sigma[masks[i]])]

        robot_history = [np.empty((0, 2)) for _ in range(N_ROBOTS)]

        while True:
            
            for i in range(N_ROBOTS):
                robot_history[i] = np.vstack([robot_history[i], robot_positions[i]])

            # Plot the goal poses
            goal_poses = np.zeros((N_ROBOTS, 2))
            for i in range(N_ROBOTS):
                Cx = np.sum(XY_mesh[:, 0][masks[i]] * d_func[i][masks[i]]) * dA / (np.sum(d_func[i][masks[i]]) * dA)
                Cy = np.sum(XY_mesh[:, 1][masks[i]] * d_func[i][masks[i]]) * dA / (np.sum(d_func[i][masks[i]]) * dA)

                goal_poses[i] = ((1 - gamma_dec[step]) * np.array([Cx, Cy]) + gamma_dec[step] * e[i]) # - robot_positions[i])

                # Move the robot
                x1, x2 = robot_positions[i] + K * (((1 - gamma_dec[step]) * np.array([Cx, Cy]) + gamma_dec[step] * e[i]) - robot_positions[i]) * dT
                robot_positions[i] = np.array([x1, x2])
            plot_figure(step, ax2, ax, robot_history, robot_positions, robot_voronois, Z, DOMAIN, goal_poses, y_preds[0])
            dist = np.linalg.norm(robot_positions - goal_poses, axis=1)
            if np.all(dist < 0.005):
                break

# Run the simulation
simulate_dvec()
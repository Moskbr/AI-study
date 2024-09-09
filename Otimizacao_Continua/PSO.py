import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# Objective Function
def Rosenbrock(x):
    x = np.asarray(x)
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# visualizing the Rosenbrock function
def plotFunction3D(X, Y, Z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surface = ax.plot_surface(X, Y, Z([X, Y]), cmap=cm.get_cmap('viridis'), linewidth=0, antialiased=False)
    fig.colorbar(surface, shrink=0.65, aspect=10)
    ax.scatter([1],[1],Rosenbrock([1,1]), color='red', s=100) # objective result: point (1,1,0)
    plt.show()

# visualizing the function though the Z plane
def plot_contour(X, Y, Z):
    plt.figure(figsize=(8,5))
    plt.contour(X, Y, Rosenbrock([X,Y]), 200)
    plt.scatter([1],[1], color='red', s=50) # objective point slicing the Z plane
    plt.show()



# this class has the variables of each particle
class Particle:
    def __init__(self, dimension, x_limits) -> None:
        self.position = np.random.uniform(x_limits[0], x_limits[1], (dimension, 1)) # consider same limits for all dimensions
        self.velocity = np.random.uniform(-1, 1, (dimension, 1)) # begins with random positions and velocity
        self.pbest = self.position.copy() # initial best position
        self.pbest_fitness = np.inf
        self.fitness = np.inf
    
    # evaluates the cost function with the actual position
    def updateFitness(self, objective_function):
        self.fitness = objective_function(self.position)
        if self.fitness < self.pbest_fitness: # if it's the lowest value found
            self.pbest_fitness = self.fitness # it's a new best
            self.pbest = self.position.copy()
    
    # apllies the velocity formula for each dimension
    def updateVelocity(self, gbest, W, C1, C2, dim, v_max):
        for d in range(dim):
            r1 = np.random.rand()
            r2 = np.random.rand()
            self.velocity[d] = W*self.velocity[d] + C1*r1*(self.pbest[d] - self.position[d]) + C2*r2*(gbest[d] - self.position[d])

            if self.velocity[d] > v_max:
                self.velocity[d] = v_max

    # applies the position formula and verifies if it's a valid position
    def updatePosition(self, dimensions, x_Limits):
        self.position += self.velocity
        inferior = x_Limits[0] # for simplicity, considiring same limits for all dimensions -> Limits = [inf_limit, sup_limit]
        superior = x_Limits[1]
        for dim in range(dimensions):
            if self.position[dim] < inferior:
                self.position[dim] = float(inferior)
            if self.position[dim] > superior:
                self.position[dim] = float(superior)

# this class has the variables of the entire swarm
class Swarm:
    def __init__(self, n_particles, dimension, x_Limits) -> None:
        self.particles = [Particle(dimension, x_Limits) for _ in range(n_particles)]
        self.gbest = np.zeros((dimension, 1)) # best position of the group
        self.gbest_fitness = np.inf # lowest return value of cost function of the group

# implementing the algorithm
class PSO:
    def __init__(self, N_particles, Dim, Limits, max_iter, v_max, objective_function, C1, C2, isW_Linear=False, W=0.9) -> None:
        self.swarm = Swarm(N_particles, Dim, Limits)
        self.fitness = [] # to plot the convergences values in each iteration
        self.linear_inertia = [] # to plot the inertia values in each iteration, is 'isW_linear' is enabled

        for it in range(max_iter):
            
            # shows the position of each particle around the convergence point, 5 times
            if (it+1)%(max_iter/5) == 0 or it == 0:
                self.plotContour(Limits, self.swarm)

            for p in self.swarm.particles:
                if isW_Linear:
                    W = self.calculateLinearInertia(it, max_iter) # if enabled, updates the value of W
                
                p.updateVelocity(self.swarm.gbest, W, C1, C2, Dim, v_max)
                p.updatePosition(Dim, Limits)

                p.updateFitness(objective_function)

                # verify if the particle has the new best parameters of the swarm
                if p.pbest_fitness < self.swarm.gbest_fitness:
                    self.swarm.gbest_fitness = p.pbest_fitness
                    self.swarm.gbest = p.pbest.copy()
            
            self.fitness.append((it, self.swarm.gbest_fitness[0]))
            print(f'it: {it}  gBest_fitness: {self.swarm.gbest_fitness} gBest_Position: {self.swarm.gbest[0]},{self.swarm.gbest[1]}') # checking
            if isW_Linear: self.linear_inertia.append((it, W))

        if isW_Linear: self.plotInertia() # for analisys
    
    # updates new value for W
    def calculateLinearInertia(self, it, max_iter):
        Wmax = 0.9
        Wmin = 0.4
        W = Wmax - it*((Wmax-Wmin)/max_iter)
        return W
    
    # plot the decreasing value of the linear inertia, if enabled
    def plotInertia(self):
        data = np.asarray(self.linear_inertia)
        plt.plot(data[:,0], data[:,1])
        plt.xlabel('iterations')
        plt.ylabel('Inertia (W)')
        plt.title("Linear Inertia")
        plt.show()
    
    # monitor the behaviour of the particles in finding the convergence point
    def plotContour(self, Limits, Swarm:Swarm):
        x = np.arange(Limits[0], Limits[1], 0.15)
        y = np.arange(Limits[0], Limits[1], 0.15)
        X, Y = np.meshgrid(x, y)
        plt.figure(figsize=(8,5))
        plt.contour(X, Y, Rosenbrock([X,Y]), 200)
        plt.scatter([1],[1], color='red', s=50) # objective point slicing the Z plane

        for p in Swarm.particles:
            plt.scatter(p.position[0], p.position[1], color = 'gray', s=25)
        
        plt.show()



# for comparing two PSOs with different configurations
def plotFitness(pso1:PSO, pso2:PSO):
    data_pso1 = np.asarray(pso1.fitness)
    plt.plot(data_pso1[:,0], data_pso1[:,1], label='Constant Inertia')
    data_pso2 = np.asarray(pso2.fitness)
    plt.plot(data_pso2[:,0], data_pso2[:,1], label='Linear Inertia')
    plt.xlabel('iterations')
    plt.ylabel('best function value')
    plt.title("PSO x Improved PSO")
    plt.legend()
    plt.show()



def main():
    Limits = [-5, 10] # for all dimensions
    x = np.arange(Limits[0], Limits[1], 0.15)
    y = np.arange(Limits[0], Limits[1], 0.15)
    X, Y = np.meshgrid(x, y)

    # visualizing the Rosenbrock function
    plotFunction3D(X, Y, Z=Rosenbrock)
    # showing the convergence point through the Z plane
    plot_contour(X, Y, Z=Rosenbrock)

    # Config Variables
    n_particles = 10
    dimensions = 2
    c1 = c2 = 1.5
    max_iterations = 100
    Vmax = 2.5 # max velocity

    pso = PSO(n_particles, dimensions, Limits, max_iterations, Vmax, objective_function=Rosenbrock, C1=c1, C2=c2, isW_Linear=False)
    print(f'PSO_1: best position = {pso.swarm.gbest[0]},{pso.swarm.gbest[1]}  best fitness = {pso.swarm.gbest_fitness[0]:.8f}')

    print("\n\n")

    pso_wLinear = PSO(n_particles, dimensions, Limits, max_iterations, Vmax, objective_function=Rosenbrock, C1=c1, C2=c2, isW_Linear=True)
    print(f'PSO_2: best position = {pso_wLinear.swarm.gbest[0]},{pso_wLinear.swarm.gbest[1]}  best fitness = {pso_wLinear.swarm.gbest_fitness[0]:.8f}')

    plotFitness(pso, pso_wLinear)



if __name__ == '__main__':
    main()
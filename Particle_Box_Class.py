import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import scipy.stats as stats
import matplotlib.animation as animation
from IPython.display import HTML


# This simulation assumes the Boltzmann constant to be 1.

class ParticleBox(object): 
    def __init__(self, m, r, T, L, N, e_loss = 0.):
        """
        Initialize the system
        
        Args:
            m: Mass of the particles
            r: Radius of the particles
            T: Temperature in Kelvin
            L: Length of the square box
            N: Total number of particles
            e_loss: Percent of kinetic energy lost per collision
        """
        self.m = m
        self.r = r
        self.T = T
        self.L = L
        self.N = N
        self.e_loss = e_loss
        self.ds = np.random.random((self.N, 2))*self.L
        
        v_avg = np.sqrt(3*T/self.m)
        # initialize velocities to be normally distributed centered at v_avg
        # The random.choice will initialize each velocity in a random direction
        self.vs = np.random.normal(v_avg/np.sqrt(2), v_avg/4, (self.N, 2))*np.random.choice([1, -1], (self.N, 2))
        
        self.wall_collisions = 0

    def update(self, dt):
        """
        Updates positions and velocities of a particle
        
        Args:
            dt: time step
            n: index of particle
        """
        # These 2 lines account for collisions with the left and bottom walls
        walls_1 = self.ds <= self.r
        self.ds[walls_1] = self.r 
        self.vs[walls_1] *= -(1 - self.e_loss)
        
        # Theses 2 lines account for collisions with the right and top walls
        walls_2 = self.ds >= self.L - self.r
        self.ds[walls_2] = self.L - self.r
        self.vs[walls_2] *= -(1 - self.e_loss)
        
        # Add to total number of wall collisions
        self.wall_collisions += np.sum(walls_1) + np.sum(walls_2)
        
        # Square matrix of each particles distance from each other
        D = squareform(pdist(self.ds))
        
        # Indices of particles that collided
        i1, i2 = np.where(D < 2*self.r)
        # Get rid of duplicate pairs, as well as diagonals which will have D = 0
        keep = i1 < i2
        i1 = i1[keep] 
        i2 = i2[keep]
        
        # Run collision function for each colliding pair
        for i,j in zip(i1, i2):
            self.collision(i, j)
        
        # Update locations
        self.ds += self.vs*dt
        
    def collision(self, i, j):
        """
        Calculates new velocities after a particle collision
        
        Args:
            i: index of the first particle
            j: index of the second particle
        """
        # momentum vector of the center of mass
        p_com = (self.m * self.vs[i] + self.m * self.vs[j]) / (2*self.m)
        
        # relative location & velocity vectors
        r_rel = self.ds[i] - self.ds[j]
        v_rel = self.vs[i] - self.vs[j]

        # collisions of spheres reflect v_rel over r_rel
        rr_dot = np.dot(r_rel, r_rel)
        
        # This prevents divide by zero errors. rr_dot will be instead set to a number
        # much smaller than would normally be encountered.
        if rr_dot == 0.0:
            rr_dot = 10e-6
        
        vr_dot = np.dot(v_rel, r_rel)
        v_rel = (2 * r_rel * vr_dot / rr_dot) - v_rel

        # assign new velocities
        vi_new = p_com + v_rel * self.m / (2*self.m)
        vj_new = p_com - v_rel * self.m / (2*self.m) 
        
        # Set new velocities
        self.vs[i] = vi_new*(1 - self.e_loss)
        self.vs[j] = vj_new*(1 - self.e_loss)
            
    
    def get_avg_velocity(self):
        """
        Return average velocity
        """
        vs = np.hypot(self.vs[:, 0], self.vs[:, 1])
        
        return np.average(vs)
    
    def get_PDF(self):
        """
        Return probability density function of velocities
        """
        vs = np.hypot(self.vs[:, 0], self.vs[:, 1])
        
        return norm.pdf(vs)
    
    def get_vs(self):
        """
        Return velocities in order
        """
        vs = np.hypot(self.vs[:, 0], self.vs[:, 1])
        
        return np.sort(vs)


# In[5]:


L = 10
N = 200

sys = ParticleBox(5, 0.1, 1000, L, N)
dt = 0.001

# set up figure and animation

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])
ax.set_xlim(-1, sys.L + 1)
ax.set_ylim(-1, sys.L + 1)


# Particles contains the particle locations
particles, = ax.plot([], [], 'bo', ms = 3)


# Box is the box the particles are in
box = plt.Rectangle((0,0), sys.L, sys.L, fill = False)
ax.add_patch(box)

def init():
    """initialize animation"""
    particles.set_data([], [])
    box.set_edgecolor('k')
    return particles, box

def animate(i):
    """perform animation step"""
    sys.update(dt)

    # update pieces of the animation
    box.set_edgecolor('k')
    particles.set_data(sys.ds[:, 0], sys.ds[:, 1])
    return particles, box

num_frames = 100
step = 10
ani = animation.FuncAnimation(fig, animate, frames = num_frames,
                              interval = step, blit=True, init_func=init)

plt.close()
ani.save('ParticleBox.mp4')

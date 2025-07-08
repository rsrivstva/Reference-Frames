# Main file with everything

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from scipy.integrate import solve_ivp

def main():
    # Main function
    print("Starting Main Function")
    initial_position = np.array([7000,0,0])
    initial_velocity = np.array([0, 7.72, 5])
    integration_time = 24*60*60*365*4
    integration_steps = 1000

    sun_mu = 1.327124e11

    earth_position = np.array([149.6e6,0,0])
    earth_velocity = np.array([0, np.sqrt(sun_mu/np.linalg.norm(earth_position)), 0])

    mars_position = np.array([228e6,0,0])
    mars_velocity = np.array([0, np.sqrt(sun_mu/np.linalg.norm(mars_position)), 0])

    theta = 90                                                             
    init_epic="Jan 1, 2000"
    earth_trajectory, times = keplerian_propagator(earth_position, earth_velocity, integration_time, integration_steps)
    mars_trajectory, times = keplerian_propagator(mars_position, mars_velocity, integration_time, integration_steps)

    init_ephem()
    et = spice.str2et(init_epic)

    # Steps for rotating to J2000
    # 1. For each Epoch (time + initial Epoch), get distance to Earth in J2000 frame
    # # Get vector from S to E
    # rse, _ = spice.spkpos('Earth', current_epic, 'J2000', 'LT', 'Sun')
    # 2. Get rotation matrix from our Sun inertial frame to J2000
    # 3. For every measurement, multiply that rotation matrix by the Earth-Mars vector
    # Final result is Earth -> Mars vector in the J2000 frame

    # Plot it
    fig = plt.figure()
    # Define axes in that figure
    ax = plt.axes(projection='3d',computed_zorder=False)
    # Plot x, y, z
    ax.plot(earth_trajectory[0],earth_trajectory[1],earth_trajectory[2],zorder=5)
    ax.plot(mars_trajectory[0],mars_trajectory[1],mars_trajectory[2],zorder=5)
    plt.title("All Orbits")
    ax.set_xlabel("X-axis (km)")
    ax.set_ylabel("Y-axis (km)")
    ax.set_zlabel("Z-axis (km)")
    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.zaxis.set_tick_params(labelsize=7)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    fig = plt.figure()
    # Define axes in that figure
    ax = plt.axes()
    # Plot x, y, z
    ax.scatter(times/(60*60*24),earth_trajectory[0]-mars_trajectory[0],zorder=5, label='Relative X')
    ax.scatter(times/(60*60*24),earth_trajectory[1]-mars_trajectory[1],zorder=5, label='Relative Y')
    ax.scatter(times/(60*60*24),earth_trajectory[2]-mars_trajectory[2],zorder=5, label='Relative Z')
    plt.title("All Orbits")
    ax.set_xlabel("Times [days]")
    ax.set_ylabel("Relative Distance (km)")
    ax.legend()
    ax.grid(True)
    plt.show()




def keplerian_propagator(init_r, init_v, tof, steps):
    """
    Function to propagate a given orbit
    """
    # Time vector
    tspan = [0, tof]
    # Array of time values
    tof_array = np.linspace(0,tof, num=steps)
    init_state = np.concatenate((init_r,init_v))
    # Do the integration
    sol = solve_ivp(fun = lambda t,x:keplerian_eoms(t,x), t_span=tspan, y0=init_state, method="DOP853", t_eval=tof_array, rtol = 1e-12, atol = 1e-12)
    # Return everything
    return sol.y, sol.t


def keplerian_eoms(t, state):
    """
    Equation of motion for 2body orbits
    """
    sun_mu = 1.327124e11
    # Extract values from init
    x, y, z, vx, vy, vz = state
    r_dot = np.array([vx, vy, vz])
    
    # Define r
    r = (x**2 + y**2 + z**2)**.5
    
    # Solve for the acceleration
    ax = -sun_mu*x/(r**3)
    ay = -sun_mu*y/(r**3)
    az = -sun_mu*z/(r**3)

    v_dot = np.array([ax, ay, az])

    dx = np.append(r_dot, v_dot)

    return dx


def init_ephem(self):
        """
        Initialize the ephemeris files for EMS
        """
        # This is for the Cassini example, comment out later
        #spice.furnsh("./Ephemeris/cassMetaK.txt")
        # Furnish the kernals we actually need
        spice.furnsh("./Ephemeris/ephemMeta.txt")

if __name__ == '__main__':
    main()
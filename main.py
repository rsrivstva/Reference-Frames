# Main file with everything

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from scipy.integrate import solve_ivp

def rotate_to_rotating_frame(states, times, mu_sun, a_earth):
    rotated_positions = np.zeros((3, len(times)))
    for i, t in enumerate(times):
        theta = 2 * np.pi * t / (60*60*24) # rotation angle
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0,              0,             1]])
        rotated_positions[:, i] = R @ states[0:3, i]

    return rotated_positions

def main():
    # Main function
    print("Starting Main Function")
    initial_position = np.array([7000,0,0])
    initial_velocity = np.array([0, 7.72, 5])
    integration_time = 24*60*60*30
    integration_steps = 1000
    init_epic="Jan 1, 2000"

    sun_mu = 1.327124e11
    a_earth = 149.6e6
    theta = 90

    init_ephem()
    et = spice.str2et(init_epic)

    rse, _ = spice.spkpos('Earth', et, 'J2000', 'LT', 'Sun')

    earth_position = np.array([149.6e6,0,0])
    earth_velocity = np.array([0, np.sqrt(sun_mu/np.linalg.norm(earth_position)), 0])
    mars_position = np.array([228e6,0,0])
    mars_velocity = np.array([0, np.sqrt(sun_mu/np.linalg.norm(mars_position)), 0]) 
    venus_position = np.array([108.2e6, 0, 0])
    venus_velocity = np.array([0, np.sqrt(sun_mu / np.linalg.norm(venus_position)), 0]) 
                                                           
    earth_trajectory, times = keplerian_propagator(earth_position, earth_velocity, integration_time, integration_steps)
    mars_trajectory, times = keplerian_propagator(mars_position, mars_velocity, integration_time, integration_steps)
    venus_trajectory, _ = keplerian_propagator(venus_position, venus_velocity, integration_time, integration_steps)

    mars_rot = rotate_to_rotating_frame(mars_trajectory - earth_trajectory, times, sun_mu, a_earth)
    venus_rot = rotate_to_rotating_frame(venus_trajectory - earth_trajectory, times, sun_mu, a_earth)

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(mars_rot[0], mars_rot[1], label="Mars (rotating frame)")
    ax.plot(venus_rot[0], venus_rot[1], label="Venus")
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_aspect("equal")
    ax.autoscale(enable=True, axis='both', tight=True)
    ax.legend()
    plt.title("Earth-Centered Rotating Frame")
    plt.show()

    # Steps for rotating to J2000
    # 1. For each Epoch (time + initial Epoch), get distance to Earth in J2000 frame
    # # Get vector from S to E
    # Rotate every state along trajectory
    em_j2000 = np.zeros((len(times), 3))
    ev_j2000 = np.zeros((len(times), 3))
    for i in range(len(times)):
        current_epoch = et + times[i]
        rse, _ = spice.spkpos('Earth', current_epoch, 'J2000', 'LT', 'Sun')
        # 2. Get rotation matrix from our Sun inertial frame to J2000
        rse_sun_centered = earth_trajectory[0:3,i]
        rot_mat = rotation_matrix_from_vectors(rse_sun_centered, rse)
        # 3. For every measurement, multiply that rotation matrix by the Earth-Mars vector
        em_sun_centered = mars_trajectory[0:3,i] - earth_trajectory[0:3,i]
        ev_sun_centered = venus_trajectory[0:3,i] - earth_trajectory[0:3,i]
        em_j2000[i, :] = rot_mat @ em_sun_centered
        ev_j2000[i, :] = rot_mat @ ev_sun_centered
        # Final result is Earth -> Mars vector in the J2000 frame
  
    # Plot it
    fig = plt.figure()
    # Define axes in that figure
    ax = plt.axes()
    # Plot x, y, z
    #ax.plot(earth_trajectory[0],earth_trajectory[1],earth_trajectory[2],zorder=5)
    #ax.plot(mars_trajectory[0],mars_trajectory[1],mars_trajectory[2],zorder=5)

    earth_centered_earth = earth_trajectory - earth_trajectory
    earth_centered_mars = mars_trajectory - earth_trajectory
    earth_centered_venus = venus_trajectory - earth_trajectory

    ax.plot(earth_centered_earth[0], earth_centered_earth[1], zorder=5, label='Earth')
    ax.plot(earth_centered_mars[0], earth_centered_mars[1], zorder=5, label='Mars')
    ax.plot(earth_centered_venus[0], earth_centered_venus[1], zorder=5, label='Venus')
    ax.legend()

    plt.title("All Orbits")
    ax.set_xlabel("X-axis (km)")
    ax.set_ylabel("Y-axis (km)")
    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    #ax.set_aspect('equal', adjustable='box')
    plt.show()

    fig = plt.figure()
    # Define axes in that figure
    ax = plt.axes()
    # Plot x, y, z
    ax.scatter(times/(60*60*24),em_j2000[:,0],zorder=5, label='Relative X in J2000')
    ax.scatter(times/(60*60*24),em_j2000[:,1],zorder=5, label='Relative Y in J2000')
    ax.scatter(times/(60*60*24),em_j2000[:,2],zorder=5, label='Relative Z in J2000')

    ax.scatter(times/(60*60*24),ev_j2000[:,0],zorder=5, label='Relative X in J2000')
    ax.scatter(times/(60*60*24),ev_j2000[:,1],zorder=5, label='Relative Y in J2000')
    ax.scatter(times/(60*60*24),ev_j2000[:,2],zorder=5, label='Relative Z in J2000')
    plt.title("All Orbits")
    ax.set_xlabel("Times [days]")
    ax.set_ylabel("Relative Distance (km)")
    ax.legend()
    ax.grid(True)
    plt.show()

    fig = plt.figure()
    # Define axes in that figure
    ax = plt.axes()
    # Plot x, y, z
    ax.scatter(times/(60*60*24),-earth_trajectory[0]+mars_trajectory[0],zorder=5, label='Relative X')
    ax.scatter(times/(60*60*24),-earth_trajectory[1]+mars_trajectory[1],zorder=5, label='Relative Y')
    ax.scatter(times/(60*60*24),-earth_trajectory[2]+mars_trajectory[2],zorder=5, label='Relative Z')

    ax.scatter(times/(60*60*24),-earth_trajectory[0]+venus_trajectory[0],zorder=5, label='Relative X')
    ax.scatter(times/(60*60*24),-earth_trajectory[1]+venus_trajectory[1],zorder=5, label='Relative Y')
    ax.scatter(times/(60*60*24),-earth_trajectory[2]+venus_trajectory[2],zorder=5, label='Relative Z')
    plt.title("All Orbits")  
    ax.set_xlabel("Times [days]")
    ax.set_ylabel("Relative Distance (km)")
    ax.legend()
    ax.grid(True)
    plt.show()
    
    diff = em_j2000 - earth_centered_mars[0:3,:].T
    diff2 = ev_j2000 - earth_centered_venus[0:3,:].T
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(times/(60*60*24), diff[:,0], zorder=5, label='ΔX')
    ax.scatter(times/(60*60*24), diff[:,1], zorder=5, label='ΔY')
    ax.scatter(times/(60*60*24), diff[:,2], zorder=5, label='ΔZ')

    ax.scatter(times/(60*60*24), diff2[:,0], zorder=5, label='ΔX')
    ax.scatter(times/(60*60*24), diff2[:,1], zorder=5, label='ΔY')
    ax.scatter(times/(60*60*24), diff2[:,2], zorder=5, label='ΔZ')
    plt.title("Difference: J2000 - Earth-Centered Sun Frame (1 Day)") 
    ax.set_xlabel("Time [days from start]")
    ax.set_ylabel("Difference (km)")
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

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2 :param vec1: 
    A 3d "source" vector :param vec2: A 3d "destination" vector 
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2. """
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v) 
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

def init_ephem():
        """
        Initialize the ephemeris files for EMS
        """
        # This is for the Cassini example, comment out later
        #spice.furnsh("./Ephemeris/cassMetaK.txt")
        # Furnish the kernals we actually need
        spice.furnsh("./Ephemeris/ephemMeta.txt")

def get_planet_state(body, epoch, frame="J2000", observer="SUN"):
    et = spice.str2et(epoch)
    state, lt = spice.spkezr(body, et, frame, "NONE", observer)
    pos = np.array(state[:3])
    vel = np.array(state[3:])
    return pos, vel, et

if __name__ == '__main__':
    main()
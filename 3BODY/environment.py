import jax
import jax.numpy as jnp
import jax.random as jrand
from typing import NamedTuple, List # for SoAs/parallelization
import time
from GRPO import *
import functools

# units
# distance: meters
# mass: kg
# time: seconds

G = 6.6743e-11 # N * (m^2) / (kg^2)
dt = 3600*24#*7*52 # seconds

class SolarBody(NamedTuple):
    position : jax.Array # B, 3
    velocity : jax.Array # B, 3
    mass : jax.Array # B, 1
    radius : jax.Array # B, 1
# 
class SolarSystem(NamedTuple):
    bodies : List[SolarBody]

true_sun_mass = 1.9e30 # our suns size
true_sun_radius = 7e8
true_planet_mass = 5.9e24 # earth
true_simulation_size = 1.5e11 # meters # dist = from our sun to earth

downscaled_sun_radius = true_sun_radius / true_simulation_size
downscaled_simulation_size = true_simulation_size / true_simulation_size

conversion_to_downscaled_distance = downscaled_simulation_size / true_simulation_size
conversion_to_true_distance = true_simulation_size / downscaled_simulation_size

# mass: true
# momentum: true
# distances (radius, sim size): downscaled

# unused currently
def optimal_starting_momentum(planets, suns):
    # best momentum for producing orbits ~= average gravity * radius? (think swinging ball on rope)
    # momentum = gravity
    # avg gravity = (avg_m / G) * ((avg_m / r) / r)
    # avg m = planets * planet_mass + suns * sun_mass ) / (planets + suns)
    # avg r = (3*(true_sim_size / (planets + suns)))**(1/3)
    avg_r = (2/3) * (true_simulation_size / (planets + suns))*jnp.sqrt(3) # volumetric? 2/3 bh tri eq but for dist func from one corner to the other of a cube?
    avg_m = planets * (true_planet_mass / (planets + suns)) + suns * (true_sun_mass / (planets + suns))
    avg_gravity = (avg_m * G) * ((avg_m / avg_r) / avg_r)
    optimal_momentum = avg_gravity * avg_r
    return optimal_momentum


# jit because it gets reused to reinit the environment a lot
@functools.partial(jax.jit, static_argnames=["batches", "planets", "suns"])
def init_solarsystems(key, batches, planets, suns):
    # make 4 bodies and treat the first as the planet
    mass_scale = jrand.uniform(key, (batches, planets + suns,), minval=1/2, maxval=1)
    mass = mass_scale * jnp.array([true_planet_mass for _ in range(planets)] + [true_sun_mass for _ in range(suns)])[None, :]
    position = jrand.uniform(key, (batches, planets + suns, 3), minval=0, maxval=downscaled_simulation_size)
    #momentum_scale = optimal_starting_momentum(planets, suns)

    bodies = SolarBody(
        position=position,
        velocity=jrand.uniform(key, (batches, planets + suns, 3), minval=-0.5, maxval=0.5) * 50000, # velocity from -0.5 to 0.5 m/s
        mass=mass,
        radius = downscaled_sun_radius * (mass / true_sun_mass),
    )
    # return the solar body
    return SolarSystem(
        bodies=bodies
    )


# compute in a way that doesnt intermittently explode to inf
@jax.jit
def gravity(bodies, body1_idx, body2_idx):
    a = bodies.mass[:, body1_idx] * G
    b = bodies.mass[:, body2_idx]
    c = true_dist(bodies, body1_idx, body2_idx)**2
    min_dist = bodies.radius[:, body1_idx] # prevents issues
    return a * (b / (c + min_dist))

@jax.jit
def acceleration(bodies, body1_idx, body2_idx):
    # get the acceleration from gravity on body1.
    # (object_i.mass * object_j.mass * G / dist(object_i, object_j)^2) / object_i.mass
    # = object_j.mass * G / dist(object_i, object_j)^2)
    b = bodies.mass[:, body2_idx] * G
    c = true_dist(bodies, body1_idx, body2_idx)**2
    eps = bodies.radius[:, body1_idx]
    return b / (c + eps)


@jax.jit
def true_dist(bodies, body1_idx, body2_idx):
    a = bodies.position[:, body2_idx] - bodies.position[:, body1_idx]
    a = jnp.sqrt(jnp.sum(a*a, axis=-1))
    return a * conversion_to_true_distance


@jax.jit
def sim_dist(bodies, body1_idx, body2_idx):
    a = bodies.position[:, body2_idx] - bodies.position[:, body1_idx]
    a = jnp.sqrt(jnp.sum(a*a, axis=-1))
    return a

@jax.jit
def direction_vector(bodies, body1_idx, body2_idx):
    a = bodies.position[:, body2_idx] - bodies.position[:, body1_idx]
    b = jnp.sqrt(jnp.sum(a*a, axis=-1))
    return a / b[:, None]


@jax.jit
def get_velocity_unit_vector_from_action(action):
    return jnp.array([
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ], dtype=jnp.float32)[action]


@jax.jit
def step_simulation_forloop(solar_system : SolarSystem, action) -> SolarSystem:
    # calculate momentum updates for each
    # f = ma = mv / dt
    # mv = f * dt
        # calculate force on each object
        # f = m1 * m2 * G / r^2
        # for each object_i:
        #   for each other object_j:
        #       object_i.velocity += dt * (object_i.mass * object_j.mass * G / dist(object_i, object_j)^2) / object_i.mass
                    # TODO cancel out the object_i.masses
    for body1_idx in range(solar_system.bodies.position.shape[1]):
        for body2_idx in range(solar_system.bodies.position.shape[1]):
            if body1_idx == body2_idx:
                continue
            direction = direction_vector(solar_system.bodies, body1_idx, body2_idx)
            acceleration_from_gravity = acceleration(solar_system.bodies, body1_idx, body2_idx)
            velocity_change = dt * direction * acceleration_from_gravity
            new_velocity = solar_system.bodies.velocity.at[:, body1_idx].set(solar_system.bodies.velocity[:, body1_idx] + velocity_change)
            solar_system = SolarSystem(
                bodies = SolarBody(
                    velocity = new_velocity,
                    position=solar_system.bodies.position,
                    mass=solar_system.bodies.mass,
                    radius=solar_system.bodies.radius
                )
            ) # I don't think this is slow despite looking like it. I think the compiler figures out it doesnt need to move mem around
    # get host planet's momentum update from the agent's response to the solar system
    velocity_shift = get_velocity_unit_vector_from_action(action)
    agent_velocity_shift = 10000*velocity_shift
    # updates the solar system with a shift in the momentum of the agent planet

    # get position based off of momentum
    # for each object:
        # object.position += dt * object.velocity
    for body_idx in range(solar_system.bodies.velocity.shape[1]):
        if body_idx == 0:
            true_velocity = (agent_velocity_shift + solar_system.bodies.velocity[:, 0])
        else:
            true_velocity = solar_system.bodies.velocity[:, body_idx]
        downscaled_velocity = conversion_to_downscaled_distance * true_velocity
        downscaled_position_change = dt * downscaled_velocity
        new_position = solar_system.bodies.position.at[:, body_idx].set(
            solar_system.bodies.position[:, body_idx] +
            downscaled_position_change
        )
        #new_position = solar_system.bodies.position.at[:, body_idx].set(
        #    solar_system.bodies.position[:, body_idx] +
        #    dt * solar_system.bodies.velocity[:, body_idx]
        #)
        solar_system = SolarSystem(
            bodies = SolarBody(
                velocity = solar_system.bodies.velocity,
                position=new_position,
                mass=solar_system.bodies.mass,
                radius=solar_system.bodies.radius
            )
        )
    # debug_data = get_state_info(solar_system)
    return solar_system #, reward


@jax.jit
def step_simulation(solar_system : SolarSystem, action) -> SolarSystem:
    # calculate momentum updates for each
    # f = ma = mv / dt
    # mv = f * dt
        # calculate force on each object
        # f = m1 * m2 * G / r^2
        # for each object_i:
        #   for each other object_j:
        #       object_i.velocity += dt * (object_i.mass * object_j.mass * G / dist(object_i, object_j)^2) / object_i.mass (object_i cancels)
    
    # for loop -> vmap across indexes
    #def velocity_updates(body1_idx, body2_idx):
    n = solar_system.bodies.position.shape[1]
    body1_idxs = jnp.arange(n)
    body2_idxs = jnp.tile(jnp.arange(n-1), n).reshape(n, n-1) + jnp.triu(jnp.ones((n, n-1)))

        
    for body1_idx in range(solar_system.bodies.position.shape[1]):
        velocity_change = jnp.zeros_like(solar_system.bodies.velocity[:, body1_idx])
        for body2_idx in range(solar_system.bodies.position.shape[1]):
            if body1_idx == body2_idx:
                continue
            direction = direction_vector(solar_system.bodies, body1_idx, body2_idx)
            acceleration_from_gravity = acceleration(solar_system.bodies, body1_idx, body2_idx)
            # vec(batch, 3) = vec(batch, 3) + scalar*vec(batch, 3)*scalar(batch, 1)
            velocity_change = velocity_change + (dt * direction * acceleration_from_gravity[:, None])
        new_velocity = solar_system.bodies.velocity.at[:, body1_idx].set(solar_system.bodies.velocity[:, body1_idx] + velocity_change)
        solar_system = SolarSystem(
            bodies = SolarBody(
                velocity = new_velocity,
                position=solar_system.bodies.position,
                mass=solar_system.bodies.mass,
                radius=solar_system.bodies.radius
            )
        ) # I don't think this is slow despite looking like it. I think the compiler figures out it doesnt need to move mem around
    # get host planet's momentum update from the agent's response to the solar system
    velocity_shift = get_velocity_unit_vector_from_action(action)
    agent_velocity_shift = 1000000*velocity_shift
    # updates the solar system with a shift in the momentum of the agent planet

    # get position based off of momentum
    # for each object:
        # object.position += dt * object.velocity
    for body_idx in range(solar_system.bodies.velocity.shape[1]):
        if body_idx == 0:
            true_velocity = (agent_velocity_shift + solar_system.bodies.velocity[:, 0])
        else:
            true_velocity = solar_system.bodies.velocity[:, body_idx]
        downscaled_velocity = conversion_to_downscaled_distance * true_velocity
        downscaled_position_change = dt * downscaled_velocity
        new_position = solar_system.bodies.position.at[:, body_idx].set(
            solar_system.bodies.position[:, body_idx] +
            downscaled_position_change
        )
        #new_position = solar_system.bodies.position.at[:, body_idx].set(
        #    solar_system.bodies.position[:, body_idx] +
        #    dt * solar_system.bodies.velocity[:, body_idx]
        #)
        solar_system = SolarSystem(
            bodies = SolarBody(
                velocity = solar_system.bodies.velocity,
                position=new_position,
                mass=solar_system.bodies.mass,
                radius=solar_system.bodies.radius
            )
        )
    # debug_data = get_state_info(solar_system)
    return solar_system #, reward


# to add
# agent picks capped momentum update (agent just repeatedly picks 'up' for now)
# 

# sqrt(x^2 + y^2 + z^2)
@jax.jit
def l2_norm(pos):
    r = jnp.sqrt(jnp.sum(pos*pos, axis=-1))
    return r

@jax.jit
def get_reward(solar_system : SolarSystem) -> float:
    # takes in the current state and determines the reward
    # for now: is the planet too far from a sun? too close? etc
    # calculate heat
    # heat is proportional to light. light is inverse square of distance

    # get the distances from the home planet at idx 0
    home_planet_position = solar_system.bodies.position[0] # first planet's position (0:1 retains final dim)
    # home_planet_position = jnp.expand_dims(solar_system.bodies.position[:, 0, :], axis=1) # also works
    relative_positions_from_home_planet = solar_system.bodies.position - home_planet_position
    relative_true_distances_from_home_planet = l2_norm(relative_positions_from_home_planet)[1:] * conversion_to_true_distance #get true distances

    earth_wattage_per_sq_km = 1361
    earth_dist_from_sun = 1.5e11 # meters

    # get the ratio of sum of inverse square of distances to that of the earth to the sun
    inv_square_dist_ratio = lambda r: (earth_dist_from_sun / (r + 1e-7))**2
    inv_square_distance_ratios = jnp.sum(jax.lax.map(inv_square_dist_ratio, relative_true_distances_from_home_planet), axis=-1) # (sim, n)

    # goal wattage per square km: 1,361 (same as sun-earth relationship). do +/- 20% idk
    wattage_per_square_km = jnp.sum(earth_wattage_per_sq_km * inv_square_distance_ratios)

    ideal_wattage_per_square_km = 1361

    death_by_heat_wattage = 2 * ideal_wattage_per_square_km
    death_by_freezing_wattage = ideal_wattage_per_square_km / 2



    # simple reward based on distance from ideal wattage (goldilocks zone)
    temp_margin = ideal_wattage_per_square_km/4
    T_rel = wattage_per_square_km - ideal_wattage_per_square_km
    reward = jax.nn.relu(1 - jnp.abs(T_rel)/temp_margin)

    return reward




@jax.jit
def get_state_summary(solar_system : SolarSystem) -> float:
    # takes in the current state and determines the reward
    # for now: is the planet too far from a sun? too close? etc
    # calculate heat
    # heat is proportional to light. light is inverse square of distance

    # get the distances from the home planet at idx 0
    relative_positions_from_home_planet = solar_system.bodies.position - solar_system.bodies.position[:, 0]
    relative_true_distances_from_home_planet = l2_norm(relative_positions_from_home_planet)[:, 1:] * conversion_to_true_distance #get true distances

    earth_wattage_per_sq_km = 1361
    earth_dist_from_sun = 1.5e11 # meters

    # get the ratio of sum of inverse square of distances to that of the earth to the sun
    inv_square_dist_ratio = lambda r: (earth_dist_from_sun / (r + 1e-7))**2
    inv_square_distance_ratios = jnp.sum(jax.lax.map(inv_square_dist_ratio, relative_true_distances_from_home_planet), axis=-1) # (sim, n)

    # goal wattage per square km: 1,361 (same as sun-earth relationship). do +/- 20% idk
    wattage_per_square_km = jnp.sum(earth_wattage_per_sq_km * inv_square_distance_ratios)

    ideal_wattage_per_square_km = 1361

    # simple reward based on distance from ideal wattage
    reward = 1 - abs(ideal_wattage_per_square_km - wattage_per_square_km) / ideal_wattage_per_square_km

    return (wattage_per_square_km, relative_true_distances_from_home_planet)




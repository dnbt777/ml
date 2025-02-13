import jax
import jax.numpy as jnp
import jax.random as jrand
from typing import NamedTuple, List # for SoAs/parallelization
import functools
from GRPO import *
import functools


# SIMULATION
# mass: true
# momentum: true
# distances (radius, sim size, velocity/accel/dist): downscaled


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

default_body_velocity = 29*1000 # m/s, earth orbit speed
downscaled_default_body_velocity = default_body_velocity * conversion_to_downscaled_distance # m/s, earth orbit speed

agent_body_velocity = 1000 # calculate to make realistic. google mass/energy efficiency
downscaled_agent_body_velocity = agent_body_velocity * conversion_to_downscaled_distance




# jit because it gets reused to reinit the environment a lot
@functools.partial(jax.jit, static_argnames=["batches", "planets", "suns"])
def init_solarsystems(key, batches, planets, suns):
    # make 4 bodies and treat the first as the planet
    mass_scale = jrand.uniform(key, (batches, planets + suns,), minval=1/2, maxval=1)
    mass = mass_scale * jnp.array([true_planet_mass for _ in range(planets)] + [true_sun_mass for _ in range(suns)])[None, :]
    position = jrand.uniform(key, (batches, planets + suns, 3), minval=0, maxval=downscaled_simulation_size)
    #momentum_scale = optimal_starting_momentum(planets, suns)

    bodies = SolarBody(
        position=position, # m, scaled dowm
        velocity=jrand.uniform(key, (batches, planets + suns, 3), minval=-1, maxval=1) * downscaled_default_body_velocity, # m/s, scaled down
        mass=mass, # kg, true
        radius = downscaled_sun_radius * (mass / true_sun_mass), # m, downscaled
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


# must be vmapped. unbatched.
# from now on i think naming should be _batched and _unbatched.

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
    # meshgrid-like, but add ones-triu to create set of all combinations without body1
    #body1_idxs = jnp.tile(jnp.arange(n, dtype=jnp.uint8).reshape(1, n), n-1).reshape(n-1, n).transpose() # probably a simpler way to do this lol
    body1_idxs = jnp.arange(n, dtype=jnp.uint8)
    body2_idxs = jnp.tile(jnp.arange(n-1, dtype=jnp.uint8), n).reshape(n, n-1) + jnp.triu(jnp.ones((n, n-1), dtype=jnp.uint8))

    #body1_idxs = jnp.ravel(body1_idxs)
    #body2_idxs = jnp.ravel(body2_idxs)

    # honestly.. the indexes are small enough (0-3) that I should just jit with static args here
    # does not scale as planets scale (massive compile times). but we arent scaling planets.
    @functools.partial(jax.jit, static_argnames=["body1_idx", "body2_idx"])
    def get_change_in_velocity(solar_system, body1_idx, body2_idx):
        # get direction
        difference_vector = solar_system.bodies.position[body2_idx] - solar_system.bodies.position[body1_idx]
        #pos2 = jax.lax.dynamic_index_in_dim(solar_system.bodies.position, body2_idx, axis=1)
        #pos1 = jax.lax.dynamic_index_in_dim(solar_system.bodies.position, body1_idx, axis=1)
        #difference_vector = pos2 - pos1
        downscaled_distance = jnp.sqrt(jnp.sum(difference_vector*difference_vector, axis=-1))
        unit_direction_vector = difference_vector / jnp.expand_dims(downscaled_distance, -1)
        # get acceleration
        ab = solar_system.bodies.mass[body2_idx] * G
        radius_squared = (downscaled_distance * conversion_to_true_distance)**2 # potential source of nans/infs
        epsilon = solar_system.bodies.radius[body1_idx] + solar_system.bodies.radius[body2_idx] # prevents infs in divide
        acceleration_from_gravity = ab / (radius_squared + epsilon)
        downscaled_acceleration_from_gravity = conversion_to_downscaled_distance * acceleration_from_gravity
        # get velocity
        downscaled_velocity_change = dt * unit_direction_vector * jnp.expand_dims(downscaled_acceleration_from_gravity, -1)
        return downscaled_velocity_change
    
    # associative scan might be faster here
    # mapreducing gravity(b1, b2) over [b2] is associative (+)
    fmap_over_body2 = jax.vmap(get_change_in_velocity, in_axes=(None, None, 0))
    fmap_over_body1 = jax.vmap(fmap_over_body2, in_axes=(None, 0, 0))
    fmap_over_solar_system_batches = jax.vmap(fmap_over_body1, in_axes=(0, None, None))
    downscaled_total_velocity_change = jnp.sum(fmap_over_solar_system_batches(solar_system, body1_idxs, body2_idxs), axis=-1)
    
    new_downscaled_velocity = solar_system.bodies.velocity + downscaled_total_velocity_change
    solar_system = SolarSystem(
        bodies = SolarBody(
            velocity = new_downscaled_velocity, # downscaled
            position = solar_system.bodies.position, # downscaled
            mass = solar_system.bodies.mass, # true
            radius = solar_system.bodies.radius # downscaled
        )
    )
    
    # get host planet's momentum update from the agent's response to the solar system
    downscaled_agent_velocity_shift = get_velocity_unit_vector_from_action(action) * downscaled_agent_body_velocity
    solar_system = SolarSystem(
            bodies = SolarBody(
                velocity = solar_system.bodies.velocity.at[:, 0].set(solar_system.bodies.velocity[:, 0] + downscaled_agent_velocity_shift),
                position = solar_system.bodies.position,
                mass = solar_system.bodies.mass,
                radius = solar_system.bodies.radius
            )
        )
    # updates the solar system with a shift in the momentum of the agent planet

    # get position based off of momentum
    # for each object:
        # object.position += dt * object.velocity
    @functools.partial(jax.jit, static_argnames=["body_idx"])
    def update_position(solar_system, body_idx):
        downscaled_velocity = solar_system.bodies.velocity[:, body_idx]
        downscaled_position_change = dt * downscaled_velocity
        new_position = solar_system.bodies.position.at[:, body_idx].set(
            solar_system.bodies.position[:, body_idx] +
            downscaled_position_change
        )

        solar_system = SolarSystem(
            bodies = SolarBody(
                velocity = solar_system.bodies.velocity,
                position = new_position,
                mass = solar_system.bodies.mass,
                radius = solar_system.bodies.radius
            )
        )
        return solar_system, None
    
    solar_system, _ = jax.lax.scan(update_position, solar_system, jnp.arange(n, dtype=jnp.uint8))

    return solar_system


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
    relative_true_distances_from_home_planet = l2_norm(relative_positions_from_home_planet)[1:] * conversion_to_true_distance #get true distances to all suns

    earth_wattage_per_sq_km = 1361
    earth_dist_from_sun = 1.5e11 # meters

    # get the ratio of sum of inverse square of distances to that of the earth to the sun
    inv_square_dist_ratio = lambda r: (earth_dist_from_sun / (r + 1e-7))**2
    inv_square_distance_ratios = jnp.sum(jax.lax.map(inv_square_dist_ratio, relative_true_distances_from_home_planet), axis=-1) # (sim, n)

    # goal wattage per square km: 1,361 (same as sun-earth relationship). do +/- 20% idk
    wattage_per_square_km = jnp.sum(earth_wattage_per_sq_km * inv_square_distance_ratios)

    ideal_wattage_per_square_km = 1361
    death_margin = ideal_wattage_per_square_km / 3

    base = jax.nn.tanh((wattage_per_square_km - ideal_wattage_per_square_km) / (2 * death_margin))
    reward = 1 - 2*(base * base * base * base) # compileable two sided. too hot = bad, too cold = bad
    return reward



@jax.jit
def get_state_summary(solar_system : SolarSystem) -> float:
    relative_positions_from_home_planet = solar_system.bodies.position - jnp.expand_dims(solar_system.bodies.position[:, 0], -2)
    relative_true_distances_from_home_planet = l2_norm(relative_positions_from_home_planet)[:, 1:] * conversion_to_true_distance #get true distances

    earth_wattage_per_sq_km = 1361
    earth_dist_from_sun = 1.5e11 # meters

    # get the ratio of sum of inverse square of distances to that of the earth to the sun
    inv_square_dist_ratio = lambda r: (earth_dist_from_sun / (r + 1e-7))**2
    inv_square_distance_ratios = jnp.sum(jax.lax.map(inv_square_dist_ratio, relative_true_distances_from_home_planet), axis=-1) # (sim, n)

    # goal wattage per square km: 1,361 (same as sun-earth relationship). do +/- 20% idk
    wattage_per_square_km = jnp.sum(earth_wattage_per_sq_km * inv_square_distance_ratios)

    return (wattage_per_square_km, relative_true_distances_from_home_planet)




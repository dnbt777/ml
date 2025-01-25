import jax
import jax.numpy as jnp
import jax.random as jrand
from typing import NamedTuple, List # for SoAs/parallelization
import time

# units
# distance: meters
# mass: kg
# time: seconds

G = 6.6743e-11 # N * (m^2) / (kg^2)
dt = 3600*24#*7*52 # seconds

class SolarBody(NamedTuple):
    position : jax.Array # B, 3
    momentum : jax.Array # B, 3
    mass : jax.Array # B, 1
    radius : jax.Array # B, 1


class SolarSystem(NamedTuple):
    bodies : List[SolarBody]

true_sun_mass = 1.9e30 # our suns size
true_sun_radius = 7e8
true_planet_mass = 5.9e24 # earth
true_simulation_size = 1.4e11 * 10 # meters # same size as our solar system

downscaled_sun_radius = true_sun_radius / true_simulation_size
downscaled_simulation_size = true_simulation_size / true_simulation_size

conversion_to_downscaled_distance = downscaled_simulation_size / true_simulation_size
conversion_to_true_distance = true_simulation_size / downscaled_simulation_size

# mass: true
# momentum: true
# distances (radius, sim size): downscaled

debug = False
if debug:
    jax.config.update("jax_disable_jit", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)



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
def init_solarsystems(key, batches, planets, suns):
    # make 4 bodies and treat the first as the planet
    mass_scale = jrand.uniform(key, (batches, planets + suns,), minval=1/2, maxval=1)
    mass = mass_scale * jnp.array([true_planet_mass for _ in range(planets)] + [true_sun_mass for _ in range(suns)])[None, :]
    position = jrand.uniform(key, (batches, planets + suns, 3), minval=0, maxval=downscaled_simulation_size)
    #momentum_scale = optimal_starting_momentum(planets, suns)

    bodies = SolarBody(
        position=position,
        momentum=jrand.uniform(key, (batches, planets + suns, 3), minval=-0.5, maxval=0.5) * mass[:, :, None], # velocity from -0.5 to 0.5 m/s
        mass = mass,
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
    c = dist_for_gravity(bodies, body1_idx, body2_idx)**2
    min_dist = bodies.radius[:, body1_idx] # prevents issues
    return a * (b / (c + min_dist))


@jax.jit
def dist_for_gravity(bodies, body1_idx, body2_idx):
    a = bodies.position[:, body2_idx] - bodies.position[:, body1_idx]
    a = jnp.sqrt(jnp.sum(a*a, axis=-1))
    return a * conversion_to_true_distance


@jax.jit
def direction_vector(bodies, body1_idx, body2_idx):
    a = bodies.position[:, body2_idx] - bodies.position[:, body1_idx]
    b = jnp.sqrt(jnp.sum(a*a, axis=-1))
    return a / b[:, None]


@jax.jit
def apply_nuke_dummy_agent_1(key, solar_system : SolarSystem) -> SolarSystem:
    # get momentum change based on solar_system
    # agent(solar_system) -> first_planet_momentum_update
    # for now just do up in the y axis. i.e. [0, 1.0, 0]
    first_planet_momentum_shift = jnp.ones_like(solar_system.bodies.momentum[:, 0]) * jnp.array([0, 100.0, 0])[None, :] * solar_system.bodies.mass[:, 0] # all batches, first planet * no batches (broadcast)
    new_first_planet_momentum = solar_system.bodies.momentum[:, 0] + first_planet_momentum_shift
    new_momentum = solar_system.bodies.momentum.at[:, 0].set(new_first_planet_momentum)
    solar_system = SolarSystem(
        bodies=SolarBody(
            position = solar_system.bodies.position,
            momentum = new_momentum,
            mass = solar_system.bodies.mass,
            radius = solar_system.bodies.radius
        )
    )
    return solar_system


@jax.jit
def apply_nuke_dummy_agent_2(key, solar_system : SolarSystem) -> SolarSystem:
    # get momentum change based on solar_system
    # agent(solar_system) -> first_planet_momentum_update
    # for now just do up in the y axis. i.e. [0, 1.0, 0]
    randspeed = 3000
    momentum_shape = solar_system.bodies.momentum[:, 0].shape
    first_planet_momentum_shift = randspeed * jrand.uniform(key, momentum_shape, minval=-1, maxval=1) * solar_system.bodies.mass[:, 0] # all batches, first planet * no batches (broadcast)
    new_first_planet_momentum = solar_system.bodies.momentum[:, 0] + first_planet_momentum_shift
    new_momentum = solar_system.bodies.momentum.at[:, 0].set(new_first_planet_momentum)
    solar_system = SolarSystem(
        bodies=SolarBody(
            position = solar_system.bodies.position,
            momentum = new_momentum,
            mass = solar_system.bodies.mass,
            radius = solar_system.bodies.radius
        )
    )
    return solar_system


@jax.jit
def step_simulation(key, solar_system : SolarSystem) -> SolarSystem:
    # calculate momentum updates for each
    # f = ma = mv / dt
    # mv = f * dt
        # calculate force on each object
        # f = m1 * m2 * G / r^2
        # for each object_i:
        #   for each other object_j:
        #       object_j.momentum += dt * (object_i.mass * object_j.mass * G / dist(object_i, object_j)^2)
    for body1_idx in range(solar_system.bodies.position.shape[1]):
        for body2_idx in range(solar_system.bodies.position.shape[1]):
            if body1_idx == body2_idx:
                continue
            pair_gravity = gravity(solar_system.bodies, body1_idx, body2_idx)
            direction = direction_vector(solar_system.bodies, body1_idx, body2_idx)
            momentum_change = dt * direction * pair_gravity[:, None]
            new_momentum = solar_system.bodies.momentum.at[:, body1_idx].set(solar_system.bodies.momentum[:, body1_idx] + momentum_change)
            solar_system = SolarSystem(
                bodies = SolarBody(
                    momentum = new_momentum,
                    position=solar_system.bodies.position,
                    mass=solar_system.bodies.mass,
                    radius=solar_system.bodies.radius
                )
            ) # I don't think this is slow despite looking like it. I think the compiler figures out it doesnt need to move mem around
    
    # get host planet's momentum update from the agent's response to the solar system
    solar_system = apply_nuke_dummy_agent_2(key, solar_system) # updates the solar system with a shift in the momentum of the agent planet

    # get position based off of momentum
    # for each object:
        # object.position += dt * object.momentum / object.mass
    for body_idx in range(solar_system.bodies.momentum.shape[1]):
        true_velocity = solar_system.bodies.momentum[:, body_idx] / solar_system.bodies.mass[:, body_idx, None]
        downscaled_velocity = conversion_to_downscaled_distance * true_velocity
        downscaled_position_change = dt * downscaled_velocity
        new_position = solar_system.bodies.position.at[:, body_idx].set(
            solar_system.bodies.position[:, body_idx] +
            downscaled_position_change
        )
        #new_position = solar_system.bodies.position.at[:, body_idx].set(
        #    solar_system.bodies.position[:, body_idx] +
        #    dt * solar_system.bodies.momentum[:, body_idx] / (solar_system.bodies.mass[:, body_idx, None])
        #)
        solar_system = SolarSystem(
            bodies = SolarBody(
                momentum = solar_system.bodies.momentum,
                position=new_position,
                mass=solar_system.bodies.mass,
                radius=solar_system.bodies.radius
            )
        )
    return solar_system




# to add
# agent picks capped momentum update (agent just repeatedly picks 'up' for now)
# 

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
dt = 0.5 # seconds

class SolarBody(NamedTuple):
    position : jax.Array # B, 3
    momentum : jax.Array # B, 3
    mass : jax.Array # B, 1
    radius : jax.Array # B, 1


class SolarSystem(NamedTuple):
    bodies : List[SolarBody]

true_sun_mass = 1.9e30 # uh
true_sun_radius = 7e8
true_planet_mass = 5.9e24
true_simulation_size = 1.4e11 * 1 # meters

sun_mass = true_sun_mass
planet_mass = true_planet_mass
sun_radius = true_sun_radius / true_simulation_size
simulation_size = true_simulation_size / true_simulation_size

# may need to jit this
def init_solarsystems(key, batches, planets, suns):
    # make 4 bodies and treat the first as the planet
    mass = jrand.uniform(key, (batches, planets + suns,), minval=1/2, maxval=1)
    mass = mass * jnp.array([planet_mass for _ in range(planets)] + [sun_mass for _ in range(suns)])[None, :]

    position = jrand.uniform(key, (batches, planets + suns, 3), minval=0, maxval=simulation_size)

    bodies = SolarBody(
        position=position,
        momentum=jrand.uniform(key, (batches, planets + suns, 3), minval=0, maxval=1) * mass[:, :, None] * 0.01, # force 0 momentum at start for now,
        mass = mass,
        radius = sun_radius * mass / sun_mass,
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
    c = dist(bodies, body1_idx, body2_idx)**2
    min_dist = bodies.radius[:, body1_idx] # prevents issues
    return a * (b / (c + min_dist))

@jax.jit
def dist(bodies, body1_idx, body2_idx):
    a = bodies.position[:, body2_idx] - bodies.position[:, body1_idx]
    a = jnp.sqrt(jnp.sum(a*a, axis=-1))
    return a * true_simulation_size

@jax.jit
def direction_vector(bodies, body1_idx, body2_idx):
    a = bodies.position[:, body2_idx] - bodies.position[:, body1_idx]
    b = jnp.sqrt(jnp.sum(a*a, axis=-1))
    return a / b[:, None]


@jax.jit
def step_simulation(solar_system : SolarSystem) -> SolarSystem:
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
            #print(f"old_momentum: {solar_system.bodies.momentum[0, body1_idx]}")
            #print(f"new_momentum: {solar_system.bodies.momentum[0, body1_idx] + dt * pair_gravity}")
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
    # get position based off of momentum
    # for each object:
        # object.position += dt * object.momentum / object.mass
    for body_idx in range(solar_system.bodies.momentum.shape[1]):
        new_position = solar_system.bodies.position.at[:, body_idx].set(
            solar_system.bodies.position[:, body_idx] +
            dt * solar_system.bodies.momentum[:, body_idx] / (solar_system.bodies.mass[:, body_idx, None])
        )
        solar_system = SolarSystem(
            bodies = SolarBody(
                momentum = solar_system.bodies.momentum,
                position=new_position,
                mass=solar_system.bodies.mass,
                radius=solar_system.bodies.radius
            )
        )
    return solar_system






import jax
import jax.numpy as jnp
import jax.random as jrand
from typing import NamedTuple, List # for SoAs/parallelization
import time


class SolarBody(NamedTuple):
    position : jax.Array # B, 3
    momentum : jax.Array # B, 3
    mass : jax.Array # B, 1
    radius : jax.Array # B, 1


class SolarSystem(NamedTuple):
    bodies : List[SolarBody]



def init_solarsystems(key, batches):
    # make 4 bodies and treat the first as the planet
    suns = 7
    planets = 1
    bodies = SolarBody(
        position=jrand.uniform(key, (batches, planets + suns, 3)),
        momentum=jrand.uniform(key, (batches, planets + suns, 3)),
        mass = jrand.uniform(key, (batches, planets + suns,)),
        radius = jrand.uniform(key, (batches, planets + suns,))
    )
    # return the solar body
    return SolarSystem(
        bodies=bodies
    )


key = jrand.PRNGKey(int(10000*time.time()))
simultaneous_simulations = 100
solarsystems = init_solarsystems(key, simultaneous_simulations)


# jax.jit
def step_simulation(solar_system : SolarSystem) -> SolarSystem:
    G = 6.6743e-11 # N * (m^2) / (kg^2)
    dt = 1e-4

    # calculate momentum updates for each
    # f = ma = mv / dt
    # mv = f * dt
        # calculate force on each object
        # f = m1 * m2 * G / r^2
        # for each object_i:
        #   for each other object_j:
        #       object_j.momentum += dt * (object_i.mass * object_j.mass * G / dist(object_i, object_j)^2)
    dist = lambda body1, body2: jnp.sqrt(
        (body1.position[0] - body2.position[0])**2 +
        (body1.position[1] - body2.position[1])**2 +
        (body1.position[2] - body2.position[2])**2
    )
    gravity = lambda body1, body2: body1.mass * body2.mass * G / (dist(body1, body2)**2)

    for body1_idx in range(solar_system.bodies.shape[1]):
        for body2_idx in range(solar_system.bodies.shape[1]):
            solar_system.bodies.momentum[:, body1_idx] += dt * gravity(solar_system.bodies[:, body1_idx], solar_system.bodies[:, body2_idx])

    # get position based off of momentum
    # for each object:
        # object.position += dt * object.momentum / object.mass
    for body_idx in range(solar_system.bodies.shape[1]):
        solar_system.bodies.position[:, body_idx] += dt * solar_system.bodies.momentum[:, body_idx] / solar_system.bodies.mass[:, body_idx]
    
    return solar_system


# hmm.. i need some way to scale this down due to the planets being so large...





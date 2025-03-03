""""""solving"""""""" the 3 body problem - achieving habitability externally
you live on a planet that orbits 3 suns. this sucks. your people burn or freeze to death or fall into the sun
historically, predicting these chaotic eras never works. 
but maybe you can alter them! after all, you are a mass.



make 3 suns and 1 planet
initialize randomly or something

every frame:
    give the transformer the positions and momentum of each sun/planet
    let it apply a tiny momentum vector to the planet

    for every thing:
        calculate net G force at the center of every sun/planet
        update momentum
        update position from momentum



win conditions:

MAKING A HOME
exactly 2 suns are expelled or the planet is expelled with exactly 1 sun. sorted(distances_to_suns)[1] > max_win_threshold
    BONUS POINTS:
        after 'winning': the planet does not fall into its sun
        after 'winning': the planet is orbiting in its goldilocks zone (min_dist_from_parent_sun, max_dist_from_parent_sun)
        after 'winning': the planet has the above two conditions and is in a stable repeating orbit for {threshold_yrs}
    SUPER BONUS POINTS
        the planet NEVER faces a soft death condition (cold era/hot era/gravity death)


ETERNAL PLATE SPINNING
the planet does not have a soft DEATH condition for a time period 1000 standard deviations longer than the average
    - do a ton of random runs with no jet to calculate the average time and variance of the survival times
    - survive longer than 1000 standard deviations above the average


DEATH conditions:
    planet colliding with a sun is immediate death always
soft death conditions
    too far from a sun (cold era)
    too close to a sun (hot era)
    lined up with three suns (gravity death)
other:
    if two suns collide, the debris is too much to handle, and you insta die. lazy version.
    if two suns collide, they turn into one sun with 1.5x the mass and 10 suns with 0.05x the mass to simulate dangerous debris. this will likely kill you. if even one touches your planet, you're dead. harder to code version.


reward function: 
    idk

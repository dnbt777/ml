

color_black = (0, 0, 0)
color_grey = (130, 130, 130)
color_red = (255, 0, 0)

color_jax_purple_light = (234, 128, 252)
color_jax_purple_med = (0, 105, 92)
color_jax_purple_dark = (106, 27, 154)
color_jax_blue_light = (94, 151, 246)
color_jax_blue_dark = (42, 86, 198)
color_jax_green_light = (38, 166, 154)
color_jax_green_dark = (0, 105, 92)


def color_string(s, color):
    r, g, b = color
    return f"\x1b[38;2;{r};{g};{b}m{s}\x1b[0m"

def print_color(*msgs, color=color_black, end="\n"):
    r, g, b = color
    msg = " ".join(msgs)
    print(f"\x1b[38;2;{r};{g};{b}m{msg}\x1b[0m", end=end, flush=True)
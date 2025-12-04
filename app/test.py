from build_3d_geometry import make_curved_channel_section_with_spherical_hole

R = 5
H = 1
W = 1
a = 0.025
L = 4
particle_maxh = 0.2*a
global_maxh = 0.3*H

make_curved_channel_section_with_spherical_hole(R, H, W, L, a, particle_maxh, global_maxh)


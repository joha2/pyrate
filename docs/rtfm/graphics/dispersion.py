#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pyrate - Optical raytracing based on Python

Copyright (C) 2014-2020
               by     Moritz Esslinger moritz.esslinger@web.de
               and    Johannes Hartung j.hartung@gmx.net
               and    Uwe Lippmann  uwe.lippmann@web.de
               and    Thomas Heinze t.heinze@uni-jena.de
               and    others

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""


import numpy as np
from numpy import pi
import matplotlib.pyplot as plt


phi = np.linspace(0,2*pi,10000, endpoint=True)

head_size = .1
linewidth = 3
xlim = 1.5






def make_fig():    
    fig = plt.figure(figsize = [55/25.4,55/25.4], dpi = 300)
    ax  = fig.add_subplot("111") 
    
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Roman"]})
    
    half_arrow_len = xlim-1.4*head_size # start point of the arrows

    ax.text(xlim,-.7*head_size,"$\Re k_x$", horizontalalignment="right", verticalalignment="top", fontsize=10)
    ax.text(-.7*head_size,xlim,"$\Re k_y$", horizontalalignment="right", verticalalignment="top", fontsize=10)

    ax.arrow(x=-half_arrow_len, y=0, dx=2*half_arrow_len, dy=0, head_length=head_size, head_width=head_size, color="black")
    ax.arrow(x=0, y=-half_arrow_len, dx=0, dy=2*half_arrow_len, head_length=head_size, head_width=head_size, color="black")

    ax.set_xlim([-xlim,xlim])
    ax.set_ylim([-xlim,xlim])
    ax.axis("off")
    ax.set_aspect("equal")
    
    return fig,ax


kx = np.cos(phi)
ky = np.sin(phi)

# isotropic
fig1, ax1 = make_fig()
ax1.plot(kx,ky, linewidth=linewidth)
fig1.savefig("dispersion_isotropic.jpg", bbox_inches="tight")
fig1.savefig("dispersion_isotropic.eps", bbox_inches="tight")

# uniaxial anisotropic
fig2, ax2 = make_fig()
ax2.plot(kx,ky, linewidth=linewidth)
ax2.plot(.8*kx,ky, linewidth=linewidth)
fig2.savefig("dispersion_uniaxial.jpg", bbox_inches="tight")
fig2.savefig("dispersion_uniaxial.eps", bbox_inches="tight")

# uniaxial anisotropic with k_inplane
fig3, ax3 = make_fig()
ax3.plot(kx,ky, linewidth=linewidth)
ax3.plot(.8*kx,ky, linewidth=linewidth)
phi_inplane = .9
x_inplane   = np.cos(phi_inplane)
y_inplane   = np.sin(phi_inplane)
ax3.plot([x_inplane,x_inplane],[-y_inplane,y_inplane])
fig3.savefig("dispersion_kinplane.jpg", bbox_inches="tight")
fig3.savefig("dispersion_kinplane.eps", bbox_inches="tight")


plt.show()

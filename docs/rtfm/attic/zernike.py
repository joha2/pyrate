"""
 This program calculates the aberration dependence of contrast in DIC microscopy
 Copyright (C) 2015 Moritz Esslinger
 moritz.esslinger@web.de

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License along
 with this program; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
import json
import sys

import numpy as np
import sympy
from math import factorial
from math import comb as binomial_coefficient


def assert_list_equal( l1:list, l2:list):
    assert len(l1) == len(l2)
    for i in range(len(l1)):
        assert l1[i] == l2[i], f"Error at element {i}: {l1[i]} != {l2[i]}"
    
def assert_almost_equal(a,b,tolerance=1E-10):
    assert abs(a-b) <= tolerance



def standard_indices_to_fringe( n: int, l:int ) -> int:
    j = (1 + ((n+abs(l))//2) )**2 - 2*abs(l) + (1 -np.sign(l)) // 2
    return int(j)


def fringe_to_standard_index( j_fringe: int ) -> (int,int):
    nex     = int( np.ceil( np.sqrt( j_fringe ) ) )
    next_sq = nex**2 # next larger square number

    absl_plus_n = 2*nex - 2
 
    absl = int( np.ceil( ( next_sq - j_fringe ) / 2 ) )
    n = absl_plus_n - absl

    signl = (-1 if (next_sq - j_fringe) % 2  else 1)
    l = absl * signl
    
    return n, l


def zernike_radial_polynomial_coefficients( n:int, l:int) -> list[float]:
    absl = abs(l)
    p = np.zeros( n+1, dtype=int )
    for i in range(abs(l),n+1,2):
        k = (n-i)//2;
        p[i] = (-1)**k * factorial(n-k) / ( factorial(k) * factorial((n+absl)//2 - k) * factorial((n-absl)//2-k) )
    return p


def is_cosine( n:int, l:int) -> bool:
    return (l > 0)


def angular_frequency( n:int, l:int) -> int:
    return abs(l)


def zernike_polynomial_in_polar_representation( n:int, l:int) -> sympy.core.symbol.Symbol:
    r   = sympy.Symbol("r")
    phi = sympy.Symbol("phi")
    
    rad_coeff = zernike_radial_polynomial_coefficients( n=n, l=l )
    rad_poly  = sum( list( rad_coeff[i] * r**i for i in range(len(rad_coeff))[::-1]  ) )
    omega     = angular_frequency(n=n, l=l)
    
    if omega:
        sine_term = ( sympy.cos(omega * phi) if is_cosine(n=n,l=l) else sympy.sin(omega*phi) ) 
    else:
        sine_term = 1
        
    zernike_polynomial = rad_poly * sine_term
        
    return zernike_polynomial


def sin_omega_phi_to_sin_phi(omega:int) -> ( sympy.core.symbol.Symbol, sympy.core.symbol.Symbol):
    phi = sympy.Symbol("phi")
    terms = list( (-1)**j * binomial_coefficient( omega, 2*j+1 ) *
            (sympy.sin(phi))**(2*j+1) *
            (sympy.cos(phi))**(omega-2*j-1)
        for j in range( (omega-1)//2 + 1) )
    return sympy.sin(omega*phi), sum(terms)

def cos_omega_phi_to_cos_phi(omega:int) -> ( sympy.core.symbol.Symbol, sympy.core.symbol.Symbol):
    phi = sympy.Symbol("phi")
    terms = list( (-1)**j * binomial_coefficient(omega, 2*j) * 
                  (sympy.sin(phi))**(2*j) * 
                  (sympy.cos(phi))**(omega-2*j) 
                  for j in range( omega//2 + 1 ))
    return sympy.cos(omega*phi), sum(terms)


def zernike_polynomial_in_cartesian_representation( n:int, l:int ) -> sympy.core.symbol.Symbol:
    polar = zernike_polynomial_in_polar_representation( n=n, l=l)
    omega = angular_frequency( n=n, l=l)
    phi   = sympy.Symbol("phi")
    r     = sympy.Symbol("r")
    x     = sympy.Symbol("x")
    y     = sympy.Symbol("y")
    
    subs_rule          = ( cos_omega_phi_to_cos_phi(omega=omega) if is_cosine(n=n,l=l) else sin_omega_phi_to_sin_phi(omega=omega))
    poly_without_omega = polar.subs( *subs_rule )
    poly_without_phi   = poly_without_omega.subs( sympy.cos(phi), x/r ).subs( sympy.sin(phi), y/r )
    poly_cartesian     = poly_without_phi.subs( r**2, x**2+y**2 ).simplify().expand()

    for i in range(5):
        poly_cartesian = poly_cartesian.subs( r**2, x**2+y**2 ).simplify().expand()

    return poly_cartesian


def cartesian_coefficients_by_xy_order(n:int, l:int)-> dict:
    cartesian = zernike_polynomial_in_cartesian_representation(n=n,l=l)

    coefficients_by_order = []

    terms = str(cartesian).replace(" ","").replace("-", "+-").split("+")
    for term in terms:
        coeff = term.split("*")[0].replace("x","",).replace("y","").strip()
        if coeff=="":
            coeff = 1
        elif coeff == "-":
            coeff = -1
        else:
            coeff = int(coeff)

        if "x**" in term:
            xorder = int(term.split("x**")[1].split("*")[0])
        elif "x" in term:
            xorder = 1
        else:
            xorder = 0
            
        if "y**" in term:
            yorder = int(term.split("y**")[1].split("*")[0])
        elif "y" in term:
            yorder = 1
        else:
            yorder = 0
            
        coefficients_by_order.append({"c":coeff, "x**":xorder, "y**":yorder})
   
    return coefficients_by_order
      


def aberration_name(n:int, l:int) -> str:
    j = standard_indices_to_fringe( n=n, l=l )
    if j == 1:
        name = "piston"
    elif j in [2,3]:
        name = "tilt"
    elif j == 4:
        name = "defocus"
    else:
        omega = angular_frequency( n=n, l=l)
        if omega == 0:
            name = "spherical aberration"
        elif omega == 1:
            name = "coma"
        elif omega == 2:
            name = "astigmatism"
        else:
            name = str(omega)+ "-foil"
        nex = int( np.ceil( np.sqrt( j ) ) )
        order = 2 * nex - 3
        name += " " + str(order) + ("rd" if order==3 else "th") + " order"
        
    return name




def full_story(n:int, l:int)-> dict:
    j = standard_indices_to_fringe( n=n, l=l )
    polar     = zernike_polynomial_in_polar_representation(n=n,l=l)
    cartesian = zernike_polynomial_in_cartesian_representation(n=n,l=l)
    dzdx      = sympy.diff( cartesian, sympy.Symbol("x"))
    dzdy      = sympy.diff( cartesian, sympy.Symbol("y"))
    print(f"j={j}, n={n}, l={l}")
    dic = {
        "name": aberration_name(n=n,l=l),
        "standard indices": (n,l),
        "fringe index": j,
        "radial coefficients": zernike_radial_polynomial_coefficients(n=n,l=l).tolist(),
        "angular function": ("cos" if is_cosine( n=n,l=l) else "sin"),
        "angular frequency": angular_frequency( n=n, l=l),
        "cartesian coefficients": cartesian_coefficients_by_xy_order(n=n,l=l),
        "polar": str(polar),
        "polar latex": sympy.latex(polar),
        "cartesian": str(cartesian),
        "cartesian latex": sympy.latex(cartesian),
        "derivative dz dx": str( dzdx ),
        "derivative dz dx latex": sympy.latex( dzdx ),
        "derivative dz dy": str( dzdy ),
        "derivative dz dy latex": sympy.latex( dzdy ),
        }
    return dic


if __name__ == "__main__":
    assert standard_indices_to_fringe( 0, 0 ) ==  1
    assert standard_indices_to_fringe( 1, 1 ) ==  2
    assert standard_indices_to_fringe( 1,-1 ) ==  3
    assert standard_indices_to_fringe( 2, 0 ) ==  4
    assert standard_indices_to_fringe( 2, 2 ) ==  5
    assert standard_indices_to_fringe( 2,-2 ) ==  6
    assert standard_indices_to_fringe( 3, 1 ) ==  7
    assert standard_indices_to_fringe( 3,-1 ) ==  8
    assert standard_indices_to_fringe( 4, 0 ) ==  9
    assert standard_indices_to_fringe( 3, 3 ) == 10
    assert standard_indices_to_fringe( 3,-3 ) == 11
    assert standard_indices_to_fringe( 4, 2 ) == 12
    assert standard_indices_to_fringe( 4,-2 ) == 13
    assert standard_indices_to_fringe( 5, 1 ) == 14
    assert standard_indices_to_fringe( 5,-1 ) == 15
    assert standard_indices_to_fringe( 6, 0 ) == 16
    assert standard_indices_to_fringe( 4, 4 ) == 17
    assert standard_indices_to_fringe( 4,-4 ) == 18
    assert standard_indices_to_fringe( 5, 3 ) == 19
    assert standard_indices_to_fringe( 5,-3 ) == 20
    
    # test fringe_to_standard_index to be the inverse of the already tested standard_indices_to_fringe
    for j in np.arange(20)+1:
        assert standard_indices_to_fringe( *fringe_to_standard_index( j )) == j
    
    assert_list_equal( zernike_radial_polynomial_coefficients( 0,0 ), [1] )
    assert_list_equal( zernike_radial_polynomial_coefficients( 1,1 ), [0,1] )
    assert_list_equal( zernike_radial_polynomial_coefficients( 2,0 ), [-1,0,2] )
    assert_list_equal( zernike_radial_polynomial_coefficients( 2,2 ), [0,0,1] )
    assert_list_equal( zernike_radial_polynomial_coefficients( 3,1 ), [0,-2,0,3] )
    assert_list_equal( zernike_radial_polynomial_coefficients( 3,3 ), [0,0,0,1] )
    assert_list_equal( zernike_radial_polynomial_coefficients( 4,0 ), [1,0,-6,0,6] )
    assert_list_equal( zernike_radial_polynomial_coefficients( 4,2 ), [0,0,-3,0,4] )
    assert_list_equal( zernike_radial_polynomial_coefficients( 4,4 ), [0,0,0,0,1] )
    assert_list_equal( zernike_radial_polynomial_coefficients( 5,1 ), [0,3,0,-12,0,10] )
    assert_list_equal( zernike_radial_polynomial_coefficients( 5,3 ), [0,0,0,-4,0,5] )
    assert_list_equal( zernike_radial_polynomial_coefficients( 5,5 ), [0,0,0,0,0,1] )
    assert_list_equal( zernike_radial_polynomial_coefficients( 6,0 ), [-1,0,12,0,-30,0,20] )


    # test sin_omega_phi_to_sin_phi, a helper to break down sin(n*phi)
    for omega in np.arange(6)+1:
        term3, term4 = sin_omega_phi_to_sin_phi(omega=omega)
        for phi_val in [0,1,sympy.pi/2, 2.235645, 42]:
            assert_almost_equal( term3.subs(sympy.Symbol("phi"), phi_val).evalf(),
                                 term4.subs(sympy.Symbol("phi"), phi_val).evalf() )

    # test cos_omega_phi_to_cos_phi, a helper to break down cos(n*phi)
    for omega in np.arange(6)+1:
        term1, term2 = cos_omega_phi_to_cos_phi(omega=omega)
        for phi_val in [0,1,sympy.pi/2, 2.235645, 42]:
            assert_almost_equal( term1.subs(sympy.Symbol("phi"), phi_val).evalf(),
                                 term2.subs(sympy.Symbol("phi"), phi_val).evalf() )


    assert aberration_name(*fringe_to_standard_index(  1 )) == "piston"
    assert aberration_name(*fringe_to_standard_index(  2 )) == "tilt"
    assert aberration_name(*fringe_to_standard_index(  3 )) == "tilt"
    assert aberration_name(*fringe_to_standard_index(  4 )) == "defocus"
    assert aberration_name(*fringe_to_standard_index(  5 )) == "astigmatism 3rd order"
    assert aberration_name(*fringe_to_standard_index(  6 )) == "astigmatism 3rd order"
    assert aberration_name(*fringe_to_standard_index(  7 )) == "coma 3rd order"
    assert aberration_name(*fringe_to_standard_index(  8 )) == "coma 3rd order"
    assert aberration_name(*fringe_to_standard_index(  9 )) == "spherical aberration 3rd order"
    assert aberration_name(*fringe_to_standard_index( 10 )) == "3-foil 5th order"
    assert aberration_name(*fringe_to_standard_index( 11 )) == "3-foil 5th order"
    assert aberration_name(*fringe_to_standard_index( 12 )) == "astigmatism 5th order"
    assert aberration_name(*fringe_to_standard_index( 13 )) == "astigmatism 5th order"
    assert aberration_name(*fringe_to_standard_index( 14 )) == "coma 5th order"
    assert aberration_name(*fringe_to_standard_index( 15 )) == "coma 5th order"
    assert aberration_name(*fringe_to_standard_index( 16 )) == "spherical aberration 5th order"
    assert aberration_name(*fringe_to_standard_index( 17 )) == "4-foil 7th order"

    # test cartesian_coefficients_by_xy_order against the already tested zernike_polynomial_in_cartesian_representation
    for j in np.arange(36)+1:
        po1   = zernike_polynomial_in_cartesian_representation( *fringe_to_standard_index( j ) )
        carco = cartesian_coefficients_by_xy_order( *fringe_to_standard_index( j ) )
        po2   = sum( list( term["c"] * sympy.Symbol("x")**term["x**"] * sympy.Symbol("y")**term["y**"] 
                           for term in carco ) )
        assert sympy.simplify( po2 - po1 ) == 0
        

    




    stories = list( full_story( *fringe_to_standard_index( j ) ) for j in np.arange(256)+1 )

    with open("zernike.json", "w") as f:
        json.dump(stories, f, indent=4)

    tex = "\\documentclass[10pt]{article}\n"
    tex += "\\usepackage{amsmath}\n"
    tex += "\\begin{document}\n"
    tex += "\\section{The First " + str(len(stories)) + " Zernike Polynomials}\n"
    tex += "This file was automatically generated by the script " + sys.argv[0].split("/")[-1] + ".\n"
    for story in stories:
        j = str(story["fringe index"])
        n = str(story["standard indices"][0])
        l = str(story["standard indices"][1])
        tex += "  \\subsection{"  + "Z" + j + " " + story["name"] + "}\n"
        tex += "    \\begin{subequations}\n"
        tex += "    \\begin{eqnarray}\n"
        tex += "        n_{standard} &=&" + n + "\\\\\n"
        tex += "        \\ell_{standard} &=&" + l + "\\\\\n"
        tex += "        j_{fringe} &=&" + j + "\\\\\n"
        tex += "        Z_{" + n + "}^{" + l +"} = Z_{" + j +"} &=& " + story["polar latex"] + "\\\\\n"
        tex += "        Z_{" + j +"} &=& " + story["cartesian latex"] + "\n"
        tex += "        \\frac{\\partial Z}{\\partial x} &=& " + story["derivative dz dx latex"] + "\n"
        tex += "        \\frac{\\partial Z}{\\partial y} &=& " + story["derivative dz dy latex"] + "\n"
        tex += "    \\end{eqnarray}\n"
        tex += "    \\end{subequations}\n"
    tex += "\\end{document}\n"
    
    with open("zernike.tex", "w") as f:
        f.write(tex)

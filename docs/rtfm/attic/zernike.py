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
import numpy as np
from math import factorial


def assert_list_equal( l1:list, l2:list):
    assert len(l1) == len(l2)
    for i in range(len(l1)):
        assert l1[i] == l2[i], f"Error at element {i}: {l1[i]} != {l2[i]}"
    




def standard_indices_to_fringe( n: int, l:int ) -> int:
    j = (1 + ((n+abs(l))//2) )**2 - 2*abs(l) + (1 -np.sign(l)) // 2
    return j


def fringe_to_standard_index( j_fringe: int ) -> (int,int):
    nex     = int( np.ceil( np.sqrt( j_fringe ) ) )
    next_sq = nex**2 # next larger square number

    absl_plus_n = 2*nex - 2
 
    absl = int( np.ceil( ( next_sq - j_fringe ) / 2 ) )
    n = absl_plus_n - absl

    signl = (-1 if (next_sq - j_fringe) % 2  else 1)
    l = absl * signl
    
    return n, l


def zernike_radial_polynomial( n:int, l:int) -> list[float]:
    absl = abs(l)
    p = np.zeros( n+1, dtype=int )
    for i in range(abs(l),n+1,2):
        k = (n-i)//2;
        p[i] = (-1)**k * factorial(n-k) / ( factorial(k) * factorial((n+absl)//2 - k) * factorial((n-absl)//2-k) )
    return p
    

def zernike_polynomial_as_polar_str( n:int, l:int):
    p = zernike_radial_polynomial( n=n, l=l )
    poly_strings = list( f"{p[i]} r^{i}" for i in range(len(p))[::-1] if p[i] )
    s = " + ".join(poly_strings) + " "
    s = s.replace("+ -","- ").replace("r^0","").replace("r^1 ","r ").strip()
    if l:
        s = "( " + s + " )" 
        s += (" cos " if (l > 0) else " sin ") + (str(abs(l)) if abs(l)>1 else "") + "φ"
    
    return s



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
    
    for j in np.arange(20)+1:
        assert standard_indices_to_fringe( *fringe_to_standard_index( j )) == j
    
    assert_list_equal( zernike_radial_polynomial( 0,0 ), [1] )
    assert_list_equal( zernike_radial_polynomial( 1,1 ), [0,1] )
    assert_list_equal( zernike_radial_polynomial( 2,0 ), [-1,0,2] )
    assert_list_equal( zernike_radial_polynomial( 2,2 ), [0,0,1] )
    assert_list_equal( zernike_radial_polynomial( 3,1 ), [0,-2,0,3] )
    assert_list_equal( zernike_radial_polynomial( 3,3 ), [0,0,0,1] )
    assert_list_equal( zernike_radial_polynomial( 4,0 ), [1,0,-6,0,6] )
    assert_list_equal( zernike_radial_polynomial( 4,2 ), [0,0,-3,0,4] )
    assert_list_equal( zernike_radial_polynomial( 4,4 ), [0,0,0,0,1] )
    assert_list_equal( zernike_radial_polynomial( 5,1 ), [0,3,0,-12,0,10] )
    assert_list_equal( zernike_radial_polynomial( 5,3 ), [0,0,0,-4,0,5] )
    assert_list_equal( zernike_radial_polynomial( 5,5 ), [0,0,0,0,0,1] )
    assert_list_equal( zernike_radial_polynomial( 6,0 ), [-1,0,12,0,-30,0,20] )





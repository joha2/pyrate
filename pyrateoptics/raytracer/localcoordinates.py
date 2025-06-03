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

import math
import random

import numpy as np

from .helpers_math import rodrigues

from ..core.base import ClassWithOptimizableVariables
from ..core.optimizable_variable import FloatOptimizableVariable, FixedState


class LocalCoordinates(ClassWithOptimizableVariables):
    """
    Class for defining local coordinate systems.
    """
    @classmethod
    def p(cls, name="", **kwargs):
        # TODO: Reference to global to be rewritten into reference to root
        '''
        Defines a local coordinate system, on which translated or tilted optical surfaces may refer.

        @param: name -- name of the coordinate system for identification (str)
                        if the value is "", a uuid will be generated as name.
        @param: kwargs -- keyword args:
                        decx, decy, decz: decenter of the surface in x, y and z, respectively (float)
                                          Default values are zeros.
                                          decz denotes the position of the current surface.
                                          decz is equivalent to the Zemax thickness value CTVA or THIC of the previous surface.
                        tiltx, tilty, tiltz: tilt around x, y, or z axis in radians (float).
                                          Default values are zeros.
                        tiltThenDecenter: order of tilt and decenter operation (bool or int).
                                          Default value is zero.
                                          0 or False means: the decenter operations are performed first, then tiltx, then tilty, then tiltz.
                                          1 or True means: tiltz first, then tilty, then tiltx, then decenter.
        '''

        (decz, decx, decy, tiltx, tilty, tiltz) = \
            (kwargs.get(key, 0.0)
             for key in ["decz", "decx", "decy", "tiltx",
                         "tilty", "tiltz"])

        tiltThenDecenter = kwargs.get("tiltThenDecenter", 0)

        decxv = FloatOptimizableVariable(FixedState(decx), name="decx")
        decyv = FloatOptimizableVariable(FixedState(decy), name="decy")
        deczv = FloatOptimizableVariable(FixedState(decz), name="decz")
        tiltxv = FloatOptimizableVariable(FixedState(tiltx), name="tiltx")
        tiltyv = FloatOptimizableVariable(FixedState(tilty), name="tilty")
        tiltzv = FloatOptimizableVariable(FixedState(tiltz), name="tiltz")
        parent = None  # None means reference to root coordinate system
        children = []  # children

        globalcoordinates = np.array([0, 0, 0])
        localdecenter = np.array([0, 0, 0])
        localrotation = np.eye(3)
        localbasis = np.eye(3)

        annotations_dict = {}
        structure_dict = {}

        annotations_dict["tiltThenDecenter"] = tiltThenDecenter
        annotations_dict["globalcoordinates"] = globalcoordinates.tolist()
        annotations_dict["localdecenter"] = localdecenter.tolist()
        annotations_dict["localrotation"] = localrotation.tolist()
        annotations_dict["localbasis"] = localbasis.tolist()

        structure_dict["decx"] = decxv
        structure_dict["decy"] = decyv
        structure_dict["decz"] = deczv
        structure_dict["tiltx"] = tiltxv
        structure_dict["tilty"] = tiltyv
        structure_dict["tiltz"] = tiltzv
        structure_dict["parent"] = parent
        structure_dict["_LocalCoordinates__children"] = children

        lc = cls(annotations_dict, structure_dict, name)
        lc.update()  # initial update
        return lc

    def updateAnnotations(self):
        self.annotations["globalcoordinates"] = self.globalcoordinates.tolist()
        self.annotations["localdecenter"] = self.localdecenter.tolist()
        self.annotations["localrotation"] = self.localrotation.tolist()
        self.annotations["localbasis"] = self.localbasis.tolist()

    def initialize_from_annotations(self):
        """
        Further initialization stages from annotations which need to be
        done to get a valid object.
        """

        self.globalcoordinates = np.array(self.annotations["globalcoordinates"])
        self.localdecenter = np.array(self.annotations["localdecenter"])
        self.localrotation = np.array(self.annotations["localrotation"])
        self.localbasis = np.array(self.annotations["localbasis"])


    def setKind(self):
        self.kind = "localcoordinates"

    def getChildren(self):
        return self.__children

    children = property(getChildren)

    def addChild(self, childlc):
        """
        Add a child coordinate system.
        That is, the child coordinate system tilt and decenter
        are defined relative to the current system, "self".

        @param: childlc -- the local coordinate system (object)

        @return: childlc -- return the input argument (object)
        """
        childlc.parent = self
        childlc.update()
        self.__children.append(childlc)
        return childlc

    def addChildToReference(self, refname, childlc):
        """
        Adds a child to the coordinate system specified.

        @param: refname -- name of the desired parent of childlc (str)
                           if name does not occur in self or its (grand)-children, nothing is done.
        @param: childlc -- the coordinate system looking for a new parent (object)

        @return: childlc -- return the input argument (object)

        TODO: refnames occuring twice may lead to undefined behavior.
        """
        if self.name == refname:
            self.debug("reference name found: adding child")
            self.addChild(childlc)
            self.debug("child added")
            self.debug("list of children: " + str([c.name for c in self.children]))
        else:
            self.debug("reference name not found, checking children")
            for x in self.__children:
                x.addChildToReference(refname, childlc)
        return childlc

    def calculateMatrixFromTilt(self, tiltx, tilty, tiltz, tiltThenDecenter=0):
        if tiltThenDecenter == 0:
            res = np.dot(rodrigues(tiltz, [0, 0, 1]),
                         np.dot(rodrigues(tilty, [0, 1, 0]),
                                rodrigues(tiltx, [1, 0, 0])))
        else:
            res = np.dot(rodrigues(tiltx, [1, 0, 0]),
                         np.dot(rodrigues(tilty, [0, 1, 0]),
                                rodrigues(tiltz, [0, 0, 1])))
        return res

    def FactorMatrixXYZ(self, mat):
        '''
        R = Rx(thetax) Ry(thetay) Rz(thetaz).
        According to www.geometrictools.com/Documentation/EulerAngles.pdf
        section 2.1. October 2016.
        '''
        thetax = thetay = thetaz = 0

        if mat[0, 2] < 1:
            if mat[0, 2] > -1:
                thetay = math.asin(mat[0, 2])
                thetax = math.atan2(-mat[1, 2], mat[2, 2])
                thetaz = math.atan2(-mat[0, 1], mat[0, 0])
            else:
                thetay = -math.pi/2
                thetax = -math.atan2(mat[1, 0], mat[1, 1])
                thetaz = 0.
        else:
            thetay = math.pi/2
            thetax = math.atan2(mat[1, 0], mat[1, 1])
            thetaz = 0.

        return (thetax, thetay, thetaz)

    def FactorMatrixZYX(self, mat):
        '''
        R = Rz(thetaz) Ry(thetay) Rx(thetax).
        According to www.geometrictools.com/Documentation/EulerAngles.pdf
        section 2.6. October 2016.
        '''
        thetax = thetay = thetaz = 0

        if mat[2, 0] < 1:
            if mat[2, 0] > -1:
                thetay = math.asin(-mat[2, 0])
                thetaz = math.atan2(mat[1, 0], mat[0, 0])
                thetax = math.atan2(mat[2, 1], mat[2, 2])
            else:
                thetay = math.pi/2
                thetaz = -math.atan2(-mat[1, 2], mat[1, 1])
                thetax = 0.
        else:
            thetay = -math.pi/2
            thetaz = math.atan2(-mat[1, 2], mat[1, 1])
            thetax = 0.

        return (thetax, thetay, thetaz)

    def calculateTiltFromMatrix(self, mat, tiltThenDecenter=0):
        res = (0., 0., 0.)
        if tiltThenDecenter == 0:
            res = self.FactorMatrixZYX(mat)
        else:
            res = self.FactorMatrixXYZ(mat)
        return res

    def calculate(self):

        """
        tiltThenDecenter=0: decx, decy, decz, tiltx, tilty, tiltz
        tiltThenDecenter=1: tiltx, tilty, tilty, decx, decy, decz

        notice negative signs for angles to make sure that for tiltx>0 the
        optical points in positive y-direction although the x-axis of the
        local coordinate system points INTO the screen
        This leads also to a clocking in the mathematical negative direction
        """
        tiltx = self.tiltx.evaluate()
        tilty = self.tilty.evaluate()
        tiltz = self.tiltz.evaluate()
        decx = self.decx.evaluate()
        decy = self.decy.evaluate()
        decz = self.decz.evaluate()

        self.localdecenter = np.array([decx, decy, decz])
        self.localrotation = self.calculateMatrixFromTilt(tiltx,
                                                          tilty,
                                                          tiltz,
                                                          self.annotations["tiltThenDecenter"])
        self.debug("local decenter: " + str(self.localdecenter))
        self.debug("local rotation: " + str(self.localrotation))

    def update(self):
        '''
        runs through all references specified and sums up
        coordinates and local rotations to get appropriate
        global coordinate
        '''
        self.debug("calculating localrotation and local decenter")
        self.calculate()
        self.debug("calculating parent coordinates")

        parentcoordinates = np.array([0, 0, 0])
        parentbasis = np.eye(3)

        if self.parent is not None:
            parentcoordinates = self.parent.globalcoordinates
            parentbasis = self.parent.localbasis

        self.localbasis = np.dot(parentbasis, self.localrotation)
        if self.annotations["tiltThenDecenter"] == 0:
            # first decenter then rotation
            self.globalcoordinates = \
                parentcoordinates + \
                np.dot(parentbasis, self.localdecenter)
            # TODO: removed .T on parentbasis to obtain correct behavior;
            # examine!
        else:
            # first rotation then decenter
            self.globalcoordinates = \
                parentcoordinates + \
                np.dot(self.localbasis, self.localdecenter)
            # TODO: removed .T on localbasis to obtain correct behavior;
            # examine!

        self.updateAnnotations()
        self.debug("updating children")

        for ch in self.__children:
            self.debug("updating " + str(ch.name))
            ch.update()

        self.debug("informing observers")

        # inform observers about update
        self.inform_observers()

    def aimAt(self, anotherlc, update=False):
        (tiltx, tilty, tiltz) = self.calculateAim(anotherlc)

        self.tiltx.setvalue(tiltx)
        self.tilty.setvalue(tilty)
        self.tiltz.setvalue(tiltz)

        if update:
            self.update()

    def calculateAim(self, anotherlc):

        rotationtransform = np.zeros((3, 3))
        direction = self.returnGlobalToLocalPoints(anotherlc.globalcoordinates)
        dist = np.linalg.norm(direction)
        localzaxis = direction/dist

        # zaxis = normal(At - Eye)
        # xaxis = normal(cross(Up, zaxis))
        # yaxis = cross(zaxis, xaxis)

        up = np.array([0, 1, 0])  # y-axis

        localxaxis = np.cross(up, localzaxis)
        localxaxis = localxaxis/np.linalg.norm(localxaxis)
        localyaxis = np.cross(localzaxis, localxaxis)
        localyaxis = localyaxis/np.linalg.norm(localyaxis)

        rotationtransform[:, 0] = localxaxis
        rotationtransform[:, 1] = localyaxis
        rotationtransform[:, 2] = localzaxis

        transformedlocalrotation = np.dot(rotationtransform,
                                          self.localrotation)

        (tiltx, tilty, tiltz) = self.calculateTiltFromMatrix(
            transformedlocalrotation, self.tiltThenDecenter)

        # seems to be only correct for
        # -pi/2 < tilty < pi/2
        # 0 < tiltx < pi
        # 0 < tiltz < pi

        return (tiltx, tilty, tiltz)

    def returnActualToOtherPoints(self, localpts, lcother):
        # TODO: constraint: lcother and self share same root,
        # check: lcother=self
        globalpts = self.returnLocalToGlobalPoints(localpts)
        return lcother.returnGlobalToLocalPoints(globalpts)

    def returnOtherToActualPoints(self, otherpts, lcother):
        # TODO: constraint: lcother and self share same root
        globalpts = lcother.returnLocalToGlobalPoints(otherpts)
        return self.returnGlobalToLocalPoints(globalpts)

    def returnActualToOtherDirections(self, localdirs, lcother):
        # TODO: constraint: lcother and self share same root
        globaldirs = self.returnLocalToGlobalDirections(localdirs)
        return lcother.returnGlobalToLocalDirections(globaldirs)

    def returnOtherToActualDirections(self, otherdirs, lcother):
        globaldirs = lcother.returnLocalToGlobalDirections(otherdirs)
        return self.returnGlobalToLocalDirections(globaldirs)

    def returnActualToOtherTensors(self, localtensors, lcother):
        # TODO: constraint: lcother and self share same root
        globaltensors = self.returnLocalToGlobalTensors(localtensors)
        return lcother.returnGlobalToLocalDirections(globaltensors)

    def returnOtherToActualTensors(self, othertensors, lcother):
        globaltensors = lcother.returnLocalToGlobalTensors(othertensors)
        return self.returnGlobalToLocalTensors(globaltensors)

    def returnLocalToGlobalPoints(self, localpts):
        """
        @param: localpts (3xN numpy array)
        @return: globalpts (3xN numpy array)
        """
        # construction to use broadcasting
        return (np.dot(self.localbasis, localpts).T + self.globalcoordinates).T

    def returnLocalToGlobalDirections(self, localdirs):
        """
        @param: localdirs (3xN numpy array)
        @return: globaldirs (3xN numpy array)
        """
        return np.dot(self.localbasis, localdirs)

    def returnGlobalToLocalPoints(self, globalpts):
        """
        @param: globalpts (3xN numpy array)
        @return: localpts (3xN numpy array)
        """
        # construction to use broadcasting
        return np.dot(self.localbasis.T, (globalpts.T - self.globalcoordinates).T)

    def returnGlobalToLocalDirections(self, globaldirs):
        """
        @param: globaldirs (3xN numpy array)
        @return: localdirs (3xN numpy array)
        """

        localpts = np.dot(self.localbasis.T, globaldirs)
        return localpts

    def returnGlobalToLocalTensors(self, globaltensor):
        """
        @param: globaldirs (3x3xN numpy array)
        @return: localdirs (3x3xN numpy array)
        """

        (num_dims_r, num_dims_c, num_pts) = np.shape(globaltensor)
        localbasisT = np.repeat(self.localbasis.T[:, :, np.newaxis], num_pts, axis=2)
        localtensor = np.einsum('lj...,ji...,ki...', localbasisT, globaltensor, localbasisT).T
        return localtensor

    def returnLocalToGlobalTensors(self, localtensor):
        """
        @param: globaldirs (3x3xN numpy array)
        @return: localdirs (3x3xN numpy array)
        """

        (num_dims_r, num_dims_c, num_pts) = np.shape(localtensor)
        localbasis = np.repeat(self.localbasis[:, :, np.newaxis], num_pts, axis=2)
        globaltensor = np.einsum('lj...,ji...,ki...', localbasis, localtensor, localbasis).T
        return globaltensor


    def returnConnectedNames(self):
        lst = [self.name]
        for ch in self.__children:
            lst = lst + ch.returnConnectedNames()
        return lst

    def returnConnectedChildren(self):
        lst = [self]
        for ch in self.__children:
            lst = lst + ch.returnConnectedChildren()
        return lst


    def pprint(self, n=0):
        """
        returns a string visualizing the tree structure of self and its children.

        @param n: indentation level (int)

        @return s: structure of self.name and its children (str)
        """
        s = n*"    " + self.name + " (" + str(self.globalcoordinates) + ")\n"
        for x in self.__children:
            s += x.pprint(n+1)

        return s

    def __str__(self):
        s = 'name \'%s\'\ntiltThenDecenter %d\nglobal coordinates: %s\nld: %s\nlr:\n%s\nlb:\n%s\nchildren %s'\
            % (self.name, self.annotations["tiltThenDecenter"],
               self.globalcoordinates,
               self.localdecenter,
               self.localrotation,
               self.localbasis,
               [i.name for i in self.__children])
        return s


if __name__ == "__main__":

    def main():
        printouttestcase1 = False
        printouttestcase2 = False
        printouttestcase3 = False
        printouttestcase4 = True

        '''testcase2: convert rotation matrix to tilt'''
        surfrt0 = LocalCoordinates.p("rt0")
        for loop in range(1000):
            (tiltx, tiltz) = (random.random()*math.pi for i in range(2))
            tilty = random.random()*math.pi - math.pi/2
            tiltThenDecenter = random.randint(0, 1)
            surfrt1 = surfrt0.addChild(LocalCoordinates.p("rt1", decz=20, tiltx=tiltx, tilty=tilty, tiltz=tiltz, tiltThenDecenter=tiltThenDecenter))
            (tiltxc, tiltyc, tiltzc) = surfrt1.calculateTiltFromMatrix(surfrt1.localrotation, tiltThenDecenter)
            if printouttestcase2:
                print("diffs %d %f %f %f: %f %f %f" % (tiltThenDecenter, tiltx, tilty, tiltz, tiltxc - tiltx, tiltyc - tilty, tiltzc - tiltz))
        '''testcase3: aimAt function'''
        surfaa0 = LocalCoordinates("aa0")
        surfaa05 = surfaa0.addChild(LocalCoordinates.p("aa05", decz=20, tiltx=20*math.pi/180.0))
        surfaa1 = surfaa05.addChild(LocalCoordinates.p("aa1", decz=20, tiltx=20*math.pi/180.0))
        surfaa2 = surfaa1.addChild(LocalCoordinates.p("aa2", decz=20))
        surfaa3 = surfaa2.addChild(LocalCoordinates.p("aa3", decz=0))
        surfaa4 = surfaa3.addChild(LocalCoordinates.p("aa4", decz=57.587705))

    main()

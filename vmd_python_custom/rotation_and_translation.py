#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmd-python custom functions.

Created on Sun Dec 20 20:52:50 2020

@author: aretaon
"""
import math
import numpy as np
from scipy.spatial import distance

def unit_vector(vector):
    """Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def xyz_from_selection(sel):
    """Read xyz coordinated from the atomsel object and generate 3xN matrix"""
    return np.concatenate((np.array(sel.x)[:, np.newaxis],
                            np.array(sel.y)[:, np.newaxis],
                            np.array(sel.z)[:, np.newaxis]),
                            axis=1)

def lsq(sel):
    """Return first component of SVD of positions (i.e. the least squares linear fit)"""
    data = xyz_from_selection(sel)

    # Calculate the mean of the points, i.e. the 'center' of the cloud
    datamean = data.mean(axis=0)

    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(data - datamean)

    return vv[0]

def to_vmd_matrix(rotation, translation=np.zeros(3)):   
    """
    Return a combination of rotation and transaltion matrices in vmd format.
    
    vmd needs a homogeneous transformation matrix for its move command
    (see http://dept.me.umn.edu/courses/me5286/manipulator/LectureNotes/2017/ME5286CourseNotes3-2017.pdf)
    This is [[R, R, R, T],
             [R, R, R, T],
             [R, R, R, T],
             [0, 0, 0, 1]]
    With R = inverse rotation matrix and T = Translation vector

    Parameters
    ----------
    rotation : 3x3 numpy matrix
        Rotation matrix.
    translation : 3x1 numpy vector, optional
        Translation matrix. The default is np.zeros(3).

    Returns
    -------
    list
        vmd compatible list format of homogeneous matrix.
    """
    m = np.append(np.linalg.inv(rotation), translation.reshape([3,1]), axis=1)
    m = np.append(m, np.array([0,0,0,1]).reshape(1,4), axis=0)
    m = m.ravel(order='F').reshape(1,16)
    return m.tolist()[0]

def align(whole, ref, orientation_ref, target_vec):
    """
    Align the first principal component of a selection
    (specified by selection_term in vmd syntax) to a path vector (target_vec).

    Parameters
    ----------
    whole : vmd-python atomsel
        Selection of the whole molecule to move.
    ref : vmd-python atomsel
        Selection used to reference to the target vector.
    orientation_ref : vmd-python atomsel
        Selection that determines the orientation of the principal component
    align_sel : vmd-python atomsel
        Selection to determine the alignation of the molecule along the vector
    target_vec : 3-element tuple
        Vector to align to (does not need to be normalised).

    Returns
    -------
    None.

    """
    def get_alignment_matrix(a, b):
        """
        Calculate a transformation matrix to align a selection vector a and a target vector b of arbitrary length.
        For reference see https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        """
        def cpm(v):

            """
            Cross-product matrix in 3D
            """
            return np.matrix([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])

        # Calculate Rodrigues' matrix
        v = np.cross(unit_vector(b), unit_vector(a))
        c = np.dot(unit_vector(b), unit_vector(a))
        R = np.identity(3) + cpm(v) + (1/(1+c))*cpm(v)**2

        # Compute acos of the cos and return answer in degrees
        angle = 180 * math.acos(c) / 3.14159265
        print('Current angle between the vectors is: ' + str(angle))
        print('Current R is:\n', R, '\n')
        return R

    # Move molecule to the origin
    com = whole.center(whole.mass)
    whole.move(to_vmd_matrix(np.identity(3), np.array([i*-1 for i in com])))

    # rotate the molecule according to the vector
    com = np.array(whole.center(whole.mass)) # calculate center of mass
    vec = lsq(ref)
    com_orient = np.array(orientation_ref.center(orientation_ref.mass))

    # use vec or -vec depending on which endpoint (com + vec) is closer
    # to the reference selection com (i.e. the reference selection determines
    # the orientation of the head of the vector arrow)
    if np.sqrt(np.sum(com - com_orient + vec)**2) >\
        np.sqrt(np.sum(com - com_orient - vec)**2):
        vec = -vec

    R = get_alignment_matrix(vec, target_vec)
    print('Does this match the alignation axis?: ', np.dot(vec, R), '\n')
    whole.move(to_vmd_matrix(R))

    # translate so that molecule fits its original COM
    com_new = np.array(whole.center(whole.mass))
    trans = com - com_new
    whole.move(to_vmd_matrix(np.identity(3), trans))
    vec = lsq(ref) # to test if the alignation worked
    R = get_alignment_matrix(vec, target_vec)

def position(ref_mobile, whole_mobile, fixed, movement_vec, target_dist, dist_thresh):
    """
    Translate a molecule along a defined vector until a target distance
    between a subset of atoms of both molecules is reached.

    Parameters
    ----------
    ref_mobile : vmd-python atomsel
        Selection in distance calculation.
    whole_mobile : vmd-python atomsel
        Selection of the whole molecule to move.
    fixed : vmd-python atomsel
        Fixed part to position against.
    movement_vec : 3-element tuple
        Vector along which the mobile element is moved.
    target_dist : int or flaot
        Distance to reach during annealing.
    dist_thresh : int or float
        Tolerance for the final distance.

    Returns
    -------
    None.
    """

    def calc_dist(ref_mobile, fixed):
        A = xyz_from_selection(ref_mobile)
        B = xyz_from_selection(fixed)

        dist = distance.cdist(A,B) # pick the appropriate distance metric

        return dist.min()

    # initially position the ref_mobile molecule far from the fixed
    trans = 250 * (movement_vec / np.linalg.norm(movement_vec))
    whole_mobile.move(to_vmd_matrix(np.identity(3), trans))

    # calculate the shortest distance between the two selections
    d = calc_dist(ref_mobile, fixed)
    print('Current distance is: ', d)

    # only leave the loop once the positioning converges
    while (d-target_dist) > dist_thresh:

        trans = -(d-target_dist) * (movement_vec / np.linalg.norm(movement_vec))
        whole_mobile.move(to_vmd_matrix(np.identity(3), trans))

        d = calc_dist(ref_mobile, fixed)
        print('Current distance is: ', d)


def rotate(sel, angle, vec):
    """
    Rotate a molecule selection by an angle around a defined vector

    Parameters
    ----------
    sel : vmd-python atomsel
        Selection that is manipulated.
    angle : int or float
        Rotation angle in degress.
    vec : 3-element tuple.
        Vector to rotate around.

    Returns
    -------
    None.

    """

    def get_rotation_matrix(v, angle):
        """
        Calculate a transformation matrix to align a selection vector a and a target vector b of arbitrary length.
        For reference see https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        """
        def cpm(v):

            """
            Cross-product matrix in 3D
            """
            return np.matrix([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])

        # Normalise vectors
        v = unit_vector(v)

        # Calculate Rodrigues' matrix
        angle = angle * np.pi / 180 # in rad
        s = np.sin(angle)
        sh= np.sin(angle/2)
        R = np.identity(3) + s*cpm(v) + 2*sh**2*cpm(v)**2
        # c = np.cos(angle)
        # R = np.identity(3) + cpm(v) + (1/(1+c))*cpm(v)**2

        print('Current R is:\n', R, '\n')
        return R

    com = np.array(sel.center(sel.mass)) # calculate center of mass

    R = get_rotation_matrix(vec, angle)
    sel.move(to_vmd_matrix(R))

    # translate so that molecule fits its original COM
    com_new = np.array(sel.center(sel.mass))
    trans = com - com_new
    sel.move(to_vmd_matrix(np.identity(3), trans))

def get_rotation_angle(sel, ref_sel, ref_vector):
    """
    Calculate the rotational angle of a molecule selection and a reference selection
    with respect to a reference vector (i.e. the z-axis (0,0,1))

    Parameters
    ----------
    sel : vmd-python atomsel
        Selection containing the whole molecule. Used to calculate the principal
        component.
    ref_sel : vmd-python atomsel
        Selection that is used to define the rotational orientation. Ideally
        located in a stable part of the protein periphery.
    ref_vector : 3-element tuple
        Vector used to span the plane with respect to which the angle is calculated.

    Returns
    -------
    float
        Angle in degrees.

    """
    r_point = np.array(ref_sel.center(ref_sel.mass)) # get center of mass of reference
    com_point = np.array(sel.center(sel.mass)) # get com of whole molecule
    v_vec = lsq(sel)

    n1 = np.cross((r_point - com_point), v_vec)
    n2 = np.cross(np.array(ref_vector), v_vec)

    def angle_between(v1, v2):
        """
        Returns the angle in radians between vectors 'v1' and 'v2'::
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) #in rad

    angle = angle_between(n1, n2)
    angle = 180*angle/np.pi
    print('Angle is: ', angle)

    return angle
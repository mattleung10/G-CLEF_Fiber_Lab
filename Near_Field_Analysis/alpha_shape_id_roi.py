#Matthew Leung
#June 2022
"""
This script contains functions to identify the fiber face (region of interest)
in an image of a near-field fiber.

PROCEDURE:
    1) Apply Canny edge detector to 8 bit image, and obtain a binary image
       which represents the Canny edges
    2) Take the nonzero points from the binary image, and find the alpha
       shape of these points
    3) Find the outer boundary of the alpha shape
    4) If desired, offset the alpha shape
    5) Create a binary mask of all pixels inside the alpha shape outer boundary
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import PIL.Image as Image
import warnings
from scipy.spatial import Delaunay

#######################################################################################
#######################################################################################
#ALPHA SHAPES FUNCTIONS

def alpha_shape(points, alpha, only_outer=True):
    #From https://stackoverflow.com/a/50159452
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        # if (j,i) in edges: edges.remove((j,i))
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it is not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    DT = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for simplex in DT.simplices:
        ia, ib, ic = simplex
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        #https://keisan.casio.com/exec/system/1223429573
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        
        if circum_r < alpha: #alpha test
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def delaunay_tri(points):
    #From https://stackoverflow.com/a/50159452
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        # if (j,i) in edges: edges.remove((j,i))
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            return
        edges.add((i, j))

    DT = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for simplex in DT.simplices:
        ia, ib, ic = simplex
        add_edge(edges, ia, ib)
        add_edge(edges, ib, ic)
        add_edge(edges, ic, ia)
    return edges

#######################################################################################
#######################################################################################
#FUNCTIONS TO PROCESS RESULTS OF ALPHA SHAPES
        
def find_edges_with(i, edge_set):
    #From https://stackoverflow.com/a/50714300
    i_first = [j for (x,j) in edge_set if x==i]
    i_second = [j for (j,x) in edge_set if x==i]
    return i_first,i_second

def stitch_boundaries(edges):
    #From https://stackoverflow.com/a/50714300
    edge_set = edges.copy()
    boundary_lst = []
    while len(edge_set) > 0:
        boundary = []
        edge0 = edge_set.pop()
        boundary.append(edge0)
        last_edge = edge0
        while len(edge_set) > 0:
            i,j = last_edge
            j_first, j_second = find_edges_with(j, edge_set)
            if j_first:
                edge_set.remove((j, j_first[0]))
                edge_with_j = (j, j_first[0])
                boundary.append(edge_with_j)
                last_edge = edge_with_j
            elif j_second:
                edge_set.remove((j_second[0], j))
                edge_with_j = (j, j_second[0])  # flip edge rep
                boundary.append(edge_with_j)
                last_edge = edge_with_j

            if edge0[0] == last_edge[1]:
                break

        boundary_lst.append(boundary)
    return boundary_lst

#######################################################################################
#######################################################################################
#READ IMAGES
 
def read_12bit_image_saved_as_16bit(filename, verbosity=0):
    """
    Reads a 12 bit image which was saved as a 16 bit image
    INPUTS:
        ::str:: filename
        ::int:: verbosity
    """
    img16bit = Image.open(filename)
    arr16bit = np.asarray(img16bit)
    arr12bit = np.divide(arr16bit, 2**4)
    if verbosity >= 1:
        num_unique_elems = np.size(np.unique(arr12bit))
        print("Image has {} unique elements".format(num_unique_elems))
        print("Image maximum value is {}".format(np.max(arr12bit)))
        print("Image minimum value is {}".format(np.min(arr12bit)))
    return arr12bit

def read_8bit_image(filename, verbosity=0):
    """
    Reads an 8 bit image
    INPUTS:
        ::str:: filename
        ::int:: verbosity
    """
    img8bit = Image.open(filename)
    arr8bit = np.asarray(img8bit)
    if verbosity >= 1:
        num_unique_elems = np.size(np.unique(arr8bit))
        print("Image has {} unique elements".format(num_unique_elems))
        print("Image maximum value is {}".format(np.max(arr8bit)))
        print("Image minimum value is {}".format(np.min(arr8bit)))
    return arr8bit

#######################################################################################
#######################################################################################
    
def get_canny_points(img8, CannyMaxFactor=1/4, CannyMinFactor=3/4, verbosity=0):
    """
    Apply Canny edge detector to image img8, then return a np.ndarray
    containing the coordinates of the nonzero points in the Canny binary image.
    INPUTS:
        ::np.ndarray:: img8      #8 bit image
        ::float:: CannyMaxFactor #maxVal factor in Canny edge
        ::float:: CannyMinFactor #minVal factor in Canny edge
        ::int:: verbosity
    """
    #Apply Canny Edge Detection to 8 bit image
    minVal = np.max(img8) * CannyMaxFactor
    maxVal = np.max(img8) * CannyMinFactor
    #edges is a np.ndarray, which is a binary image with the same dimensions as img
    edges = cv.Canny(img8, minVal, maxVal)
    
    if verbosity >= 1:
        print("The number of elements in Canny is {}".format(np.count_nonzero(edges)))
    
    if verbosity >= 2: #Plot Canny edges
        plt.figure()
        plt.imshow(edges)
        plt.title('Canny Edge applied to 8 bit Image')
        plt.show()
    
    ###################################################################################
    #Find and plot contours in Canny; this code is not necessary and is just
    #for visualization purposes.
    if verbosity >= 3:
        contours_ce, hierarchy = cv.findContours(edges, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    
        #Plot contours in Canny
        img8_copy = img8.copy()
        cv.drawContours(img8_copy, contours_ce, -1, (255, 255, 255), 1)
        plt.imshow(img8_copy)
        plt.colorbar()
        plt.title('All Contours from Canny Edge')
        plt.show()
    
        #Find the contour with the largest area
        contour_ce_largest = max(contours_ce, key = cv.contourArea)
        img8_copy2 = img8.copy()
        contour_ce_largest_unzipped = list(zip(*contour_ce_largest[:,0,:]))
        #cv.drawContours(img8_copy2, [contour_ce_largest], 0, (255, 255, 255), 1)
        plt.figure()
        plt.imshow(img8_copy2)
        #Plot line from first point to last point in CW direction
        x = np.array(contour_ce_largest_unzipped[0])
        y = np.array(contour_ce_largest_unzipped[1])
        plt.plot(contour_ce_largest_unzipped[0], contour_ce_largest_unzipped[1], color='red')
        #Plot line from first point to last point, complete the shape
        plt.plot([contour_ce_largest_unzipped[0][0],contour_ce_largest_unzipped[0][-1]], [contour_ce_largest_unzipped[1][1],contour_ce_largest_unzipped[1][-1]], color='red')
        plt.colorbar()
        plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', color='orange', scale=1)
        plt.title('Largest Contour from Canny Edge')
        plt.show()
        
        blank = np.zeros(shape=img8.shape)
        cv.drawContours(blank, [contour_ce_largest], -1, 255, 1)
        plt.imshow(blank)
        plt.colorbar()
        plt.title('Binary image of Largest Contour from Canny Edge')
        plt.show()
    ###################################################################################
    
    a = np.nonzero(edges) #indicies of the nonzero elements from the Canny binary image
    a = np.vstack([a[1], a[0]]).T #coordinates of nonzero elements from Canny binary image
    return a

#######################################################################################
#######################################################################################

def offset_polygon(subj, shape_offset_amount, plot=True):
    """
    Offsets a polygon
    INPUTS:
        ::list of list:: subj
        ::float:: shape_offset_amount
        ::boolean:: plot
    OUTPUT
        ::list of list::
    """
    if type(shape_offset_amount) != int and type(shape_offset_amount) != float:
        warnings.warn("TypeError: shape_offset_amount must be of type int or float")
        return False
    try:
        import pyclipper
    except ImportError:
        warnings.warn("ImportError: pyclipper not found. Will not clip alpha shape.")
        return False
    
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(subj, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(shape_offset_amount)

    subj_np = np.array(subj)
    solution_np = np.array(solution[0])
    
    if plot == True:
        plt.figure()
        plt.plot(subj_np[:,0], subj_np[:,1], color='tab:blue', label='Original Polygon')
        plt.plot([subj_np[0,0],subj_np[-1,0]], [subj_np[0,1],subj_np[-1,1]], color='tab:blue')
        plt.plot(solution_np[:,0], solution_np[:,1], color='tab:orange', label='Offset Polygon')
        plt.plot([solution_np[0,0],solution_np[-1,0]], [solution_np[0,1],solution_np[-1,1]], color='tab:orange')
        plt.gca().set_aspect('equal', adjustable='box') #https://stackoverflow.com/a/17996099
        plt.gca().invert_yaxis()
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.show()
    
    return solution_np

#######################################################################################
#######################################################################################


def hull_mask_from_alpha_shape(img, alpha=100, CannyMaxFactor=1/4, CannyMinFactor=3/4, is12bit=True, return_boundary_points=False, shape_offset_amount=None, verbosity=0):
    """
    Returns a binary mask containing the pixels within the fiber face.
    The fiber face is identified by the alpha shape which encloses it.
    INPUTS:
        ::np.array:: img                   #image
        ::float:: alpha                    #alpha value in alpha shapes; alpha=infinity means convex hull, alpha=0 means individual points
        ::float:: CannyMaxFactor           #maxVal factor for Canny
        ::float:: CannyMinFactor           #minVal factor for Canny
        ::boolean:: is12bit                #whether or not the image is 12 bit
        ::boolean:: return_boundary_points #whether or not to reutrn the boundary points
        ::float:: shape_offset_amount      #number of pixels to offset the original alpha shape by
        ::int:: verbosity
    OUTPUT:
        ::np.ndarray:: hull_mask #binary mask; same dimensions as original image
        ::two-element list of tuple:: list_of_lines_sorted_unzipped #list of points in longest alpha shape boundary
    PROCEDURE:
        1) Apply Canny edge detector to 8 bit image, and obtain a binary image
           which represents the Canny edges
        2) Take the nonzero points from the binary image, and find the alpha
           shape of these points
        3) Find the outer boundary of the alpha shape
        4) If desired, offset the alpha shape
        5) Create a binary mask of all pixels inside the alpha shape outer boundary
    """    
    
    if is12bit == True: #12 bit image  
        img8 = (img / 2**4).astype(np.uint8)
    else: #8 bit image
        img8 = img.copy()
    
    if verbosity >= 2: #Plot original image
        plt.figure()
        plt.imshow(img)
        plt.colorbar()
        plt.title('Original Image')
        plt.show()
    
    #Run Canny edge detector on image
    canny_points = get_canny_points(img8, CannyMaxFactor=1/4, CannyMinFactor=3/4, verbosity=verbosity)
    
    #Run alpha shapes
    #Alpha shapes returns a set of tuples, where each tuple is a point
    points = canny_points #we feed all of Canny's output points into alpha shape
    as_edges = alpha_shape(points, alpha=alpha, only_outer=True)
    
    if verbosity >= 2: #Plot the alpha shape
        plt.figure()
        plt.imshow(np.zeros(shape=img8.shape))
        for i, j in as_edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1])
        plt.title('Alpha shape')
        plt.show()
    
    if verbosity >= 4: #Plot the Delaunay Triangulation
        DT_edges = delaunay_tri(points)
        plt.figure()
        plt.imshow(np.zeros(shape=img8.shape))
        for i, j in DT_edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1])
        plt.title('Delaunay Triangulation')
        plt.show()
    
    #Combine the points together in order to find the boundary of the alpha shape.
    #Note that the boundary can be made up of several disjoint curves/contours.
    s_as_edges = stitch_boundaries(as_edges)
    if verbosity >= 1: print("There are {} borders".format(len(s_as_edges)))
    #Find the outer boundary of the alpha shape; we'll assume it's the contour with the most points
    lengths = [len(elem) for elem in s_as_edges] #later: better to use residual as metric
    longest_path = s_as_edges[np.argmax(lengths)]

    #list_of_lines is a list of list of two-element np.ndarrays.
    #Each list of two-element np.ndarray represents a line segment.
    #Each np.ndarray in each list is an endpoint in the line segment.    
    list_of_lines = []
    for as_edge in longest_path:
        line = [points[as_edge[0]], points[as_edge[1]]]
        list_of_lines += [line]
    
    #Sort the list_of_lines so that all the points are connected. E.g. we want
    #something like [[[1009,430],[1010,431]], [[1010,431],[1011,431]], [[1011,431],[1012,432]]]
    list_of_lines_copy = list_of_lines.copy() #copy of list_of_lines; we will pop from this in order to do the sort
    list_of_lines_sorted = [list_of_lines_copy[0]]
    list_of_lines_copy.pop(0)
    for i in range(1,len(list_of_lines),1):
        prev_line = list_of_lines_sorted[i-1]
        for j in range(0,len(list_of_lines_copy),1):
            curr_line = list_of_lines_copy[j]
            if np.array_equal(prev_line[1], curr_line[0]):
                list_of_lines_sorted += [curr_line]
                list_of_lines_copy.pop(j)
                break
    
    #contour_points is a list of two-element np.ndarrays, consisting of the
    #points which form the longest contour of the alpha shape boundary.
    #contour_points is formatted so that it could be used by cv.drawContours
    contour_points = [list_of_lines_sorted[i][0] for i in range(0,len(list_of_lines_sorted),1)]
    
    #If desired, offset the polygon
    if shape_offset_amount is not None:
        if shape_offset_amount != 0:
            plot_offset = True if verbosity >= 2 else False
            rval = offset_polygon(contour_points, shape_offset_amount, plot=plot_offset)
            if rval is not False: #valid return value from offset_polygon
                contour_points = rval

    #List of points in longest alpha shape boundary
    list_of_lines_sorted_unzipped = list(zip(*contour_points))
    
    if verbosity >= 2:
        img8_copy2 = img8.copy()
        #cv.drawContours(img8_copy2, [contour_ce_largest], 0, (255, 255, 255), 1)
        plt.figure()
        plt.imshow(img8_copy2)
        #Plot line from first point to last point in CW direction
        x = np.array(list_of_lines_sorted_unzipped[0])
        y = np.array(list_of_lines_sorted_unzipped[1])
        plt.plot(list_of_lines_sorted_unzipped[0], list_of_lines_sorted_unzipped[1], color='red')
        #Plot line from first point to last point, complete the shape
        plt.plot([list_of_lines_sorted_unzipped[0][0],list_of_lines_sorted_unzipped[0][-1]], [list_of_lines_sorted_unzipped[1][1],list_of_lines_sorted_unzipped[1][-1]], color='red')
        plt.colorbar()
        plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', color='orange', scale=1)
        plt.title('Largest Contour from Alpha Shape')
        plt.show()
        
        #Plot alpha shape on original image
        plt.figure()
        plt.imshow(img)
        #Plot line from first point to last point in CW direction
        x = np.array(list_of_lines_sorted_unzipped[0])
        y = np.array(list_of_lines_sorted_unzipped[1])
        plt.plot(list_of_lines_sorted_unzipped[0], list_of_lines_sorted_unzipped[1], color='red')
        #Plot line from first point to last point, complete the shape
        plt.plot([list_of_lines_sorted_unzipped[0][0],list_of_lines_sorted_unzipped[0][-1]], [list_of_lines_sorted_unzipped[1][1],list_of_lines_sorted_unzipped[1][-1]], color='red')
        plt.colorbar()
        plt.title('Fiber Face Boundary')
        plt.show()
    
    hull_mask = np.zeros(shape=img8.shape)
    cv.drawContours(hull_mask, [np.array(contour_points)], 0, (255, 255, 255), -1) #fill the hull mask (thickness=-1)
    hull_mask /= 255 #normalize all values to be between 0 and 1
    
    if verbosity >= 2:
        plt.figure()
        plt.imshow(hull_mask)
        plt.plot(list_of_lines_sorted_unzipped[0], list_of_lines_sorted_unzipped[1], color='red')
        #Plot line from first point to last point, complete the shape
        plt.plot([list_of_lines_sorted_unzipped[0][0],list_of_lines_sorted_unzipped[0][-1]], [list_of_lines_sorted_unzipped[1][1],list_of_lines_sorted_unzipped[1][-1]], color='red')
        plt.colorbar()
        plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', color='orange', scale=1)
        plt.title('Filled Alpha Shape')
        plt.show()
    
    #Find the points in the original image which are inside the Convex Hull
    img_points_in_hull = np.where(hull_mask==1, img, 0) #This array is the original image, except all points outside the Convex Hull are forced to 0
    if verbosity >= 2:
        plt.figure()
        plt.imshow(img_points_in_hull)
        plt.colorbar()
        plt.title('Points of Original Image inside Alpha Shape')
        plt.show()
    
    #Find the points in the original image which were not in the Convex Hull
    residual = img - img_points_in_hull
    if verbosity >= 2:
        plt.figure()
        plt.imshow(residual)
        plt.colorbar()
        plt.title('Residuals')
        plt.show()
    
    if return_boundary_points == False:
        return hull_mask
    else:
        return hull_mask, list_of_lines_sorted_unzipped

if __name__ == "__main__":
    filename = 'Images/20220602_Stack_Vary/20220602_143915.tif' #square fiber
    filename = 'Images/20220623_Freq_Vary/20220623_153338.tif' #rectangular fiber
    img = read_12bit_image_saved_as_16bit(filename, verbosity=0)
    hull_mask = hull_mask_from_alpha_shape(img, alpha=100, CannyMaxFactor=1/4, CannyMinFactor=3/4, is12bit=True, shape_offset_amount=-3, verbosity=2)
    #cv.imwrite('a.bmp', 255 * hull_mask)
    
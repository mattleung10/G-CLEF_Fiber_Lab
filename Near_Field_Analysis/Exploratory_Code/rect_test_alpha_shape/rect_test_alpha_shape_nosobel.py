#Matthew Leung
#April 2022
#NO SOBEL OPERATOR APPLIED

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import PIL.Image as Image
import warnings

###############################################################################
###############################################################################

from scipy.spatial import Delaunay
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

###############################################################################
    
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

###############################################################################
###############################################################################

def read_12bit_image_saved_as_16bit(filename, verbosity=0):
    """
    Reads a 12 bit image which was saved as a 16 bit image
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

def apply_sobel_op(img):
    """
    Apply Sobel operator to image
    """
    #Resources:
    #https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    #https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
    #https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
    
    #Set depth of output image
    #https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#filter_depths
    #ddepth = cv.CV_16S
    ddepth = cv.CV_64F
    ksize = 3 #kernel size
    
    grad_x = cv.Sobel(img, ddepth, dx=1, dy=0, ksize=ksize) #x gradient
    grad_y = cv.Sobel(img, ddepth, dx=0, dy=1, ksize=ksize) #y gradient
    grad = (np.abs(grad_x) + np.abs(grad_y))/2 #approximation of total gradient
    return grad

###############################################################################
###############################################################################

if __name__ == "__main__":
    filename = 'Images/20220413_141608.tif' #octagonal fiber
    #filename = 'Images/20220323_1155 Near Field Rect Laser.tif'
    #filename = 'Images/20220413_141650.tif'
    #filename = 'Images/20220413_143326.tif'
    filename = 'Images/20220418_153630.tif' #rectangular fiber, mode scrambler off
    #filename = 'Images/20220418_153752.tif' #rectangular fiber, mode scrambler on
    filename = 'Images/20220418_163807.tif' #square fiber, mode scrambler off
    filename = 'Images/20220418_163913.tif' #square fiber, mode scrambler on
    
    plot_delaunay = True
    
    #savedir = os.path.join(os.getcwd(), 'alpha_shape_demo')
    #if os.path.isdir(savedir) == False: os.mkdir(savedir)
    #filebasename = os.path.basename(filename).split('.')[0] #basename of file, without extension
    
    img = read_12bit_image_saved_as_16bit(filename)
    #img = img[350:700,800:1100]
    
    img8 = (img / 2**4).astype(np.uint8)
    
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.title('Original Image')
    plt.show()
    
    #Apply Canny Edge Detection to 8 bit version of original image
    minVal = np.max(img8) / 4
    maxVal = np.max(img8) * 3/4
    edges = cv.Canny(img8, minVal, maxVal)
    #edges is a np.ndarray, which is a binary image with the same dimensions as img
    
    plt.imshow(edges)
    #plt.colorbar()
    plt.title('Canny Edge applied to 8 bit Image')
    plt.show()
    
    ###################################################################################
    #Find contours in Canny
    contours_ce, hierarchy = cv.findContours(edges, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
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
    
    print("The number of elements in Canny is {}".format(np.count_nonzero(edges)))
    
    a = np.nonzero(edges) #indicies of the nonzero elements from the Canny binary image
    a = np.vstack([a[1], a[0]]).T #coordinates of nonzero elements from Canny binary image
    print(a)
    print(a.shape)
    
    points = contour_ce_largest[:,0,:] #OLD; we feed the largest contour into alpha shape
    points = a #NEW; we feed all of Canny's output points into alpha shape
    #print(points)
    #print(points.shape)
    
    #Run alpha shapes
    #Alpha shapes returns a set of tuples, where each tuple is a point
    as_edges = alpha_shape(points, alpha=100, only_outer=True)
    
    # Plotting the output
    plt.figure()
    #plt.plot(points[:, 0], points[:, 1], '.')
    plt.imshow(np.zeros(shape=img8.shape))
    for i, j in as_edges:
        plt.plot(points[[i, j], 0], points[[i, j], 1])
    plt.title('Alpha shape')
    plt.show()
    
    if plot_delaunay:
        DT_edges = delaunay_tri(points)
        plt.figure()
        #plt.plot(points[:, 0], points[:, 1], '.')
        plt.imshow(np.zeros(shape=img8.shape))
        for i, j in DT_edges:
            plt.plot(points[[i, j], 0], points[[i, j], 1])
        plt.title('Delaunay Triangulation')
        plt.show()
    
    #Combine the points together in order to find the boundary of the alpha shape.
    #Note that the boundary can be made up of several disjoint curves/contours.
    s_as_edges = stitch_boundaries(as_edges)
    print("There are {} borders".format(len(s_as_edges)))
    #Find the longest contour of the boundary
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
    
    #List of points in longest alpha shape boundary
    list_of_lines_sorted_unzipped = list(zip(*contour_points))
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
    plt.imshow(img_points_in_hull)
    plt.colorbar()
    plt.title('Points of Original Image inside Alpha Shape')
    plt.show()
    
    #Find the points in the original image which were not in the Convex Hull
    residual = img - img_points_in_hull
    plt.imshow(residual)
    plt.colorbar()
    plt.title('Residuals')
    plt.show()
    
    #Obtain the pixel values which are inside the Convex Hull
    whered = np.where(hull_mask==1, img, -1000) #-1000 can instead be any arbitrary number which is NOT a valid pixel value
    list_of_pixel_vals_in_hull = whered[np.where(whered != -1000)]
    print("There are {} pixels inside the Convex Hull".format(len(list_of_pixel_vals_in_hull)))
    mu = np.mean(list_of_pixel_vals_in_hull)
    sigma = np.std(list_of_pixel_vals_in_hull)
    print("Image metrics of these pixels:")
    print("mu = {}".format(mu))
    print("sigma = {}".format(sigma))
    print("mu/sigma = {}".format(mu/sigma))
    
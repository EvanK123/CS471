import math
from collections import namedtuple
from operator import itemgetter

# define a circle data type with center(x,y) and radius
Circle = namedtuple("Circle", ["x", "y", "r"])

# determines if the given circles form a cluster.
# circles are provided as a list of Circle(x, y, r).
def circles_form_cluster(circles):
    
    # sort by radius
    circles_sorted = sorted(circles, key=itemgetter(2) )
    
    # number of circles
    n = len(circles_sorted)

    connected = [False] * n
    # start with the smallest circle
    connected[0] = True  

    # continue checking circles until none remain
    updated = True
    while updated:
        updated = False
        for i in range(n):
            if not connected[i]:
                continue
            # check larger radius circles
            for j in range(i + 1, n): 
                 # compare current connected circle (i) with larger ones (j) 
                c1, c2 = circles_sorted[i], circles_sorted[j]
                # distance between circles
                dist = math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2) 
                 # touch / overlap condition
                if dist <= (c1.r + c2.r): 
                    if not connected[j]:
                        connected[j] = True
                        # new circle was added, loop continues
                        updated = True 

   # returns boolean (true/false) if all circles are connected
    return all(connected)
    # true = all connected
    # false = at least 1 circle is not connected


# true
circles1 = [Circle(1, 3, 0.7), Circle(2, 3, 0.4), Circle(3, 3, 0.9)]
print("Test 1 (Given):", circles_form_cluster(circles1))  

# false
circles2 = [Circle(1.5, 1.5, 1.3), Circle(4, 4, 0.7)]
print("Test 2 (Given):", circles_form_cluster(circles2))  

# false
circles3 = [Circle(0.5, 0.5, 0.5), Circle(1.5, 1.5, 1.1), Circle(0.7, 0.7, 0.4), Circle(4, 4, 0.7)]
print("Test 3 (Given):", circles_form_cluster(circles3))  

# true
circles4 = [Circle(0, 0, 1), Circle(2, 0, 1), Circle(2, 2, 1), Circle(0, 2, 1)]
print("Test 4 (User Defined):", circles_form_cluster(circles4))  

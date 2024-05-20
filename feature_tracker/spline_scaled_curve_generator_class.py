import numpy as np
import matplotlib.pyplot as plt
import math

class GenCoord():
    '''
    The purpose of this file is to generate coordinates of the curves
    in a way similar to how they are generated in pathfinder, by
    starting with a random seed and then growing point-by-point

    This is an addition in the previous series of code, wherein
    this one returns a scaled version according to the given min
    and max values of x and y.
    The min and max values of x and y can be the size of the screen.

    PS: IN CASE OF VISUAL DISPLAYS, THE MIN IS NOT ZERO (0), BUT
    RATHER NEGATIVE OF MAX_X/2.
    '''

    def get_points(self, nPoints):
        #if seeded, it will return the exact same points for all curves
        #np.random.seed(999)
        np.random.seed()
        return np.random.rand(nPoints,2)*200

    def get_circle_points(self,angle,radius,x,y):
        x_n=np.cos(angle)*radius+x
        y_n=np.sin(angle)*radius+y
        return np.array((x_n,y_n))
        
        

    def apply_random_operator(self,x,y,operator):
        if operator=='add':
            return x+y
        else:
            return x-y

    def scale_coordinates(self,x,y,x_min,x_max,y_min,y_max,x_min_gen,x_max_gen,y_min_gen,y_max_gen):
        #min-max normalization
        x_std = (x - x_min_gen) / (x_max_gen - x_min_gen)
        x_scaled = x_std * (x_max - x_min) + x_min

        y_std = (y - y_min_gen) / (y_max_gen - y_min_gen)
        y_scaled = y_std * (y_max - y_min) + y_min

        return x_scaled,y_scaled
        
        
    def get_coordinates(self,nPoints=1,length_curve=150, distance_points=0.001,angle_range=[10,30],delta_angle_max=5,
                        wiggle_room=0.95,rigidity=0.975,x_min=-115.0,x_max=115.0,y_min=-115.0,y_max=115.0,
                        draw_plot=False, attractor=None, start_point=None):
        

        '''
        Parameters (with default values) and what they do, how they work
        nPoints=1 #Random seed to be generated
        length_curve=150-1 #number of points on the curve (-1 since the first point is generated from the random seed)
        distance_points=1 #cartesian distance between two points
        angle_range = [10,30] #range of angles at which the curve can bend at any point while sampling 
        delta_angle_max=5 #controls the curviness when others do not work 
        wiggle_room=0.95 #should be non-zero and less than delta_angle_max. Controls smoothness (by variation in choice of angle), larger the number, smoother the curve
        rigidity=0.95 #between 0 (straight line) and 1 (full circle, depending on the delta_angle_max). Controls curve. More than 1 goes from spiral to random
        '''

        self.nPoints=nPoints
        self.length_curve=length_curve-1 #number of points on the curve (-1 since the first point is generated from the random seed)
        self.distance_points=distance_points #cartesian distance between two points
        self.angle_range = angle_range #range of angles at which the curve can bend at any point while sampling 
        self.delta_angle_max=delta_angle_max
        self.wiggle_room=np.float64(wiggle_room) #should be non-zero and less than delta_angle_max. Controls smoothness (by variation in choice of angle), larger the number, smoother the curve
        self.rigidity=rigidity #between 0 (straight line) and 1 (full circle, depending on the delta_
        self.x_min=x_min
        self.x_max=x_max
        self.y_min=y_min
        self.y_max=y_max
        self.draw_plot=draw_plot
        if start_point is not None:
            a=start_point
        else:
            a=self.get_points(self.nPoints)[0]
        coordinates=[]
        coordinates.append(a)
        x=[]
        y=[]
        x.append(a[0])
        y.append(a[1])
        angle_list=[np.arctan2(a[1],a[0])] #estimating the very first angle to be arctan(y/x) 

        #maintaining the min and max for both x and y
        x_min_gen=a[0]
        x_max_gen=a[0]
        y_min_gen=a[1]
        y_max_gen=a[1]
        
        #choose the random operator to apply for curve to bend downwards v/s upwards
        op=np.random.choice(['add','sub'])
        #op_circle=np.random.choice(['add','sub'])

        k=0

        for i in range(self.length_curve):
            # ang=np.random.choice(range(angle_range[0],angle_range[1]))
            # angle_list.append(ang)

            d_ang = np.random.choice(np.arange(0, self.delta_angle_max, self.wiggle_room))
            # ang=rigidity*(angle_list[i]+d_ang)
            ang = self.rigidity * self.apply_random_operator(angle_list[i], d_ang, op)
            angle_list.append(ang)

            for j in range(5):
                b = self.get_circle_points(np.radians(ang), self.distance_points, coordinates[k][0], coordinates[k][1])
                # b[0]=b[0]+(distance_points*j)
                # b[1]=b[1]+(distance_points*j)
                coordinates.append(b)

                if b[0] < x_min_gen:
                    x_min_gen = b[0]
                if b[0] > x_max_gen:
                    x_max_gen = b[0]
                if b[1] < y_min_gen:
                    y_min_gen = b[1]
                if b[1] > y_max_gen:
                    y_max_gen = b[1]

                k = k + 1
                # i=i+1
            # print i,k

            if self.draw_plot:
                # store coordinates only if we need to plot
                x.append(b[0])
                y.append(b[1])

        coordinates_s=[]
        # scale down the coordinates to the min/max range
        for i in range(len(coordinates)):
            c=self.scale_coordinates(coordinates[i][0],coordinates[i][1],
                              x_min=self.x_min,x_max=self.x_max,
                              y_min=self.y_min,y_max=self.y_max,
                              x_min_gen=x_min_gen,x_max_gen=x_max_gen,
                              y_min_gen=y_min_gen,y_max_gen=y_max_gen)
            coordinates_s.append(c)

            

        #b=get_circle_points(np.radians(20),distance_points,a[0],a[1])
        #c=get_circle_points(np.radians(20),distance_points,b[0],b[1])

        #plt.scatter((a[0],b[0],c[0]),(a[1],b[1],c[1]))
        if draw_plot:
            plt.scatter(x,y)
            plt.show()

            plt.plot(x,y)
            plt.show()
            #print(angle_list)

        #print op
        # randomly put points from left to right or vice verse
        # default direction of generation is left to right
        direction=np.random.choice([0,1])
        if direction:
            return coordinates_s
        else:
            return list(reversed(coordinates_s))


#coord=GenCoord()
#cd=coord.get_coordinates(length_curve=40, angle_range=[90,100], distance_points=.001, delta_angle_max=90,wiggle_room=.5,rigidity=.95, draw_plot=True)


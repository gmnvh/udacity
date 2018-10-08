'''
This class has static functions to calculate Google Maps coordinates.
'''

import math

class GoogleMapsCoord:
    TITLE_SIZE = 256

    @staticmethod
    def latlngToWorld(lat, lng):
        '''
            Generates an X,Y world coordinate based on the latitude, longitude

            Returns: An X,Y world coordinate
        '''

        siny = math.sin(lat * math.pi / 180)
        x = GoogleMapsCoord.TITLE_SIZE * (0.5 + lng / 360)
        y = GoogleMapsCoord.TITLE_SIZE * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))

        return x, y

    @staticmethod
    def latlngToPixel(lat, lng, zoom):
        '''
            Generates an X,Y pixel coordinate based on the latitude, longitude
            and zoom level

            Returns: An X,Y pixel coordinate
        '''

        scale = 1 << zoom

        x, y = GoogleMapsCoord.latlngToWorld(lat, lng)
        x = math.floor(x * scale)
        y = math.floor(y * scale)
        
        return int(x), int(y)

    @staticmethod
    def latlngToTile(lat, lng, zoom):
        '''
            Generates an X,Y tile coordinate based on the latitude, longitude
            and zoom level

            Returns: An X,Y tile coordinate
        '''

        x, y = GoogleMapsCoord.latlngToPixel(lat, lng, zoom)
        x = math.floor(x/GoogleMapsCoord.TITLE_SIZE)
        y = math.floor(y/GoogleMapsCoord.TITLE_SIZE)
        
        return int(x), int(y)
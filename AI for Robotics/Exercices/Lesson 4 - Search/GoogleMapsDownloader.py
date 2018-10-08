'''
    A script which when given a longitude, latitude and zoom level downloads a
    high resolution google map
'''

import os
import math
import urllib
from PIL import Image
from string import Template
from GoogleMapsCoord import GoogleMapsCoord

MAP_DESC_TEMPLATE = '''
[map]
lat_ref:$lat_ref
lng_ref:$lng_ref
zoom_ref:$zoom_ref
height:$height
width:$width
'''

class GoogleMapsDownloader:
    '''
        A class which generates high resolution google maps images given
        a longitude, latitude and zoom level
    '''

    def __init__(self, lat, lng, zoom=12, satellite=True, width=5, height=5):
        '''
            GoogleMapDownloader Constructor
            Args:
                lat:       The latitude of the location required
                lng:       The longitude of the location required
                zoom:      The zoom level of the location required, ranges
                           from 0 - 23 defaults to 12
                satellite: If True, downloads the satellite map
                width:     Number of tiles in width
                height:    Number of tiles in height
        '''
        self._lat = lat
        self._lng = lng
        self._zoom = zoom
        self.satellite = satellite
        self.width = width
        self.height = height

    def generateImage(self):
        '''
            Generates an image by stitching a number of google map tiles together.

            Args:
                start_x:        The top-left x-tile coordinate
                start_y:        The top-left y-tile coordinate
                tile_width:     The number of tiles wide the image should be -
                                defaults to 5
                tile_height:    The number of tiles high the image should be -
                                defaults to 5
            Returns:
                A high-resolution Goole Map image and a string with map
                description
        '''

        # Gets x, y tile start positions
        start_x, start_y = GoogleMapsCoord.latlngToTile(self._lat, self._lng, self._zoom)
        print 'Tiles coordinates: ', start_x, start_y

        # Determine the size of the image
        width, height = 256 * self.width, 256 * self.height

        #Create a new image of the size require
        map_img = Image.new('RGB', (width, height))

        for x in range(0, self.width):
            for y in range(0, self.height) :
                print 'Downloading GoogleMaps tile ', x, ', ', y
                if self.satellite is True:
                    url = 'https://mt0.google.com/vt/lyrs=y&?x='+str(start_x+x)+'&y='+str(start_y+y)+'&z='+str(self._zoom)
                else:
                    url = 'https://mt0.google.com/vt?x='+str(start_x+x)+'&y='+str(start_y+y)+'&z='+str(self._zoom)

                current_tile = str(x)+'-'+str(y)
                urllib.urlretrieve(url, current_tile)

                im = Image.open(current_tile)
                map_img.paste(im, (x*256, y*256))

                os.remove(current_tile)

        # Map description
        l_desc = {'lat_ref':self._lat, 'lng_ref':self._lng, 'zoom_ref':self._zoom, 'height':self.height, 'width':self.width}
        f_desc = Template(MAP_DESC_TEMPLATE)
        s_desc = f_desc.substitute(l_desc)

        return map_img, s_desc

def main():
    # Create a new instance of GoogleMap Downloader

    # Nexteer test track
    gmd = GoogleMapsDownloader(43.405802, -83.886641, 19, height=7)

    try:
        # Get the high resolution image
        img, desc = gmd.generateImage()
        print desc
        pass
    except IOError:
        print("Could not generate the image - try adjusting the zoom level and checking your coordinates")
    else:
        # Save the image to disk
        img.save("high_resolution_image_2.png")

        # Save map description
        f = open("map_desc.txt", "w")
        f.write(desc)
        f.close()
        print("The map has successfully been created")


if __name__ == '__main__':  main()
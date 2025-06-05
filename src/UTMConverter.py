'''
This program provides functions for converting between UTM and Lat/Long coordinates.
'''

import pyproj 

# credit to Ting On Chan's solution at:
# https://stackoverflow.com/questions/6778288/lat-lon-to-utm-to-lat-lon-is-extremely-flawed-how-come

def convertToUTM(lat, long, zone):
    projection = pyproj.Proj("+proj=utm +zone=" + str(zone) + ",+north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    x, y = projection(long, lat)

    return (x, y)

def convertToLatLong(x, y, zone):
    projection = pyproj.Proj("+proj=utm +zone=" + str(zone) + ",+north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    long, lat = projection(x, y, inverse=True)

    return (lat, long)

'''
print(convertToUTM(41.29125, -95.857778))
print(utm.from_latlon(41.29125, -95.857778))

print(convertToLatLong(260704, 4575029))
print(utm.to_latlon(260704, 4575029, 15, "T"))
'''

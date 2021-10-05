from pyproj import Transformer
import geopandas as gpd
import pandas as pd
from shapely.wkt import loads
import math
import matplotlib.pyplot as plt
import numpy as np

transformer = Transformer.from_crs( "epsg:4326", "epsg:26910")

def calculate_bearing(x1,y1,x2,y2):
    tangent = (x2-x1, y2-y1)
    angle = np.arctan2(tangent[1], tangent[0])*180/np.pi
    return angle



def distance_utm(origin, destination):
    origin_utm = transformer.transform(origin[1], origin[0])
    destination_utm = transformer.transform(destination[1], destination[0])
    d = pow(pow(origin_utm[0]-destination_utm[0],2)+pow(origin_utm[1]-destination_utm[1],2), 0.5)
    return d

def add_new_point(start, end, proportion):
    new_x = (end[0]-start[0])*proportion+ start[0]
    new_y = (end[1]-start[1])*proportion+ start[1]
    return new_x,new_y

def get_new_coord(coord):
    output = [coord[0]]
    j = 0
    for i, coord_tmp in enumerate(coord[:-1]):
        if i < j:
            continue
        else:
            con_find = True
            while con_find:
                j += 1
                dis_tmp = distance_utm(coord[i], coord[j])
                if dis_tmp >= 12:
                    add_num = int(dis_tmp // 6) - 1
                    for k in range(add_num):
                        proportion = 6 * (k + 1) / dis_tmp
                        coord_tmp_new = add_new_point(coord[i], coord[j], proportion)
                        output.append(coord_tmp_new)
                    output.append(coord[j])
                    con_find = False
                elif dis_tmp >= 6:
                    output.append(coord[j])
                    con_find = False
                else:
                    con_find = False
                    pass
    return output

# newl = get_new_coord([(-122.687695, 37.908725), (-122.687573, 37.908212), (-122.6875361, 37.9081832)])
roads_df = pd.read_csv('../projects/bolinas/network_inputs/bolinas_edges_sim.csv')
# roads_gdf = gpd.GeoDataFrame(roads_df, crs='epsg:4326', geometry=roads_df['geometry'].map(loads)).to_crs(26910)
roads_df['geometry'] = roads_df['geometry'].map(loads)

roads_df['points_list'] = roads_df.apply(lambda x: [y for y in x['geometry'].coords], axis=1)
roads_df['long_points_list'] = roads_df.apply(lambda x: get_new_coord(x['points_list']), axis=1)
# list_points = mypoints.values[0]

link_id=192
points_list = roads_df['points_list'].iloc[link_id]
long_list = roads_df['long_points_list'].iloc[link_id]
roads_df.to_csv('links_new.csv', index=False)

print("list: {}".format(points_list))
print("long list: {}".format(long_list))

dis_points, dis_long = [], []
#
for i in range(len(points_list)-1):
    dis_tmp = distance_utm(points_list[i], points_list[i+1])
    dis_points.append(dis_tmp)
print("distance origin: {}".format(dis_points))

angle = []
for i in range(len(long_list)-1):
    dis_tmp = distance_utm(long_list[i], long_list[i+1])
    angel_tmp = calculate_bearing(long_list[i][0], long_list[i][1], long_list[i + 1][0], long_list[i + 1][1])
    dis_long.append(dis_tmp)
    angle.append(angel_tmp)
print("distance after: {}".format(dis_long))
print("angle list: {}".format(angle))


# new_coord = get_new_coord(gdf_line)
# print(new_coord)
# new_coord_final1 = []
#
fig = plt.figure()
ax = plt.subplot()
#
for i in range(len(long_list)):
#     bearing1 = calculate_bearing(new_coord[i][0], new_coord[i][1], new_coord[i+1][0], new_coord[i+1][1])
#     new_coord_final1.append((i, new_coord[i][0],new_coord[i][1],bearing1))
    ax.text(long_list[i][0], long_list[i][1],str(i))
#
#     # dis_tmp = distance_utm(new_coord[i], new_coord[i+1])
#     # dis_after.append(dis_tmp)
#
# new_coord_final1.append((i+1,new_coord[-1][0],new_coord[-1][1],new_coord_final1[-1][-1]))
#
#
# # print(dis_after)
# print("new_coord_final1: {}".format(new_coord_final1))
#
x = [i[0] for i in long_list]
y = [i[1] for i in long_list]
#
plt.plot(x,y)
#
# # player_points =pd.read_csv("../output/player.txt")
# # print(player_points)
# x2, y2 = [], []
# # for row in player_points.iterrows():
# #     if row[1]['link_id']==192:
# #         x2.append(row[1]["lon"])
# #         y2.append(row[1]["lat"])
# #         print(row[1]["heading"])
# #         ax.text(row[1]["lon"], row[1]["lat"], row[1]['t'])
# # ax.scatter(x2, y2)
#
plt.show()

import numpy as np 
import pandas as pd 
import geopandas as gpd
from pyproj import Proj, Transformer

def veh_offset(x1, y1, x2, y2):
    # tangential slope approximation
    tangent = (x2-x1, y2-y1)
    perp = (y2-y1, -(x2-x1))
    mode = np.sqrt(perp[0]**2+perp[1]**2)
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    delta_x = perp[0]/mode*1.75
    delta_y = perp[1]/mode*1.75

    angle = np.arctan2(tangent[1], tangent[0])*180/np.pi
    return (mid_x + delta_x, mid_y + delta_y), angle

    # return (mid_x + delta_x, mid_y + delta_y), (tangent[0]/mode, tangent[1]/mode)


def extract_vehicle_locations(network, t, agent_id=None, logger = None):
    links_dict = network.links.copy()
    node_dict = network.nodes.copy()
    queue_vehicle_position = []
    run_vehicle_position = []
    transformer = Transformer.from_crs( "epsg:26910", "epsg:4326")

    is_stop = False
    trace_agent = network.agents[agent_id]
    agent_link = network.node2link_dict[(trace_agent.current_link_start_nid, trace_agent.current_link_end_nid)]
    select_node_id = trace_agent.current_link_end_nid
    select_node = node_dict[select_node_id]
    select_links_id = select_node.outgoing_links+list(select_node.incoming_links.keys())

    # for link_id, link in links_dict.items():
    for link_id in select_links_id:
        link = links_dict[link_id]
        
        ### skip virtual links
        if link.link_type == 'v':
            continue

        ### link stats
        link_fft = link.fft
        link_geometry = link.geometry
        link_length = link.geometry.length

        run_index_tmp = None

        ### queuing
        queue_end = link_length - 4
        for q_index_tmp, q_veh_id in enumerate(link.queue_vehicles):
            q_veh_loc = queue_end
            queue_end = max(queue_end-8, 4)
            q_veh_coord_1 = link_geometry.interpolate(q_veh_loc/link_length-0.001, normalized=True)
            q_veh_coord_2 = link_geometry.interpolate(q_veh_loc/link_length+0.001, normalized=True)
            q_veh_coord_offset, q_veh_dir = veh_offset(q_veh_coord_1.x, q_veh_coord_1.y, q_veh_coord_2.x, q_veh_coord_2.y)
            # queue_vehicle_position.append([q_veh_id, 'q', link_id, q_veh_coord_offset[0], q_veh_coord_offset[1], q_veh_dir])
            lat_tmp2, lon_tmp2 = transformer.transform(q_veh_coord_offset[0], q_veh_coord_offset[1])

            queue_vehicle_position.append([q_veh_id, q_veh_coord_offset[0], q_veh_coord_offset[1], lon_tmp2, lat_tmp2, q_veh_dir, link_id])

        ### running
        for run_index_tmp, r_veh_id in enumerate(link.run_vehicles):
            r_veh_current_link_enter_time = network.agents[r_veh_id].current_link_enter_time
            if link_length*(t-r_veh_current_link_enter_time)/link_fft>queue_end:
                r_veh_loc = queue_end
                queue_end = max(queue_end-8, 0)
            else:
                r_veh_loc = link_length*(t-r_veh_current_link_enter_time)/link_fft
            r_veh_loc = max(r_veh_loc, 4)
            r_veh_coord_1 = link_geometry.interpolate(r_veh_loc/link_length-0.001, normalized=True)
            r_veh_coord_2 = link_geometry.interpolate(r_veh_loc/link_length+0.001, normalized=True)
            r_veh_coord_offset, r_veh_dir = veh_offset(r_veh_coord_1.x, r_veh_coord_1.y, r_veh_coord_2.x, r_veh_coord_2.y)
            # run_vehicle_position.append([r_veh_id, 'r', link_id, r_veh_coord_offset[0], r_veh_coord_offset[1], r_veh_dir])

            lat_tmp2, lon_tmp2 = transformer.transform(r_veh_coord_offset[0], r_veh_coord_offset[1])
            run_vehicle_position.append([r_veh_id, r_veh_coord_offset[0], r_veh_coord_offset[1], lon_tmp2, lat_tmp2, r_veh_dir, link_id])
        # if agent_id and agent_link == link_id:
        if agent_link == link_id:
            # if run_index_tmp and r_veh_id==agent_id and r_veh_loc<8:
            #     is_stop=True
            # elif (not run_index_tmp) and q_veh_id ==agent_id and q_veh_loc<8:
            #     is_stop=True
            if len(link.queue_vehicles)>0 and agent_id == link.queue_vehicles[0]:
                logger.info("queue_vehicles: {}, in the first of queue, need stop!".format(link.queue_vehicles))
                is_stop=True
            # elif len(link.queue_vehicles)==0 and len(link.run_vehicles)>0 and agent_id == link.run_vehicles[0]:
            #     logger.info("queue_vehicles is 0, run_vehicles is {} in the first of queue, need stop!".format(link.queue_vehicles, link.run_vehicles))
            #     is_stop=True
            else:
                is_stop = False


    '''

    veh_df = pd.DataFrame(queue_vehicle_position + run_vehicle_position, columns=['veh_id', 'status', 'link_id', 'lon_offset_utm', 'lat_offset_utm', 'angle'])

    # veh_df = pd.DataFrame(queue_vehicle_position + run_vehicle_position, columns=['veh_id', 'status', 'link_id', 'lon_offset_utm', 'lat_offset_utm', 'dir_x', 'dir_y'])
    veh_df['lon_offset_sumo'] = veh_df['lon_offset_utm']-525331.68
    veh_df['lat_offset_sumo'] = veh_df['lat_offset_utm']-4194202.74

    # print(veh_df.iloc[0])
    # print(veh_df['lon_offset_utm'].iloc[0], veh_df['lon_offset_utm'].iloc[0]-518570.38)
    # veh_df.to_csv(simulation_outputs+'/veh_loc_interpolated/veh_loc_t{}.csv'.format(t), index=False)
    veh_gdf = gpd.GeoDataFrame(veh_df, crs='epsg:32610', geometry=gpd.points_from_xy(veh_df.lon_offset_utm, veh_df.lat_offset_utm))
    # veh_gdf.loc[veh_gdf['veh_id']==game_veh_id, 'status'] = 'g'

        
    return veh_gdf
    '''

    whole_position=[]
    whole_veh = queue_vehicle_position+run_vehicle_position
    whole_vehs = pd.DataFrame(whole_veh, columns=['veh_id', 'lon_offset_utm', 'lat_offset_utm','lon', 'lat', 'angle', 'link_id'])
    whole_vehs["t"] = t

    if len(whole_veh)>0:
        whole_veh.sort(key=keyfun)
        for veh in whole_veh:
            whole_position.append(','.join(map(str,veh)))


    return len(whole_vehs), ';'.join(whole_position), whole_vehs, is_stop

def keyfun(item):
    return item[0]
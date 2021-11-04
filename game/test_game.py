# %%

import os
import gc
import sys
import time
import json
import random
import logging
import warnings
import numpy as np
import pandas as pd
from pyproj import Transformer
import bisect
pd.set_option('display.max_columns', 500)
import geopandas as gpd
from ctypes import c_double
# geometry
import shapely.wkt
import shapely.ops
from shapely.wkt import loads
from shapely.ops import substring
from shapely.geometry import Point
# plot
import matplotlib.pyplot as plt
from matplotlib import cm
import queue
import math
### dir
from model.queue_class import Network
import game.extract_vehicle_locations as extract_vehicle_location
from communicate import server_rpc
print(sys.path)
import ast

root_dir = server_rpc.root_directory
links_list = pd.read_csv(os.path.join(root_dir ,'split_link/links_new.csv'))

def preparation(random_seed=0, fire_id=None, comm_id=None, vphh=None, visitor_cnts=None, contra_id=None,
                link_closed_time=None, closed_mode=None, shelter_scen_id=None, scen_nm=None, logger=None):
    ### logging and global variables

    project_location = '/projects/bolinas'
    network_file_edges = project_location + '/network_inputs/bolinas_edges_sim.csv'
    network_file_nodes = project_location + '/network_inputs/bolinas_nodes_sim.csv'
    network_file_special_nodes = project_location + '/network_inputs/bolinas_special_nodes.json'
    demand_files = [
        project_location + '/demand_inputs/od_csv/resident_visitor_od_rs{}_commscen{}_vphh{}_visitor{}.csv'.format(
            random_seed, comm_id, vphh, visitor_cnts)]
    simulation_outputs = ''  ### scratch_folder

    if contra_id == '0':
        cf_files = []
    elif contra_id == '1':
        cf_files = [project_location + '/network_inputs/bolinas_contraflow.csv']
    else:
        cf_files = []

    scen_nm = scen_nm

    logger.info(scen_nm)
    print('log file created for {}'.format(scen_nm))

    ### network
    with open(root_dir + network_file_special_nodes) as special_nodes_file:
        special_nodes = json.load(special_nodes_file)
    network = Network()
    network.dataframe_to_network(project_location=project_location, network_file_edges=network_file_edges,
                                 network_file_nodes=network_file_nodes, cf_files=cf_files,
                                 special_nodes=special_nodes,
                                 scen_nm=scen_nm)
    network.add_connectivity()

    ### demand
    network.add_demand(demand_files=demand_files)
    logger.info('total numbers of agents taken {}'.format(len(network.agents.keys())))

    return {'network': network}, {'scen_nm': scen_nm, 'simulation_outputs': simulation_outputs, 'fire_id': fire_id,
                                  'comm_id': comm_id,
                                  'special_nodes': special_nodes,
                                  'link_closed_time': link_closed_time, 'closed_mode': closed_mode,
                                  'shelter_scen_id': shelter_scen_id}, {'in_fire_dict': {}, 'shelter_capacity_122': 200,
                                                                        'shelter_capacity_202': 100}


def link_model(t, network, link_closed_time=None, closed_mode=None):
    # run link model
    for link_id, link in network.links.items():
        link.run_link_model(t, agent_id_dict=network.agents)

    return network


def node_model(t, network, special_nodes=None, driver_id=None, driver_next_link = None, isStop = None):
    # only run node model for those with vehicles waiting
    node_ids_to_run = set([link.end_nid for link in network.links.values() if len(link.queue_vehicles) > 0])
    if t==9:
        print("stop")
    # run node model
    for node_id in node_ids_to_run:
        if node_id==125:
            print("stop")
        node = network.nodes[node_id]
        n_t_move, transfer_links = node.run_node_model(t, node_id_dict=network.nodes, link_id_dict=network.links,
                                                       agent_id_dict=network.agents,
                                                       node2link_dict=network.node2link_dict,
                                                       special_nodes=special_nodes, driver_id=driver_id, driver_next_link = driver_next_link, isStop=isStop)

    return network


def one_step(t, data, config, update_data, driver_id = None):
    network = data['network']

    scen_nm, simulation_outputs, fire_id, comm_id, special_nodes, link_closed_time, closed_mode, shelter_scen_id = \
    config['scen_nm'], config['simulation_outputs'], config['fire_id'], config['comm_id'], config['special_nodes'], \
    config['link_closed_time'], config['closed_mode'], config['shelter_scen_id']

    in_fire_dict, shelter_capacity_122, shelter_capacity_202 = update_data['in_fire_dict'], update_data[
        'shelter_capacity_122'], update_data['shelter_capacity_202']

    ### update link travel time before rerouting
    reroute_freq_dict = {'1': 300, '2': 900, '3': 1800}
    reroute_freq = reroute_freq_dict[comm_id]

    ### reset link congested counter
    for link in network.links.values():
        link.congested = 0
        if (t % 100 == 0):
            link.update_travel_time_by_queue_length(network.g)

    if t == 0:
        for agent_id, agent in network.agents.items():
            if (agent_id!=driver_id):
                network.agents[agent_id].departure_time = np.random.randint(1, 100)
            else:
                network.agents[agent_id].departure_time = float("inf")
                # network.agents[agent_id].origin_nid =
                # network.agents[agent_id].destin_nid = -1
    elif t==1:
        network.agents[driver_id].get_path(t, g=network.g)
        network.agents[driver_id].find_next_link(node2link_dict=network.node2link_dict)

        # data["network"].agents[driver_id].origin_nid = server_rpc.start_nid
        # data["network"].agents[driver_id].destin_nid = server_rpc.end_nid
        # network.agents[driver_id].departure_time = 1



    #     if t%120==0:
    #         print(t, network.agents[0].current_link_end_nid)

    ### agent model
    t_agent_0 = time.time()
    stopped_agents_list = []
    for agent_id, agent in network.agents.items():
        if agent_id==447:
            print("stop")
        ### first remove arrived vehicles
        if agent.status == 'arrive':
            network.agents_stopped[agent_id] = (agent.status, t, agent.agent_type)
            stopped_agents_list.append(agent_id)
            continue
        ### find congested vehicles: spent too long in a link
        current_link = network.links[network.node2link_dict[(agent.current_link_start_nid, agent.current_link_end_nid)]]
        if (current_link.link_type != 'v') and (agent.current_link_enter_time is not None) and (
                t - agent.current_link_enter_time > 3600 * 0.5):
            current_link.congested += 1
            if (shelter_scen_id == '0'):
                if (t - agent.current_link_enter_time > 3600 * 3):
                    agent.status = 'shelter_a1'
            elif shelter_scen_id == '1':
                pass
            else:
                pass
        ### agents need rerouting
        # initial route
        # if (t == 0) or (t % reroute_freq == agent_id % reroute_freq):
        if (t == 0) or (t % reroute_freq == 0):
            routing_status = agent.get_path(t, g=network.g)
            agent.find_next_link(node2link_dict=network.node2link_dict)
        agent.load_vehicle(t, node2link_dict=network.node2link_dict, link_id_dict=network.links)
        ### remove passively sheltered vehicles immediately, no need to wait for node model
        if agent.status in ['shelter_p', 'shelter_a1', 'shelter_park']:
            current_link.queue_vehicles = [v for v in current_link.queue_vehicles if v != agent_id]
            current_link.run_vehicles = [v for v in current_link.run_vehicles if v != agent_id]
            network.nodes[agent.current_link_end_nid].shelter_counts += 1
            network.agents_stopped[agent_id] = (agent.status, t, agent.agent_type)
            stopped_agents_list.append(agent_id)
    for agent_id in stopped_agents_list:
        del network.agents[agent_id]
    return network



def calculate_position(network, config, t, agent_id = None, logger=None, isStop = None):
    scen_nm, simulation_outputs, fire_id, comm_id, special_nodes, link_closed_time, closed_mode, shelter_scen_id = \
    config['scen_nm'], config['simulation_outputs'], config['fire_id'], config['comm_id'], config['special_nodes'], \
    config['link_closed_time'], config['closed_mode'], config['shelter_scen_id']

    ### link model
    ### Each iteration in the link model is not time-consuming. So just keep using one process.
    t_link_0 = time.time()
    network = link_model(t, network, link_closed_time=link_closed_time, closed_mode=closed_mode)
    t_link_1 = time.time()

    trace_agent = network.agents[agent_id]
    current_link_id = network.node2link_dict[(trace_agent.current_link_start_nid, trace_agent.current_link_end_nid)]
    next_link_id = trace_agent.next_link


    ### node model
    t_node_0 = time.time()
    network = node_model(t, network, special_nodes=special_nodes, driver_id=agent_id, driver_next_link = next_link_id, isStop = isStop)
    t_node_1 = time.time()

    logger.info("t:{}, current link {}' s queue_vehicles: {}, next link {}' s queue_vehicles: {}".format(t, current_link_id, network.links[current_link_id].queue_vehicles, next_link_id , network.links[next_link_id].queue_vehicles))
    logger.info("t:{}, current link {}' s run_vehicles: {}, next link {}' s run_vehicles: {}".format(t, current_link_id, network.links[current_link_id].run_vehicles, next_link_id, network.links[next_link_id].run_vehicles ))

    # stop
    if len(network.agents) == 0:
        logging.info("all agents arrive at destinations")
        return network, 'stop'
    else:
        return network, 'continue'


def plot_run_queue_fire(t, current_link=None, fire_id=None, comm_id=None, shelter_scen_id=None, scen_nm=None,
                        roads_gdf=None, fire_raster=None, fire_raster_extent=None, link_stats_gdf=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ### road network
    roads = roads_gdf.plot(ax=ax, lw=0.5, color='gray', alpha=0.5)
    #     print(type(roads_gdf), roads_gdf.head())
    ### current position
    current_link_plot = current_link.plot(ax=ax, lw=3, color='purple', alpha=1)
    #     print(type(current_link), current_link.head())
    ### flames
    if (fire_raster is not None):
        cax = fig.add_axes([0.2, 0.2, 0.6, 0.02])
        cmap = cm.get_cmap('Oranges', 5)
        cmap.set_under(color='white', alpha=100)
        flames = ax.imshow(fire_raster, extent=fire_raster_extent, cmap=cmap, vmin=0, vmax=5)
        fig.colorbar(flames, orientation='horizontal', cax=cax)
    ### queue plot
    queue_gdf = link_stats_gdf[link_stats_gdf['q'] > 0].copy()
    queue_gdf['geometry'] = queue_gdf.apply(
        lambda x: shapely.ops.substring(x['geometry'], 1 - x['q'] * 8 / (x['length'] * x['lanes']), 1, normalized=True),
        axis=1)
    queue = queue_gdf.plot(ax=ax, lw=1, color='red')
    ### run plot
    run_gdf = link_stats_gdf[link_stats_gdf['r'] > 0].copy()
    run_gdf['geometry'] = run_gdf.apply(
        lambda x: shapely.ops.substring(x['geometry'], 0, x['r'] * 8 / (x['length'] * x['lanes']), normalized=True),
        axis=1)
    run = run_gdf.plot(ax=ax, lw=1, color='blue')
    (xlim_1, xlim_2) = ax.get_xlim()
    ax.set_xlim([xlim_1 + (xlim_2 - xlim_1) * 0.1, xlim_1 + (xlim_2 - xlim_1) * 0.8])
    (ylim_1, ylim_2) = ax.get_ylim()
    ax.set_ylim([ylim_1 + (ylim_2 - ylim_1) * 0.1, ylim_1 + (ylim_2 - ylim_1) * 0.6])
    shelter_text = 'no shelter'  # {1: 'No sheltering', 2: 'Sheltering'}[shelter_scen_id]
    ax.text(0.6, 0.75,
            'Fire location {}\nComm. scenario {}\n{}\n{:.1f} Hr'.format(fire_id, comm_id, shelter_text, t / 3600),
            fontsize=22, transform=ax.transAxes)


#     plt.show()
#     plt.savefig('../visualization_outputs/python_map/{}_t{}.png'.format(scen_nm, t))
#     plt.close()

# %%
def initialize(logger):
    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)

    fire_id = '1'  # '1', '2', '3'
    comm_id = '1'  # '1', '2', '3'
    vphh = 2  # vehicles per household
    visitor_cnts = 300
    contra_id = '0'
    shelter_scen_id = '0'
    link_closed_time = 0
    closed_mode = 'flame'

    # base network as the base layer of plotting
    roads_df = pd.read_csv(os.path.join(root_dir,'projects/bolinas/network_inputs/bolinas_edges_sim.csv'))
    roads_gdf = gpd.GeoDataFrame(roads_df, crs='epsg:4326', geometry=roads_df['geometry'].map(loads)).to_crs(26910)

    # set scenario name
    scen_nm = "r{}_fire{}_comm{}_vphh{}_vistor{}_contra{}_close{}m{}_shelter{}".format(random_seed, fire_id, comm_id,
                                                                                       vphh,
                                                                                       visitor_cnts, contra_id,
                                                                                       link_closed_time, closed_mode,
                                                                                       shelter_scen_id)

    # reset random seed
    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)

    data, config, update_data = preparation(random_seed=random_seed, fire_id=fire_id, comm_id=comm_id, vphh=vphh,
                                            visitor_cnts=visitor_cnts, contra_id=contra_id,
                                            shelter_scen_id=shelter_scen_id,
                                            link_closed_time=link_closed_time, closed_mode=closed_mode, scen_nm=scen_nm, logger = logger)
    return data, config, update_data

def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def get_image(points_df, whole_vehs, trace_agent, current_link_id):
    if type(current_link_id)!=int:
        return None

    all_points_in_link = points_df[points_df['link_id'] == current_link_id]
    all_points_index = np.where(points_df['link_id']==current_link_id)
    if trace_agent.agent_id not in list(whole_vehs['veh_id']):
        return None
    player_index_list = list(whole_vehs['veh_id']).index(trace_agent.agent_id)

    player_position_x = whole_vehs['lon_offset_utm'][player_index_list]
    player_position_y = whole_vehs['lat_offset_utm'][player_index_list]

    x_list = list(all_points_in_link['x'])

    for i in range(1, len(x_list)):
        if player_position_x>=x_list[i-1] and player_position_x<=x_list[i] or (player_position_x<=x_list[i-1] and player_position_x>=x_list[i]):
            player_index=i+all_points_index[0][0]
            position_left = distance(all_points_in_link['x'][player_index - 1],
                                     all_points_in_link['y'][player_index - 1],
                                     player_position_x, player_position_y)
            postion_right = distance(player_position_x, player_position_y,
                                     all_points_in_link['x'][player_index], all_points_in_link['y'][player_index])
            if position_left < postion_right:
                image_position = str(all_points_in_link['lon'][player_index - 1]), str(all_points_in_link['lat'][player_index - 1]),\
                                 str(all_points_in_link['angle'][player_index - 1])
            else:
                image_position = str(all_points_in_link['lon'][player_index]), str(all_points_in_link['lat'][player_index]),\
                                  str(all_points_in_link['angle'][player_index])

            return '('+','.join(image_position)+')'

    return None



# def test_each(t, data, config, update_data, checked_links, logger):
def test_each(t, data, config, update_data, logger, old_link_id, points_df):

    # for t in range(0, 1001):

    # run simulation for one step
    network = one_step(t, data, config, update_data)
    network, status = calculate_position(network, config, t)

    # plot vehicle routing options
    agent_id = 447
    trace_agent = network.agents[agent_id]
    current_link_id = network.node2link_dict[(trace_agent.current_link_start_nid, trace_agent.current_link_end_nid)]
    if t == 0:
        old_link_id = current_link_id

        # if (current_link_id not in checked_links) and (len(network.links[current_link_id].queue_vehicles) > 0)\
    #         and (network.links[current_link_id].queue_vehicles[0] == trace_agent.agent_id):

    # current link
    print('\n')
    print('current time {}'.format(t))
    print('vehicle {} is on link {}'.format(trace_agent.agent_id, current_link_id))

    # alternative links
    alternative_links = []
    checked_outgoing_links = []
    current_link_angle = network.links[current_link_id].out_angle
    for alt_link_id in network.nodes[trace_agent.current_link_end_nid].outgoing_links:
        alternative_link_angle = (current_link_angle - network.links[alt_link_id].in_angle) / 3.14 * 180
        if alternative_link_angle < -180: alternative_link_angle += 360
        if alternative_link_angle > 180: alternative_link_angle -= 360
        alternative_links.append([alt_link_id, alternative_link_angle])
    # print(alternative_links)
    alternative_links_by_direction = {'forward': None, 'left': None, 'right': None, 'back': None}
    # forward
    # (forward_link, angle) = min(alternative_links, key=lambda l: abs(l[1] - 0))
    potential_forward_link = [(forward_link, angle) for (forward_link, angle) in alternative_links if (abs(angle - 0) <= 45 and forward_link not in checked_outgoing_links)]
    if (len(potential_forward_link)>0):
        alternative_links_by_direction['forward'] = min(potential_forward_link, key=lambda l: abs(l[1] - 0))
        checked_outgoing_links.append(alternative_links_by_direction['forward'])

    # left
    potential_left_link = [(left_link, angle) for (left_link, angle) in alternative_links if (abs(angle - (-90)) <= 45 and (left_link not in checked_outgoing_links))]
    if (len(potential_left_link)>0):
        alternative_links_by_direction['left'] = min(potential_left_link, key=lambda l: abs(l[1]- (-90)))
        checked_outgoing_links.append(alternative_links_by_direction['left'])

    # right
    potential_right_link = [(right_link, angle) for (right_link, angle) in alternative_links if (abs(angle - 90) <= 45 and (right_link not in checked_outgoing_links))]
    # (right_link, angle) = min(alternative_links, key=lambda l: abs(l[1] - 90))
    if(len(potential_right_link)>0):
        alternative_links_by_direction['right'] = min(potential_right_link, key=lambda l: abs(l[1]- 90))
        checked_outgoing_links.append(alternative_links_by_direction['right'])

    # backward
    potential_back_link = [(back_link, angle) for (back_link, angle) in alternative_links if (abs(angle - 0) > 135 and (back_link not in checked_outgoing_links))]
    if len(potential_back_link)>0:
        alternative_links_by_direction['back'] = max(potential_back_link, key=lambda l: abs(l[1]))
        checked_outgoing_links.append(alternative_links_by_direction['back'])

    print(alternative_links_by_direction)
    logger.info(alternative_links_by_direction)
    print('   turning angles {}'.format(alternative_links))

    #         link_stats_df = pd.DataFrame(
    #             [(link.link_id, len(link.queue_vehicles), len(link.run_vehicles)) for link in network.links.values() if (link.link_type!='v') and (len(link.queue_vehicles)+len(link.run_vehicles)>0)], columns=['link_id', 'q', 'r'])
    #         link_stats_gdf = roads_gdf[['eid', 'length', 'lanes', 'geometry']].merge(link_stats_df, how='right', left_on='eid', right_on='link_id')
    #         if link_stats_gdf.shape[0]>0:
    #             current_link = roads_gdf.loc[roads_gdf['eid'] == current_link_id]
    #             plot_run_queue_fire(t, current_link=current_link, scen_nm=scen_nm, fire_id=fire_id, comm_id=comm_id, shelter_scen_id=shelter_scen_id, roads_gdf=roads_gdf, link_stats_gdf=link_stats_gdf)
    #             plt.show()

    #         print("agent position at time {}s, current link {}, next link {}, alternative next links {}".format(t, current_link_id, trace_agent.next_link, network.nodes[trace_agent.current_link_end_nid].outgoing_links ))
    #         alternative_next_links = {link_id: (len(network.links[link_id].run_vehicles), len(network.links[link_id].queue_vehicles)) for link_id in network.nodes[trace_agent.current_link_end_nid].outgoing_links}
    #         print(['Link {} has {} running vehicle and {} queuing vehicles'.format(k, v[0], v[1]) for k, v in alternative_next_links.items()])

    # agent_next_link_direction = input('Choose from the following actions: ' + ', '.join(
    #     [k for k, v in alternative_links_by_direction.items() if v is not None]))

    agent_next_link_direction = gl.get_value("direction_message")
    valid_links_by_direction = [k for k,v in alternative_links_by_direction.items() if v is not None]

    if agent_next_link_direction in valid_links_by_direction:
        logger.info(agent_next_link_direction)
        agent_next_link = alternative_links_by_direction[agent_next_link_direction][0]
    else:
        agent_next_link = network.agents[agent_id].next_link
            # random.choice(valid_links_by_direction)

    print('  next link is {}'.format(agent_next_link))
    logger.info("current link is {}".format(current_link_id))
    logger.info('  next link is {}'.format(agent_next_link))

    if old_link_id!=current_link_id:

        gl.set_value("direction_message","test")
    agent_next_link_direction = None
    # close all other directions
    for [link_id, _] in alternative_links:
        if link_id != agent_next_link:
            # print('close {}'.format(link_id))
            link = network.links[link_id]
            network.g.update_edge(link.start_nid, link.end_nid, c_double(1e8))

    routing_status = network.agents[agent_id].get_path(t, g=network.g)
    network.agents[agent_id].find_next_link(node2link_dict=network.node2link_dict)
    # print(agent_next_link, network.agents[0].next_link)

    for [link_id, _] in alternative_links:
        if link_id != agent_next_link:
            link = network.links[link_id]
            network.g.update_edge(link.start_nid, link.end_nid, c_double(link.fft))
    data['network'] = network

    # extract vehicle positions
    # if t in [998, 999, 1000]:
    veh_num, vehicle_positions, whole_vehs, isStop = extract_vehicle_location.extract_vehicle_locations(network, t)

    # server.output = str(t)+": \n"+vehicle_positions

    # output.put(str(t)+":"+vehicle_positions)

    logger.info("veh_num: "+str(len(whole_vehs)))
    
    # global output
    output = str(t)+":"+vehicle_positions


    # print(str(t)+": \n"+vehicle_positions)
        # vehicle_positions.to_csv('test_veh_positions_t{}.csv'.format(t))



    image_position = get_image(points_df, whole_vehs, trace_agent, current_link_id)

    # checked_links.append(current_link_id)


    # return data, config, update_data,checked_links, output
    old_link_id = current_link_id
    return data, config, update_data, output, old_link_id, image_position

def calculate_perpen_point(x0, y0, x1, y1, x2, y2):
    k = -((x1-x0)*(x2-x1)+(y1-y0)*(y2-y1))/(pow(x2-x1,2)+pow(y2-y1,2))
    xn = k*(x2-x1)+x1
    yn = k*(y2-y1)+y1
    return xn,yn

def get_image_rpc(driver_link_list, driver_position):
    find = False
    for i in range(len(driver_link_list)-1):
        x1, y1 = driver_link_list[i]
        x2, y2 = driver_link_list[i+1]
        x0, y0 = driver_position
        perpe_point = calculate_perpen_point(x0, y0, x1, y1, x2, y2)
        if min(x1,x2)<=perpe_point[0]<max(x1,x2) and min(y1, y2) <=perpe_point[1]< max(y1,y2):
            find = True
            # return "p{}_{}_{}.jpg".format(i,x1,y1)
            return "p{}_{}_{}__final.jpg".format(i, x1, y1)

    if not find:
        print("stop")
    return ""




# def test_each(t, data, config, update_data, checked_links, logger):
def test_each_grpc(t, data, config, update_data, logger, direction_tmp, isStop):
    # for t in range(0, 1001):
    # run simulation for one step
    agent_id = 447
    if data["network"].agents[agent_id].current_link_end_nid == data["network"].agents[agent_id].destin_nid:
        print("wrong")
    network = one_step(t, data, config, update_data, agent_id)

    trace_agent = network.agents[agent_id]
    if trace_agent.departure_time != float("inf"):
        current_link_id = network.node2link_dict[(trace_agent.current_link_start_nid, trace_agent.current_link_end_nid)]

        print('\n')
        # logger.info('current time {}'.format(t))
        # logger.info('vehicle {} is on link {}'.format(trace_agent.agent_id, current_link_id))

        # checked_links.append(current_link_id)

        # alternative links
        alternative_links = []
        checked_outgoing_links = []
        current_link_angle = network.links[current_link_id].out_angle
        for alt_link_id in network.nodes[trace_agent.current_link_end_nid].outgoing_links:
            alternative_link_angle = (current_link_angle - network.links[alt_link_id].in_angle) / 3.14 * 180
            if alternative_link_angle < -180: alternative_link_angle += 360
            if alternative_link_angle > 180: alternative_link_angle -= 360
            alternative_links.append([alt_link_id, network.links[alt_link_id].end_nid, alternative_link_angle])
        # print(alternative_links)
        alternative_links_by_direction = {'forward': None, 'left': None, 'right': None, 'back': None}
        # forward
        # (forward_link, angle) = min(alternative_links, key=lambda l: abs(l[1] - 0))
        potential_forward_link = [(forward_link, end_nid, angle) for (forward_link, end_nid, angle) in alternative_links if
                                  (abs(angle - 0) <= 45 and forward_link not in checked_outgoing_links)]
        if (len(potential_forward_link) > 0):
            alternative_links_by_direction['forward'] = min(potential_forward_link, key=lambda l: abs(l[2] - 0))
            checked_outgoing_links.append(alternative_links_by_direction['forward'])

        # left
        potential_left_link = [(left_link, end_nid, angle) for (left_link, end_nid, angle) in alternative_links if
                               (abs(angle - (-90)) <= 45 and (left_link not in checked_outgoing_links))]
        if (len(potential_left_link) > 0):
            alternative_links_by_direction['left'] = min(potential_left_link, key=lambda l: abs(l[2] - (-90)))
            checked_outgoing_links.append(alternative_links_by_direction['left'])

        # right
        potential_right_link = [(right_link, end_nid, angle) for (right_link, end_nid, angle) in alternative_links if
                                (abs(angle - 90) <= 45 and (right_link not in checked_outgoing_links))]
        # (right_link, angle) = min(alternative_links, key=lambda l: abs(l[1] - 90))
        if (len(potential_right_link) > 0):
            alternative_links_by_direction['right'] = min(potential_right_link, key=lambda l: abs(l[2] - 90))
            checked_outgoing_links.append(alternative_links_by_direction['right'])

        # backward
        potential_back_link = [(back_link, end_nid, angle) for (back_link, end_nid, angle) in alternative_links if
                               (abs(angle - 0) > 135 and (back_link not in checked_outgoing_links))]
        if len(potential_back_link) > 0:
            alternative_links_by_direction['back'] = max(potential_back_link, key=lambda l: abs(l[2]))
            checked_outgoing_links.append(alternative_links_by_direction['back'])

        print(alternative_links_by_direction)
        logger.info(alternative_links_by_direction)

        # agent_next_link_direction = gl.get_value("direction_message")
        agent_next_link_direction = direction_tmp
        valid_links_by_direction = [k for k, v in alternative_links_by_direction.items() if v is not None]

        logger.info("agent' s route: {}".format(network.agents[agent_id].route))
        if len(network.agents[agent_id].route)==0:
            logger.info("get destination")

        if agent_next_link_direction in valid_links_by_direction:
            agent_next_link = alternative_links_by_direction[agent_next_link_direction][0]#link_id
            logger.info("driver's self decision: {}".format(agent_next_link))
        else:
            agent_next_link = network.agents[agent_id].next_link #link_id
            logger.info("driver's system decision: {}".format(agent_next_link))
            # random.choice(valid_links_by_direction)

        logger.info("time: {}, current link is {}, next link is {}".format(t, current_link_id, agent_next_link))
        # print('  next link is {}'.format(agent_next_link))
        # logger.info('  next link is {}'.format(agent_next_link))

        # agent_next_link_direction = None
        # close all other directions
        for [link_id, _, angle_tmp] in alternative_links:
            if link_id != agent_next_link:
                # print('close {}'.format(link_id))
                link = network.links[link_id]
                network.g.update_edge(link.start_nid, link.end_nid, c_double(1e8))

        routing_status = network.agents[agent_id].get_path(t, g=network.g)
        network.agents[agent_id].find_next_link(node2link_dict=network.node2link_dict)
        # logger.info("agent' s route: {}".format(network.agents[agent_id].route))
        # print(agent_next_link, network.agents[0].next_link)

        for [link_id, _, angle_tmp] in alternative_links:
            if link_id != agent_next_link:
                link = network.links[link_id]
                network.g.update_edge(link.start_nid, link.end_nid, c_double(link.fft))

        network, status = calculate_position(network, config, t, agent_id, logger, isStop)


        data['network'] = network

        # extract vehicle positions
        veh_num, vehicle_positions, whole_vehs, isStop = extract_vehicle_location.extract_vehicle_locations(network, t, agent_id, logger)
        agent_link = network.node2link_dict[(trace_agent.current_link_start_nid, trace_agent.current_link_end_nid)]
        if type(agent_link)==str and "vl" in agent_link:
            image_path = ""
        else:
            driver_link_list = ast.literal_eval(links_list["long_points_list"].loc[links_list["eid"]==agent_link].values[0])
            driver_position = (whole_vehs["lon"].loc[whole_vehs["veh_id"]==agent_id].values[0],
                               whole_vehs["lat"].loc[whole_vehs["veh_id"]==agent_id].values[0])
            image_path_tmp = get_image_rpc(driver_link_list, driver_position)
            # image_path = os.path.join("link_"+str(whole_vehs["link_id"].loc[whole_vehs["veh_id"]==agent_id].values[0]), image_path_tmp)
            image_path = os.path.join(str(whole_vehs["link_id"].loc[whole_vehs["veh_id"]==agent_id].values[0]), image_path_tmp)

        # server.output = str(t)+": \n"+vehicle_positions

        # output.put(str(t)+":"+vehicle_positions)

        # logger.info("veh_num: " + str(veh_num))

        # global output
        # output = str(t) + ":" + vehicle_positions

        # print(str(t)+": \n"+vehicle_positions)
        # vehicle_positions.to_csv('test_veh_positions_t{}.csv'.format(t))

        # plot vehicle routing options


        # return data, config, update_data,checked_links, output


        # return data, config, update_data, output, old_link_id, image_position
        return data, config, update_data, whole_vehs, isStop, image_path
    else:
        return data, config, update_data, None, isStop, ""


def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                                           * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

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
    return (x1 + delta_x, y1 + delta_y), (x2 + delta_x, y2 + delta_y), angle


def extra_link_info(data):
    network = data['network']
    links_dict = network.links
    transformer = Transformer.from_crs( "epsg:26910", "epsg:4326")

    points = []
    for link_id, link in links_dict.items():

        ### skip virtual links
        if link.link_type == 'v':
            continue

        ### link stats

        link_geometry = link.geometry
        link_length = link.geometry.length
        link_num = (int)(link_length // 10)

        tmp_location_pre = link_geometry.interpolate(0, normalized=True)
        if link_num>=1:
            proportion = 1 / link_num
            for i in range(1, link_num):
                tmp_location_cur = link_geometry.interpolate(proportion*i, normalized = True)
                q1_veh_coord_offset, q2_veh_coord_offset, q_veh_dir = veh_offset(tmp_location_pre.x, tmp_location_pre.y, tmp_location_cur.x, tmp_location_cur.y)
                tmp_location_pre = tmp_location_cur
                lat,lon = transformer.transform(q1_veh_coord_offset[0], q1_veh_coord_offset[1])
                lat_2, lon_2 = transformer.transform(q2_veh_coord_offset[0], q2_veh_coord_offset[1])
                bearing = calculate_initial_compass_bearing((lat, lon), (lat_2, lon_2))
                points.append([link_id, q1_veh_coord_offset[0], q1_veh_coord_offset[1], lat, lon, bearing])

        tmp_location_cur = link_geometry.interpolate(1.0, normalized=True)
        q1_veh_coord_offset, q2_veh_coord_offset, q_veh_dir = veh_offset(tmp_location_pre.x, tmp_location_pre.y,
                                                                         tmp_location_cur.x, tmp_location_cur.y)

        lat, lon = transformer.transform(q1_veh_coord_offset[0], q1_veh_coord_offset[1])
        lat2, lon2 = transformer.transform(q2_veh_coord_offset[0], q2_veh_coord_offset[1])
        bearing = calculate_initial_compass_bearing((lat, lon), (lat2, lon2))

        points.append([link_id, q1_veh_coord_offset[0], q1_veh_coord_offset[1], lat, lon, bearing])
        points.append([link_id, q2_veh_coord_offset[0], q2_veh_coord_offset[1], lat2, lon2, bearing])

    points_df = pd.DataFrame(points, columns=['link_id', 'x', 'y', 'lat', 'lon', 'angle'])
    points_df.to_csv('../projects/bolinas/simulation_outputs/points_bearing.csv', index=False)
    print("extracting points complete")
    return points_df





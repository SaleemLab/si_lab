%reload_ext autoreload
%autoreload 2

import logging
import numpy as np

# Make analysis reproducible
np.random.seed(0)

# Enable logging
logging.basicConfig(level=logging.INFO)


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

from replay_trajectory_classification.simulate import simulate_neuron_with_place_field
from track_linearization import get_linearized_position
from replay_trajectory_classification import make_track_graph
import numpy as np


def generate_position(traversal_path, track_graph, step_size=0.020, n_traversals=5):
    points = []
    for _ in range(n_traversals):
        for node1, node2 in traversal_path:
            x1, y1 = track_graph.nodes[node1]['pos']
            x2, y2 = track_graph.nodes[node2]['pos']
            dx, dy = x2 - x1, y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            n_points = int(dist // step_size)
            w = np.linspace(0, 1, n_points)
            points.append((x1 + dx * w, y1 + dy * w))

    return np.concatenate(points, axis=1).T

def get_unique_place_field_centers(place_field_centers):
    return np.unique((place_field_centers * 10_000).astype(int), axis=0) / 10_000

def make_two_environment_data():

    dges1 = [(0, 1), (1, 2), (2, 3)]
        node_positions1 = [
                       (50, 40),
                       (50, 80),
                       (50, 120),
                       (50, 160),
                       ]
        
    edge_order1 = ((0, 1), (1, 2), (2, 3))
    edge_spacing1 = 0
    track_graph1 = make_track_graph(node_positions2, edges2)
    
    traversal_path2 = [(0, 1), (1, 2), (2, 3),
                       (3, 2), (2, 1), (1, 0)]
    position2 = generate_position(traversal_path2, track_graph2)
    position_df2 = get_linearized_position(position2,
                                          track_graph2,
                                          edge_order=edge_order2,
                                          edge_spacing=edge_spacing2,
                                          use_HMM=False)
    
    sampling_frequency = 1000

    place_field_centers1 = generate_position(traversal_path1, track_graph1, step_size=10, n_traversals=1)
    place_field_centers1 = get_unique_place_field_centers(place_field_centers1)
    
    spikes1 = np.stack([simulate_neuron_with_place_field(center, position1,
                                                        sampling_frequency=sampling_frequency,
                                                        variance=6.0**2)
                       for center in place_field_centers1], axis=1)
    


    edges2 = [(0, 1), (1, 2), (2, 3)]
    node_positions2 = [
                   (0, 40),
                   (0, 80),
                   (0, 120),
                   (0, 160),
                   ]
    
    edge_order2 = ((0, 1), (1, 2), (2, 3))
    edge_spacing2 = 0
    track_graph2 = make_track_graph(node_positions2, edges2)
    
    traversal_path2 = [(0, 1), (1, 2), (2, 3),
                       (3, 2), (2, 1), (1, 0)]
    position2 = generate_position(traversal_path2, track_graph2)
    position_df2 = get_linearized_position(position2,
                                          track_graph2,
                                          edge_order=edge_order2,
                                          edge_spacing=edge_spacing2,
                                          use_HMM=False)
    
    place_field_centers2 = generate_position(traversal_path2, track_graph2, step_size=10, n_traversals=1)
    place_field_centers2 = get_unique_place_field_centers(place_field_centers2)
    
    spikes2_temp = np.stack([simulate_neuron_with_place_field(center, position2,
                                                    sampling_frequency=sampling_frequency,
                                                    variance=6.0**2)
                   for center in place_field_centers2], axis=1)
    spikes2 = np.zeros((spikes2_temp.shape[0], spikes1.shape[1]))
    spikes2[:, [7, 17, 27, 4, 14, 24, 10, 1, 12, 23]] = spikes2_temp
    
    return (spikes1, spikes2,
            position_df1, position_df2,
            track_graph1, track_graph2, 
            place_field_centers1, place_field_centers2,
            position1, position2,
            edge_order1, edge_spacing1,
            edge_order2, edge_spacing2
           )
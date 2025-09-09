"""
Padding-based defense implementation
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

class PaddingDefense:
    """Implement padding-based traffic analysis defense"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.defense_config = config['defense']['padding']
    
    def pad_packet_sizes(self, sizes, target_size=None):
        """Pad packet sizes to target size or fixed bins"""
        if target_size is None:
            target_size = self.defense_config['target_size']
        
        padded_sizes = []
        for size in sizes:
            if size < target_size:
                padded_sizes.append(target_size)
            else:
                # Round up to next multiple of target_size
                padded_size = int(np.ceil(size / target_size) * target_size)
                padded_sizes.append(padded_size)
        
        return padded_sizes
    
    def add_dummy_packets(self, flow_features, dummy_rate=0.1):
        """Add dummy packets to the flow"""
        total_packets = int(flow_features.get('total_packets', 0))  # Convert to int
        
        if total_packets == 0:
            return flow_features
        
        # Calculate number of dummy packets to add
        num_dummies = max(0, int(total_packets * dummy_rate))
        
        # Update packet counts
        modified_features = flow_features.copy()
        modified_features['total_packets'] = total_packets + num_dummies
        
        # Randomly distribute dummy packets between incoming/outgoing
        if num_dummies > 0:
            dummy_out = np.random.binomial(num_dummies, 0.5)
            dummy_in = num_dummies - dummy_out
        else:
            dummy_out = dummy_in = 0
        
        modified_features['outgoing_packets'] = int(flow_features.get('outgoing_packets', 0)) + dummy_out
        modified_features['incoming_packets'] = int(flow_features.get('incoming_packets', 0)) + dummy_in
        
        # Adjust other features
        dummy_size = self.defense_config['target_size']
        original_bytes = float(flow_features.get('total_bytes', 0))
        modified_features['total_bytes'] = original_bytes + (num_dummies * dummy_size)
        
        # Recalculate mean packet size
        total_packets_new = modified_features['total_packets']
        if total_packets_new > 0:
            modified_features['mean_packet_size'] = modified_features['total_bytes'] / total_packets_new
        
        return modified_features
    
    def apply_constant_padding(self, flow_features):
        """Apply constant rate padding defense"""
        # Pad all packets to target size
        target_size = self.defense_config['target_size']
        
        modified_features = flow_features.copy()
        
        # Update size-based features
        total_packets = int(flow_features.get('total_packets', 0))
        if total_packets > 0:
            modified_features['total_bytes'] = total_packets * target_size
            modified_features['mean_packet_size'] = target_size
            modified_features['std_packet_size'] = 0  # No variation after padding
            modified_features['min_packet_size'] = target_size
            modified_features['max_packet_size'] = target_size
            
            # Update percentiles
            for p in [25, 50, 75, 90, 95]:
                modified_features[f'size_p{p}'] = target_size
            
            # Update first packet sizes
            for i in range(10):
                if f'first_pkt_size_{i}' in modified_features:
                    modified_features[f'first_pkt_size_{i}'] = target_size
        
        return modified_features
    
    def apply_random_padding(self, flow_features):
        """Apply random padding defense"""
        target_size = self.defense_config['target_size']
        
        modified_features = flow_features.copy()
        
        # Add random variation to packet sizes
        total_packets = int(flow_features.get('total_packets', 0))  # Convert to int
        
        if total_packets > 0:
            # Generate random sizes around target size
            random_sizes = np.random.normal(target_size, target_size * 0.1, size=total_packets)  # Use size parameter
            random_sizes = np.clip(random_sizes, target_size * 0.5, target_size * 2)
            
            # Update features with random sizes
            modified_features['total_bytes'] = float(np.sum(random_sizes))
            modified_features['mean_packet_size'] = float(np.mean(random_sizes))
            modified_features['std_packet_size'] = float(np.std(random_sizes))
            modified_features['min_packet_size'] = float(np.min(random_sizes))
            modified_features['max_packet_size'] = float(np.max(random_sizes))
            
            # Update percentiles
            for p in [25, 50, 75, 90, 95]:
                modified_features[f'size_p{p}'] = float(np.percentile(random_sizes, p))
        
        return modified_features
    
    def apply_defense(self, X):
        """Apply padding defense to feature matrix"""
        if not self.defense_config['enabled']:
            return X
        
        self.logger.info("Applying padding defense")
        
        X_defended = X.copy()
        
        for idx in range(len(X_defended)):
            row = X_defended.iloc[idx].to_dict()
            
            if self.defense_config['random_padding']:
                defended_row = self.apply_random_padding(row)
            else:
                defended_row = self.apply_constant_padding(row)
            
            # Add dummy packets
            defended_row = self.add_dummy_packets(defended_row)
            
            # Update the dataframe
            for key, value in defended_row.items():
                if key in X_defended.columns:
                    X_defended.iloc[idx, X_defended.columns.get_loc(key)] = value
        
        self.logger.info(f"Padding defense applied to {len(X_defended)} samples")
        
        return X_defended

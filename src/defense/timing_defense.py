"""
Timing-based defense implementation
"""

import numpy as np
import pandas as pd
import logging

class TimingDefense:
    """Implement timing-based traffic analysis defense"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.defense_config = config['defense']['timing']
    
    def add_timing_jitter(self, flow_features):
        """Add random jitter to timing features"""
        jitter_min, jitter_max = self.defense_config['jitter_range']
        
        modified_features = flow_features.copy()
        
        # Add jitter to inter-arrival time features
        if 'mean_iat' in modified_features and modified_features['mean_iat'] is not None:
            jitter = np.random.uniform(jitter_min, jitter_max) / 1000  # Convert ms to seconds
            current_iat = float(modified_features['mean_iat'])
            modified_features['mean_iat'] = max(0, current_iat + jitter)
        
        if 'std_iat' in modified_features and modified_features['std_iat'] is not None:
            # Increase variation due to jitter
            jitter_std = (jitter_max - jitter_min) / 1000 / 4  # Quarter of range as additional std
            current_std = float(modified_features['std_iat'])
            modified_features['std_iat'] = current_std + jitter_std
        
        # Modify flow duration (slightly increase due to jitter)
        if 'flow_duration' in modified_features and modified_features['flow_duration'] is not None:
            duration_jitter = np.random.uniform(0, jitter_max) / 1000
            current_duration = float(modified_features['flow_duration'])
            modified_features['flow_duration'] = current_duration + duration_jitter
        
        return modified_features
    
    def apply_constant_rate(self, flow_features):
        """Apply constant rate transmission"""
        rate_ms = self.defense_config['rate_ms']
        
        modified_features = flow_features.copy()
        
        # Recalculate timing features assuming constant rate
        total_packets = int(flow_features.get('total_packets', 0))
        
        if total_packets > 1:
            # Inter-arrival time becomes constant
            constant_iat = rate_ms / 1000  # Convert to seconds
            modified_features['mean_iat'] = constant_iat
            modified_features['std_iat'] = 0.0  # No variation in constant rate
            
            # Flow duration becomes predictable
            modified_features['flow_duration'] = (total_packets - 1) * constant_iat
        
        return modified_features
    
    def apply_burst_shaping(self, flow_features, burst_size=5, burst_interval=100):
        """Apply burst-based traffic shaping"""
        modified_features = flow_features.copy()
        
        total_packets = int(flow_features.get('total_packets', 0))
        
        if total_packets > 0:
            # Calculate new burst features
            num_bursts = max(1, int(np.ceil(total_packets / burst_size)))
            
            modified_features['num_bursts'] = num_bursts
            modified_features['avg_burst_size'] = min(burst_size, total_packets / num_bursts)
            modified_features['max_burst_size'] = min(burst_size, total_packets)
            
            # Recalculate timing based on burst intervals
            burst_interval_sec = burst_interval / 1000
            modified_features['flow_duration'] = num_bursts * burst_interval_sec
            
            # Inter-arrival time within bursts vs between bursts
            if num_bursts > 1:
                modified_features['mean_iat'] = burst_interval_sec / burst_size
        
        return modified_features
    
    def apply_defense(self, X):
        """Apply timing defense to feature matrix"""
        if not self.defense_config['enabled']:
            return X
        
        self.logger.info("Applying timing defense")
        
        X_defended = X.copy()
        
        for idx in range(len(X_defended)):
            row = X_defended.iloc[idx].to_dict()
            
            # Apply jitter
            defended_row = self.add_timing_jitter(row)
            
            # Apply constant rate if enabled
            if self.defense_config['constant_rate']:
                defended_row = self.apply_constant_rate(defended_row)
            else:
                # Apply burst shaping
                defended_row = self.apply_burst_shaping(defended_row)
            
            # Update the dataframe
            for key, value in defended_row.items():
                if key in X_defended.columns:
                    X_defended.iloc[idx, X_defended.columns.get_loc(key)] = value
        
        self.logger.info(f"Timing defense applied to {len(X_defended)} samples")
        
        return X_defended

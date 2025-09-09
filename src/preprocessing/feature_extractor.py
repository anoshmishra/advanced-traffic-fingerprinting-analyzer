"""
Feature extraction from PCAP files
"""

import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, TCP
import logging
from pathlib import Path
from collections import defaultdict
import hashlib
import json

class FeatureExtractor:
    """Extract features from captured traffic"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def parse_pcap(self, pcap_file):
        """Parse PCAP file and extract packet information"""
        try:
            packets = rdpcap(str(pcap_file))
            flow_data = []
            
            for pkt in packets:
                if IP in pkt and TCP in pkt:
                    pkt_info = {
                        'timestamp': float(pkt.time),
                        'size': len(pkt),
                        'src_ip': pkt[IP].src,
                        'dst_ip': pkt[IP].dst,
                        'src_port': pkt[TCP].sport,
                        'dst_port': pkt[TCP].dport,
                        'flags': pkt[TCP].flags
                    }
                    flow_data.append(pkt_info)
            
            return flow_data
        
        except Exception as e:
            self.logger.error(f"Error parsing {pcap_file}: {e}")
            return []
    
    def identify_flows(self, packet_data, client_ip=None):
        """Group packets into flows"""
        flows = defaultdict(list)
        
        for pkt in packet_data:
            # Create flow key (5-tuple)
            if client_ip and pkt['src_ip'] == client_ip:
                direction = 1  # Outgoing
                flow_key = (pkt['src_ip'], pkt['dst_ip'], pkt['src_port'], pkt['dst_port'])
            else:
                direction = -1  # Incoming
                flow_key = (pkt['dst_ip'], pkt['src_ip'], pkt['dst_port'], pkt['src_port'])
            
            flows[flow_key].append({
                'timestamp': pkt['timestamp'],
                'size': pkt['size'],
                'direction': direction
            })
        
        return flows
    
    def extract_statistical_features(self, flow):
        """Extract statistical features from a flow"""
        if not flow:
            return {}
        
        # Basic statistics
        sizes = [pkt['size'] for pkt in flow]
        timestamps = [pkt['timestamp'] for pkt in flow]
        directions = [pkt['direction'] for pkt in flow]
        
        # Sort by timestamp
        flow_sorted = sorted(flow, key=lambda x: x['timestamp'])
        
        features = {
            # Packet count features
            'total_packets': len(flow),
            'outgoing_packets': sum(1 for d in directions if d == 1),
            'incoming_packets': sum(1 for d in directions if d == -1),
            
            # Size features
            'total_bytes': sum(sizes),
            'mean_packet_size': np.mean(sizes),
            'std_packet_size': np.std(sizes),
            'min_packet_size': min(sizes),
            'max_packet_size': max(sizes),
            
            # Timing features
            'flow_duration': max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0,
            'mean_iat': np.mean(np.diff(sorted(timestamps))) if len(timestamps) > 1 else 0,
            'std_iat': np.std(np.diff(sorted(timestamps))) if len(timestamps) > 1 else 0,
            
            # Direction features
            'direction_changes': sum(1 for i in range(1, len(directions)) 
                                   if directions[i] != directions[i-1]),
        }
        
        # Packet size percentiles
        for p in [25, 50, 75, 90, 95]:
            features[f'size_p{p}'] = np.percentile(sizes, p)
        
        # First N packet sizes (for sequence analysis)
        for i in range(min(10, len(sizes))):
            features[f'first_pkt_size_{i}'] = sizes[i]
            features[f'first_pkt_dir_{i}'] = directions[i]
        
        return features
    
    def extract_burst_features(self, flow, burst_threshold=0.5):
        """Extract burst-related features"""
        if len(flow) < 2:
            return {}
        
        # Sort by timestamp
        flow_sorted = sorted(flow, key=lambda x: x['timestamp'])
        
        # Identify bursts (groups of packets close in time)
        bursts = []
        current_burst = [flow_sorted[0]]
        
        for i in range(1, len(flow_sorted)):
            time_diff = flow_sorted[i]['timestamp'] - flow_sorted[i-1]['timestamp']
            
            if time_diff <= burst_threshold:
                current_burst.append(flow_sorted[i])
            else:
                if len(current_burst) > 1:
                    bursts.append(current_burst)
                current_burst = [flow_sorted[i]]
        
        if len(current_burst) > 1:
            bursts.append(current_burst)
        
        # Extract burst features
        features = {
            'num_bursts': len(bursts),
            'avg_burst_size': np.mean([len(b) for b in bursts]) if bursts else 0,
            'max_burst_size': max([len(b) for b in bursts]) if bursts else 0,
            'burst_bytes_ratio': sum([sum(pkt['size'] for pkt in b) for b in bursts]) / 
                               sum(pkt['size'] for pkt in flow_sorted) if bursts else 0
        }
        
        return features
    
    def extract_sequence_features(self, flow, max_length=100):
        """Extract sequence-based features for deep learning"""
        if not flow:
            return {}
        
        # Sort by timestamp
        flow_sorted = sorted(flow, key=lambda x: x['timestamp'])
        
        # Create sequences
        size_sequence = [pkt['size'] for pkt in flow_sorted[:max_length]]
        direction_sequence = [pkt['direction'] for pkt in flow_sorted[:max_length]]
        
        # Pad sequences if needed
        while len(size_sequence) < max_length:
            size_sequence.append(0)
            direction_sequence.append(0)
        
        features = {
            'size_sequence': size_sequence,
            'direction_sequence': direction_sequence,
            'sequence_length': min(len(flow_sorted), max_length)
        }
        
        return features
    
    def extract_features_from_pcap(self, pcap_file, label_file):
        """Extract all features from a single PCAP file"""
        # Parse packets
        packet_data = self.parse_pcap(pcap_file)
        
        if not packet_data:
            return None
        
        # Read label
        try:
            with open(label_file, 'r') as f:
                label = f.read().strip()
        except:
            label = "unknown"
        
        # Group into flows
        flows = self.identify_flows(packet_data)
        
        # For website fingerprinting, we typically use the main flow
        # (largest flow or first flow to port 443)
        main_flow = None
        if flows:
            # Find HTTPS flow (port 443) or largest flow
            https_flows = [flow for key, flow in flows.items() if 443 in key]
            if https_flows:
                main_flow = max(https_flows, key=len)
            else:
                main_flow = max(flows.values(), key=len)
        
        if not main_flow:
            return None
        
        # Extract features
        features = {}
        features.update(self.extract_statistical_features(main_flow))
        features.update(self.extract_burst_features(main_flow))
        features.update(self.extract_sequence_features(main_flow))
        
        # Add metadata
        features['label'] = label
        features['pcap_file'] = str(pcap_file)
        
        return features
    
    def extract_features(self):
        """Extract features from all PCAP files"""
        pcap_dir = Path(self.config['paths']['data']) / 'raw_pcaps'
        
        if not pcap_dir.exists():
            self.logger.error(f"PCAP directory not found: {pcap_dir}")
            return None, None
        
        feature_list = []
        pcap_files = list(pcap_dir.glob('*.pcap'))
        
        self.logger.info(f"Processing {len(pcap_files)} PCAP files")
        
        for pcap_file in pcap_files:
            label_file = pcap_file.with_suffix('.label')
            
            if not label_file.exists():
                self.logger.warning(f"No label file for {pcap_file}")
                continue
            
            features = self.extract_features_from_pcap(pcap_file, label_file)
            
            if features:
                feature_list.append(features)
        
        if not feature_list:
            self.logger.error("No features extracted")
            return None, None
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_list)
        
        # Separate features and labels
        label_col = 'label'
        sequence_cols = ['size_sequence', 'direction_sequence']
        metadata_cols = ['pcap_file']
        
        # Get feature columns (exclude label, sequences, and metadata)
        feature_cols = [col for col in df.columns 
                       if col not in [label_col] + sequence_cols + metadata_cols]
        
        X = df[feature_cols].fillna(0)
        y = df[label_col]
        
        # Normalize features if specified
        if self.config['features']['normalize']:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Save processed features
        processed_dir = Path(self.config['paths']['data']) / 'processed'
        processed_dir.mkdir(exist_ok=True)
        
        X.to_csv(processed_dir / 'features.csv', index=False)
        y.to_csv(processed_dir / 'labels.csv', index=False)
        
        self.logger.info(f"Extracted {X.shape[1]} features from {X.shape[0]} samples")
        self.logger.info(f"Unique labels: {y.nunique()}")
        
        return X, y

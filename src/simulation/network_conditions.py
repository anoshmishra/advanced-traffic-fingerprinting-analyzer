"""
Network conditions simulation for realistic traffic analysis
"""

import numpy as np
import random
import time
from collections import deque
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scapy.all import Packet

logger = logging.getLogger(__name__)

@dataclass
class NetworkConditions:
    """Network condition parameters"""
    latency_ms: float = 0.0          # Base latency in milliseconds
    jitter_ms: float = 0.0           # Jitter range in milliseconds  
    packet_loss_rate: float = 0.0    # Packet loss rate (0.0 to 1.0)
    bandwidth_mbps: float = 100.0    # Available bandwidth in Mbps
    duplicate_rate: float = 0.0      # Packet duplication rate
    reorder_probability: float = 0.0  # Packet reordering probability
    corruption_rate: float = 0.0     # Packet corruption rate
    congestion_window: int = 64       # TCP congestion window size

class NetworkConditionSimulator:
    """Simulates various network conditions on traffic flows"""
    
    def __init__(self, conditions: NetworkConditions = None, seed: int = None):
        self.conditions = conditions or NetworkConditions()
        self.random = random.Random(seed)
        self.np_random = np.random.RandomState(seed)
        self.packet_queue = deque()
        self.bandwidth_tokens = self.conditions.bandwidth_mbps * 1024 * 1024 / 8  # Convert to bytes per second
        self.last_token_update = time.time()
        
    def simulate_latency(self, packets: List[Packet]) -> List[Packet]:
        """Apply latency and jitter to packets"""
        modified_packets = []
        
        for packet in packets:
            # Base latency
            delay = self.conditions.latency_ms / 1000.0
            
            # Add jitter (normally distributed)
            if self.conditions.jitter_ms > 0:
                jitter = self.np_random.normal(0, self.conditions.jitter_ms / 1000.0)
                delay += abs(jitter)  # Ensure delay is positive
            
            # Apply delay to packet timestamp
            packet.time += delay
            modified_packets.append(packet)
        
        return modified_packets
    
    def simulate_packet_loss(self, packets: List[Packet]) -> List[Packet]:
        """Simulate packet loss"""
        if self.conditions.packet_loss_rate <= 0:
            return packets
        
        surviving_packets = []
        
        for packet in packets:
            if self.random.random() > self.conditions.packet_loss_rate:
                surviving_packets.append(packet)
            else:
                logger.debug(f"Dropped packet due to loss simulation")
        
        return surviving_packets
    
    def simulate_bandwidth_limitation(self, packets: List[Packet]) -> List[Packet]:
        """Simulate bandwidth limitations using token bucket algorithm"""
        if self.conditions.bandwidth_mbps <= 0:
            return packets
        
        current_time = time.time()
        time_elapsed = current_time - self.last_token_update
        
        # Replenish tokens
        tokens_to_add = time_elapsed * self.conditions.bandwidth_mbps * 1024 * 1024 / 8
        self.bandwidth_tokens = min(
            self.bandwidth_tokens + tokens_to_add,
            self.conditions.bandwidth_mbps * 1024 * 1024 / 8  # Max bucket size
        )
        self.last_token_update = current_time
        
        throttled_packets = []
        additional_delay = 0.0
        
        for packet in packets:
            packet_size = len(packet)
            
            if self.bandwidth_tokens >= packet_size:
                # Sufficient tokens, send packet
                self.bandwidth_tokens -= packet_size
                packet.time += additional_delay
                throttled_packets.append(packet)
            else:
                # Insufficient tokens, calculate delay
                tokens_needed = packet_size - self.bandwidth_tokens
                delay_needed = tokens_needed / (self.conditions.bandwidth_mbps * 1024 * 1024 / 8)
                
                additional_delay += delay_needed
                packet.time += additional_delay
                
                self.bandwidth_tokens = 0  # Consume all tokens
                throttled_packets.append(packet)
        
        return throttled_packets
    
    def simulate_packet_reordering(self, packets: List[Packet]) -> List[Packet]:
        """Simulate packet reordering"""
        if self.conditions.reorder_probability <= 0 or len(packets) < 2:
            return packets
        
        reordered_packets = packets.copy()
        
        for i in range(len(reordered_packets) - 1):
            if self.random.random() < self.conditions.reorder_probability:
                # Swap with next packet
                reordered_packets[i], reordered_packets[i + 1] = \
                    reordered_packets[i + 1], reordered_packets[i]
        
        return reordered_packets
    
    def simulate_packet_duplication(self, packets: List[Packet]) -> List[Packet]:
        """Simulate packet duplication"""
        if self.conditions.duplicate_rate <= 0:
            return packets
        
        duplicated_packets = []
        
        for packet in packets:
            duplicated_packets.append(packet)
            
            if self.random.random() < self.conditions.duplicate_rate:
                # Create duplicate with slight delay
                duplicate = packet.copy()
                duplicate.time += 0.001  # 1ms delay for duplicate
                duplicated_packets.append(duplicate)
                logger.debug("Duplicated packet")
        
        return duplicated_packets
    
    def simulate_packet_corruption(self, packets: List[Packet]) -> List[Packet]:
        """Simulate packet corruption (simplified as packet loss)"""
        if self.conditions.corruption_rate <= 0:
            return packets
        
        uncorrupted_packets = []
        
        for packet in packets:
            if self.random.random() > self.conditions.corruption_rate:
                uncorrupted_packets.append(packet)
            else:
                logger.debug("Packet corrupted (dropped)")
        
        return uncorrupted_packets
    
    def apply_all_conditions(self, packets: List[Packet]) -> List[Packet]:
        """Apply all network conditions to packet list"""
        logger.info(f"Applying network conditions to {len(packets)} packets")
        
        # Apply conditions in realistic order
        result_packets = packets
        
        # 1. Bandwidth limitation (affects timing)
        result_packets = self.simulate_bandwidth_limitation(result_packets)
        
        # 2. Latency and jitter
        result_packets = self.simulate_latency(result_packets)
        
        # 3. Packet reordering
        result_packets = self.simulate_packet_reordering(result_packets)
        
        # 4. Packet duplication
        result_packets = self.simulate_packet_duplication(result_packets)
        
        # 5. Packet loss and corruption
        result_packets = self.simulate_packet_loss(result_packets)
        result_packets = self.simulate_packet_corruption(result_packets)
        
        # Sort by timestamp after all modifications
        result_packets.sort(key=lambda p: p.time)
        
        logger.info(f"Network simulation complete: {len(result_packets)} packets remaining")
        return result_packets

class NetworkProfileSimulator:
    """Simulates different network profiles (WiFi, 4G, 5G, etc.)"""
    
    NETWORK_PROFILES = {
        'ethernet_1gbps': NetworkConditions(
            latency_ms=1.0,
            jitter_ms=0.5,
            packet_loss_rate=0.0001,
            bandwidth_mbps=1000.0
        ),
        'ethernet_100mbps': NetworkConditions(
            latency_ms=2.0,
            jitter_ms=1.0,
            packet_loss_rate=0.0005,
            bandwidth_mbps=100.0
        ),
        'wifi_good': NetworkConditions(
            latency_ms=5.0,
            jitter_ms=3.0,
            packet_loss_rate=0.001,
            bandwidth_mbps=50.0,
            reorder_probability=0.001
        ),
        'wifi_poor': NetworkConditions(
            latency_ms=15.0,
            jitter_ms=10.0,
            packet_loss_rate=0.02,
            bandwidth_mbps=10.0,
            reorder_probability=0.005
        ),
        '4g_good': NetworkConditions(
            latency_ms=30.0,
            jitter_ms=15.0,
            packet_loss_rate=0.005,
            bandwidth_mbps=20.0,
            reorder_probability=0.002
        ),
        '4g_poor': NetworkConditions(
            latency_ms=100.0,
            jitter_ms=50.0,
            packet_loss_rate=0.05,
            bandwidth_mbps=2.0,
            reorder_probability=0.01
        ),
        '5g_good': NetworkConditions(
            latency_ms=10.0,
            jitter_ms=5.0,
            packet_loss_rate=0.001,
            bandwidth_mbps=100.0,
            reorder_probability=0.0005
        ),
        '3g': NetworkConditions(
            latency_ms=200.0,
            jitter_ms=100.0,
            packet_loss_rate=0.1,
            bandwidth_mbps=1.0,
            reorder_probability=0.02
        ),
        'satellite': NetworkConditions(
            latency_ms=600.0,
            jitter_ms=50.0,
            packet_loss_rate=0.01,
            bandwidth_mbps=25.0,
            reorder_probability=0.003
        ),
        'dialup': NetworkConditions(
            latency_ms=100.0,
            jitter_ms=20.0,
            packet_loss_rate=0.02,
            bandwidth_mbps=0.056,  # 56k modem
            reorder_probability=0.005
        )
    }
    
    @classmethod
    def get_profile(cls, profile_name: str) -> NetworkConditions:
        """Get predefined network profile"""
        if profile_name not in cls.NETWORK_PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(cls.NETWORK_PROFILES.keys())}")
        
        return cls.NETWORK_PROFILES[profile_name]
    
    @classmethod
    def list_profiles(cls) -> List[str]:
        """List available network profiles"""
        return list(cls.NETWORK_PROFILES.keys())
    
    @classmethod
    def simulate_profile(cls, packets: List[Packet], profile_name: str, seed: int = None) -> List[Packet]:
        """Simulate packets under specific network profile"""
        conditions = cls.get_profile(profile_name)
        simulator = NetworkConditionSimulator(conditions, seed)
        return simulator.apply_all_conditions(packets)

def create_custom_conditions(**kwargs) -> NetworkConditions:
    """Create custom network conditions"""
    return NetworkConditions(**kwargs)

def compare_network_conditions(packets: List[Packet], profiles: List[str]) -> dict:
    """Compare packet flows under different network conditions"""
    results = {}
    
    for profile_name in profiles:
        logger.info(f"Simulating {profile_name} network conditions")
        
        # Create fresh copy of packets
        packet_copy = [p.copy() for p in packets]
        
        # Apply network conditions
        modified_packets = NetworkProfileSimulator.simulate_profile(packet_copy, profile_name)
        
        # Calculate metrics
        original_count = len(packets)
        modified_count = len(modified_packets)
        
        if packets and modified_packets:
            original_duration = max(p.time for p in packets) - min(p.time for p in packets)
            modified_duration = max(p.time for p in modified_packets) - min(p.time for p in modified_packets)
            
            results[profile_name] = {
                'original_packet_count': original_count,
                'modified_packet_count': modified_count,
                'packet_loss_rate': 1 - (modified_count / original_count),
                'original_duration': original_duration,
                'modified_duration': modified_duration,
                'duration_increase': modified_duration - original_duration,
                'packets': modified_packets
            }
        else:
            results[profile_name] = {
                'original_packet_count': original_count,
                'modified_packet_count': modified_count,
                'packet_loss_rate': 1.0 if original_count > 0 else 0.0,
                'packets': modified_packets
            }
    
    return results

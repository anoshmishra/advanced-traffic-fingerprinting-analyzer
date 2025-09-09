"""
Traffic morphing and transformation utilities
"""

import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging
from scapy.all import Packet, Raw

logger = logging.getLogger(__name__)

@dataclass
class TrafficProfile:
    """Traffic profile characteristics"""
    avg_packet_size: float
    packet_size_std: float
    avg_inter_arrival_time: float
    iat_std: float
    packet_count_distribution: Dict[str, float]  # e.g., {'small': 0.3, 'medium': 0.5, 'large': 0.2}
    burst_characteristics: Dict[str, float]  # burst patterns
    directional_ratio: float  # ratio of outgoing to incoming packets

class TrafficMorpher:
    """Advanced traffic morphing for website fingerprinting defense"""
    
    def __init__(self, seed: int = None):
        self.random = random.Random(seed)
        self.np_random = np.random.RandomState(seed)
        
    def create_profile_from_packets(self, packets: List[Packet]) -> TrafficProfile:
        """Extract traffic profile from packet list"""
        if not packets:
            return self._get_default_profile()
        
        # Calculate packet sizes
        sizes = [len(pkt) for pkt in packets]
        avg_size = np.mean(sizes)
        size_std = np.std(sizes)
        
        # Calculate inter-arrival times
        if len(packets) > 1:
            times = [pkt.time for pkt in packets]
            iats = np.diff(times)
            avg_iat = np.mean(iats)
            iat_std = np.std(iats)
        else:
            avg_iat = 0.1
            iat_std = 0.05
        
        # Analyze packet size distribution
        size_dist = self._analyze_size_distribution(sizes)
        
        # Analyze burst characteristics
        burst_chars = self._analyze_burst_patterns(packets)
        
        # Calculate directional ratio (simplified)
        directional_ratio = 0.4  # Default assumption
        
        return TrafficProfile(
            avg_packet_size=avg_size,
            packet_size_std=size_std,
            avg_inter_arrival_time=avg_iat,
            iat_std=iat_std,
            packet_count_distribution=size_dist,
            burst_characteristics=burst_chars,
            directional_ratio=directional_ratio
        )
    
    def _analyze_size_distribution(self, sizes: List[int]) -> Dict[str, float]:
        """Analyze packet size distribution"""
        if not sizes:
            return {'small': 0.33, 'medium': 0.33, 'large': 0.34}
        
        small_count = sum(1 for s in sizes if s < 200)
        medium_count = sum(1 for s in sizes if 200 <= s < 800)
        large_count = sum(1 for s in sizes if s >= 800)
        
        total = len(sizes)
        return {
            'small': small_count / total,
            'medium': medium_count / total,
            'large': large_count / total
        }
    
    def _analyze_burst_patterns(self, packets: List[Packet]) -> Dict[str, float]:
        """Analyze burst patterns in traffic"""
        if len(packets) < 2:
            return {'avg_burst_size': 1.0, 'burst_frequency': 0.1}
        
        # Simple burst detection based on inter-arrival times
        times = [pkt.time for pkt in packets]
        iats = np.diff(times)
        
        # Consider packets with IAT < 0.1s as part of a burst
        burst_threshold = 0.1
        bursts = []
        current_burst_size = 1
        
        for iat in iats:
            if iat < burst_threshold:
                current_burst_size += 1
            else:
                if current_burst_size > 1:
                    bursts.append(current_burst_size)
                current_burst_size = 1
        
        if current_burst_size > 1:
            bursts.append(current_burst_size)
        
        avg_burst_size = np.mean(bursts) if bursts else 1.0
        burst_frequency = len(bursts) / len(packets)
        
        return {
            'avg_burst_size': avg_burst_size,
            'burst_frequency': burst_frequency
        }
    
    def _get_default_profile(self) -> TrafficProfile:
        """Get default traffic profile"""
        return TrafficProfile(
            avg_packet_size=500.0,
            packet_size_std=200.0,
            avg_inter_arrival_time=0.05,
            iat_std=0.02,
            packet_count_distribution={'small': 0.3, 'medium': 0.5, 'large': 0.2},
            burst_characteristics={'avg_burst_size': 3.0, 'burst_frequency': 0.2},
            directional_ratio=0.4
        )
    
    def morph_to_profile(self, packets: List[Packet], target_profile: TrafficProfile, 
                        strength: float = 1.0) -> List[Packet]:
        """Morph traffic to match target profile"""
        if not packets:
            return packets
        
        morphed_packets = []
        
        # Step 1: Adjust packet sizes
        morphed_packets = self._adjust_packet_sizes(packets, target_profile, strength)
        
        # Step 2: Adjust timing
        morphed_packets = self._adjust_timing(morphed_packets, target_profile, strength)
        
        # Step 3: Adjust packet count if necessary
        morphed_packets = self._adjust_packet_count(morphed_packets, target_profile, strength)
        
        # Step 4: Add burst patterns
        morphed_packets = self._apply_burst_patterns(morphed_packets, target_profile, strength)
        
        logger.info(f"Morphed {len(packets)} packets to target profile (strength={strength})")
        return morphed_packets
    
    def _adjust_packet_sizes(self, packets: List[Packet], profile: TrafficProfile, 
                           strength: float) -> List[Packet]:
        """Adjust packet sizes towards target profile"""
        morphed_packets = []
        
        for packet in packets:
            current_size = len(packet)
            
            # Generate target size based on profile
            target_size = max(40, int(self.np_random.normal(
                profile.avg_packet_size, 
                profile.packet_size_std
            )))
            
            # Interpolate between current and target size
            new_size = int(current_size + strength * (target_size - current_size))
            new_size = max(40, min(1500, new_size))  # Ensure reasonable bounds
            
            # Create new packet with adjusted size
            new_packet = packet.copy()
            if new_size > len(packet):
                # Add padding
                padding_size = new_size - len(packet)
                if Raw in new_packet:
                    new_packet[Raw].load += b'\x00' * padding_size
                else:
                    new_packet = new_packet / Raw(b'\x00' * padding_size)
            elif new_size < len(packet):
                # Truncate (simplified - just note the target size)
                # In real implementation, would need more sophisticated packet modification
                pass
            
            morphed_packets.append(new_packet)
        
        return morphed_packets
    
    def _adjust_timing(self, packets: List[Packet], profile: TrafficProfile, 
                      strength: float) -> List[Packet]:
        """Adjust packet timing towards target profile"""
        if len(packets) < 2:
            return packets
        
        morphed_packets = packets.copy()
        base_time = morphed_packets[0].time
        current_time = base_time
        
        for i in range(1, len(morphed_packets)):
            # Calculate current IAT
            current_iat = morphed_packets[i].time - morphed_packets[i-1].time
            
            # Generate target IAT
            target_iat = max(0.001, self.np_random.normal(
                profile.avg_inter_arrival_time,
                profile.iat_std
            ))
            
            # Interpolate between current and target IAT
            new_iat = current_iat + strength * (target_iat - current_iat)
            new_iat = max(0.001, new_iat)  # Ensure minimum IAT
            
            current_time += new_iat
            morphed_packets[i].time = current_time
        
        return morphed_packets
    
    def _adjust_packet_count(self, packets: List[Packet], profile: TrafficProfile, 
                           strength: float) -> List[Packet]:
        """Adjust packet count based on profile distribution"""
        if strength < 0.5:  # Only apply significant changes at higher strength
            return packets
        
        current_count = len(packets)
        
        # Determine if we need to add or remove packets based on profile
        # This is a simplified approach
        size_factor = profile.avg_packet_size / 500.0  # Normalize around 500 bytes
        target_multiplier = 1.0 / size_factor  # More/fewer packets for smaller/larger sizes
        
        target_count = int(current_count * target_multiplier)
        target_count = max(1, min(target_count, current_count * 2))  # Reasonable bounds
        
        if target_count > current_count:
            # Add dummy packets
            return self._add_dummy_packets(packets, target_count - current_count, profile)
        elif target_count < current_count:
            # Remove some packets (randomly)
            indices_to_keep = sorted(self.random.sample(range(current_count), target_count))
            return [packets[i] for i in indices_to_keep]
        
        return packets
    
    def _add_dummy_packets(self, packets: List[Packet], num_to_add: int, 
                          profile: TrafficProfile) -> List[Packet]:
        """Add dummy packets to the flow"""
        if not packets or num_to_add <= 0:
            return packets
        
        extended_packets = packets.copy()
        
        for _ in range(num_to_add):
            # Choose a random position to insert dummy packet
            insert_pos = self.random.randint(0, len(extended_packets))
            
            # Create dummy packet based on profile
            if insert_pos > 0:
                base_packet = extended_packets[insert_pos - 1].copy()
            else:
                base_packet = extended_packets[0].copy()
            
            # Adjust dummy packet properties
            dummy_size = max(40, int(self.np_random.normal(
                profile.avg_packet_size, 
                profile.packet_size_std
            )))
            
            # Create dummy payload
            dummy_payload = b'\x00' * (dummy_size - len(base_packet) + len(base_packet[Raw].load) if Raw in base_packet else dummy_size - len(base_packet))
            
            if Raw in base_packet:
                base_packet[Raw].load = dummy_payload[:dummy_size - len(base_packet) + len(base_packet[Raw].load)]
            else:
                base_packet = base_packet / Raw(dummy_payload)
            
            # Adjust timestamp
            if insert_pos > 0:
                time_offset = self.np_random.exponential(profile.avg_inter_arrival_time)
                base_packet.time = extended_packets[insert_pos - 1].time + time_offset
            
            extended_packets.insert(insert_pos, base_packet)
        
        # Resort by time
        extended_packets.sort(key=lambda p: p.time)
        return extended_packets
    
    def _apply_burst_patterns(self, packets: List[Packet], profile: TrafficProfile, 
                            strength: float) -> List[Packet]:
        """Apply burst patterns to packet flow"""
        if strength < 0.3 or len(packets) < 2:
            return packets
        
        burst_size = profile.burst_characteristics.get('avg_burst_size', 3.0)
        burst_freq = profile.burst_characteristics.get('burst_frequency', 0.2)
        
        morphed_packets = packets.copy()
        
        # Apply burst timing pattern
        i = 0
        while i < len(morphed_packets) - 1:
            if self.random.random() < burst_freq * strength:
                # Create a burst starting at position i
                burst_length = max(1, int(self.np_random.poisson(burst_size)))
                burst_length = min(burst_length, len(morphed_packets) - i)
                
                # Compress timing within burst
                burst_duration = 0.01 * burst_length  # Very short burst duration
                base_time = morphed_packets[i].time
                
                for j in range(1, burst_length):
                    if i + j < len(morphed_packets):
                        morphed_packets[i + j].time = base_time + (j / burst_length) * burst_duration
                
                i += burst_length
            else:
                i += 1
        
        return morphed_packets
    
    def morph_to_popular_site(self, packets: List[Packet], site_type: str = 'social_media', 
                            strength: float = 0.7) -> List[Packet]:
        """Morph traffic to look like popular website categories"""
        profiles = self._get_popular_site_profiles()
        
        if site_type not in profiles:
            logger.warning(f"Unknown site type: {site_type}. Using 'general'")
            site_type = 'general'
        
        target_profile = profiles[site_type]
        return self.morph_to_profile(packets, target_profile, strength)
    
    def _get_popular_site_profiles(self) -> Dict[str, TrafficProfile]:
        """Get predefined profiles for popular website categories"""
        return {
            'social_media': TrafficProfile(
                avg_packet_size=800.0,
                packet_size_std=400.0,
                avg_inter_arrival_time=0.03,
                iat_std=0.02,
                packet_count_distribution={'small': 0.2, 'medium': 0.4, 'large': 0.4},
                burst_characteristics={'avg_burst_size': 5.0, 'burst_frequency': 0.3},
                directional_ratio=0.6
            ),
            'video_streaming': TrafficProfile(
                avg_packet_size=1200.0,
                packet_size_std=200.0,
                avg_inter_arrival_time=0.01,
                iat_std=0.005,
                packet_count_distribution={'small': 0.1, 'medium': 0.2, 'large': 0.7},
                burst_characteristics={'avg_burst_size': 10.0, 'burst_frequency': 0.4},
                directional_ratio=0.9
            ),
            'news_site': TrafficProfile(
                avg_packet_size=600.0,
                packet_size_std=300.0,
                avg_inter_arrival_time=0.05,
                iat_std=0.03,
                packet_count_distribution={'small': 0.3, 'medium': 0.5, 'large': 0.2},
                burst_characteristics={'avg_burst_size': 3.0, 'burst_frequency': 0.25},
                directional_ratio=0.3
            ),
            'e_commerce': TrafficProfile(
                avg_packet_size=700.0,
                packet_size_std=350.0,
                avg_inter_arrival_time=0.04,
                iat_std=0.025,
                packet_count_distribution={'small': 0.25, 'medium': 0.45, 'large': 0.3},
                burst_characteristics={'avg_burst_size': 4.0, 'burst_frequency': 0.2},
                directional_ratio=0.4
            ),
            'search_engine': TrafficProfile(
                avg_packet_size=400.0,
                packet_size_std=200.0,
                avg_inter_arrival_time=0.08,
                iat_std=0.04,
                packet_count_distribution={'small': 0.5, 'medium': 0.4, 'large': 0.1},
                burst_characteristics={'avg_burst_size': 2.0, 'burst_frequency': 0.15},
                directional_ratio=0.2
            ),
            'general': TrafficProfile(
                avg_packet_size=500.0,
                packet_size_std=250.0,
                avg_inter_arrival_time=0.05,
                iat_std=0.03,
                packet_count_distribution={'small': 0.35, 'medium': 0.4, 'large': 0.25},
                burst_characteristics={'avg_burst_size': 3.5, 'burst_frequency': 0.2},
                directional_ratio=0.4
            )
        }
    
    def create_decoy_traffic(self, base_packets: List[Packet], decoy_ratio: float = 0.3) -> List[Packet]:
        """Create decoy traffic to obfuscate real traffic patterns"""
        if not base_packets:
            return base_packets
        
        num_decoy = int(len(base_packets) * decoy_ratio)
        if num_decoy == 0:
            return base_packets
        
        decoy_packets = []
        time_span = base_packets[-1].time - base_packets[0].time
        
        for _ in range(num_decoy):
            # Create decoy packet
            base_packet = self.random.choice(base_packets).copy()
            
            # Random size
            decoy_size = self.random.randint(40, 1500)
            decoy_payload = b'\x00' * decoy_size
            
            if Raw in base_packet:
                base_packet[Raw].load = decoy_payload
            else:
                base_packet = base_packet / Raw(decoy_payload)
            
            # Random timestamp within the flow timespan
            base_packet.time = base_packets[0].time + self.random.uniform(0, time_span)
            
            decoy_packets.append(base_packet)
        
        # Combine and sort
        all_packets = base_packets + decoy_packets
        all_packets.sort(key=lambda p: p.time)
        
        logger.info(f"Added {num_decoy} decoy packets to {len(base_packets)} real packets")
        return all_packets

# Convenience functions
def create_traffic_profile(packets: List[Packet]) -> TrafficProfile:
    """Create traffic profile from packet list"""
    morpher = TrafficMorpher()
    return morpher.create_profile_from_packets(packets)

def morph_traffic(packets: List[Packet], target_profile: TrafficProfile, 
                 strength: float = 1.0) -> List[Packet]:
    """Morph traffic to target profile"""
    morpher = TrafficMorpher()
    return morpher.morph_to_profile(packets, target_profile, strength)

def morph_to_site_category(packets: List[Packet], category: str, 
                          strength: float = 0.7) -> List[Packet]:
    """Morph traffic to look like specific site category"""
    morpher = TrafficMorpher()
    return morpher.morph_to_popular_site(packets, category, strength)

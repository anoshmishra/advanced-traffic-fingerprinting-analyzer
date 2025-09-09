#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def load_config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def generate_realistic_features(n_samples_per_site=100):
    config = load_config()
    websites = config['collection']['target_websites']
    site_names = []
    for url in websites:
        site_name = url.split('//')[1].split('/')[0].replace('.', '_')
        site_names.append(site_name)
    all_features, all_labels = [], []
    np.random.seed(42)
    for site_idx, site_name in enumerate(site_names):
        print(f"Generating data for {site_name}...")
        for sample in range(n_samples_per_site):
            base_packets = 50 + site_idx * 20 + np.random.poisson(30)
            base_bytes = base_packets * (500 + site_idx * 100) + np.random.exponential(10000)
            features = {
                'total_packets': max(1, base_packets + np.random.normal(0, 10)),
                'outgoing_packets': None,
                'incoming_packets': None,
                'total_bytes': max(100, base_bytes + np.random.normal(0, 5000)),
                'mean_packet_size': None,
                'std_packet_size': np.random.exponential(200) + 50,
                'min_packet_size': max(40, np.random.normal(60, 20)),
                'max_packet_size': np.random.exponential(800) + 200,
                'flow_duration': max(0.1, np.random.exponential(5) + site_idx * 0.5),
                'mean_iat': max(0.001, np.random.exponential(0.1) + site_idx * 0.01),
                'std_iat': max(0, np.random.exponential(0.05)),
                'direction_changes': max(0, int(np.random.poisson(10) + site_idx * 2)),
            }
            total_packets = features['total_packets']
            out_ratio = 0.3 + site_idx * 0.05 + np.random.normal(0, 0.1)
            out_ratio = max(0.1, min(0.9, out_ratio))
            features['outgoing_packets'] = int(total_packets * out_ratio)
            features['incoming_packets'] = int(total_packets * (1 - out_ratio))
            features['mean_packet_size'] = features['total_bytes'] / total_packets
            size_base = features['mean_packet_size']
            for p in [25, 50, 75, 90, 95]:
                noise = np.random.normal(0, size_base * 0.2)
                percentile_multiplier = 0.5 + (p / 100) * 1.0
                features[f'size_p{p}'] = max(40, size_base * percentile_multiplier + noise)
            for i in range(10):
                size_variation = np.random.normal(size_base, size_base * 0.3)
                features[f'first_pkt_size_{i}'] = max(40, size_variation)
                features[f'first_pkt_dir_{i}'] = np.random.choice([-1, 1])
            features['num_bursts'] = max(1, int(np.random.poisson(5) + site_idx))
            features['avg_burst_size'] = max(1, np.random.exponential(8) + 2)
            features['max_burst_size'] = features['avg_burst_size'] + np.random.exponential(3)
            features['burst_bytes_ratio'] = min(1.0, max(0.1, np.random.beta(2, 3)))
            all_features.append(features)
            all_labels.append(site_name)
    return pd.DataFrame(all_features), pd.Series(all_labels)

def main():
    print("Generating sample traffic data...")
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/raw_pcaps').mkdir(parents=True, exist_ok=True)
    X, y = generate_realistic_features(n_samples_per_site=50)
    X = X.fillna(0)
    X.to_csv('data/processed/features.csv', index=False)
    y.to_csv('data/processed/labels.csv', index=False)
    print(f"Generated {len(X)} samples with {len(X.columns)} features")
    print(f"Sites: {sorted(y.unique())}")
    print(f"Samples per site: {y.value_counts().describe()}")
    print("\nSample feature statistics:")
    print(X.describe())
    print("\nData saved to:\n- data/processed/features.csv\n- data/processed/labels.csv")

if __name__ == "__main__":
    main()

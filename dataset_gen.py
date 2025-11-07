#!/usr/bin/env python3
"""
Synthetic Security Dataset Generator
Creates realistic home security event sequences with reasoning traces
"""

import json
import argparse
import random
from faker import Faker
from datetime import datetime, timedelta
import os

fake = Faker()

# Security event types and their characteristics
SECURITY_EVENTS = {
    'motion_detected': {
        'sensors': ['living_room_motion', 'hallway_motion', 'bedroom_motion', 'kitchen_motion'],
        'severity': ['low', 'medium'],
        'time_patterns': ['evening', 'night', 'early_morning']
    },
    'door_opened': {
        'sensors': ['front_door', 'back_door', 'garage_door', 'side_door'],
        'severity': ['medium', 'high'],
        'time_patterns': ['day', 'evening', 'night']
    },
    'window_broken': {
        'sensors': ['living_room_window', 'bedroom_window', 'kitchen_window'],
        'severity': ['high', 'critical'],
        'time_patterns': ['night', 'early_morning']
    },
    'glass_break_detected': {
        'sensors': ['glass_break_sensor'],
        'severity': ['high', 'critical'],
        'time_patterns': ['night', 'evening']
    },
    'smoke_detected': {
        'sensors': ['kitchen_smoke', 'living_room_smoke', 'bedroom_smoke'],
        'severity': ['high', 'critical'],
        'time_patterns': ['any']
    },
    'carbon_monoxide_detected': {
        'sensors': ['co_detector'],
        'severity': ['critical'],
        'time_patterns': ['any']
    },
    'temperature_anomaly': {
        'sensors': ['temp_sensor_1', 'temp_sensor_2', 'temp_sensor_3'],
        'severity': ['low', 'medium'],
        'time_patterns': ['any']
    },
    'camera_motion': {
        'sensors': ['front_cam', 'back_cam', 'garage_cam', 'driveway_cam'],
        'severity': ['low', 'medium', 'high'],
        'time_patterns': ['day', 'evening', 'night']
    }
}

# Reasoning templates for different scenarios
REASONING_TEMPLATES = {
    'false_positive': [
        "Motion detected in {location} at {time}. Analysis shows pet movement patterns typical for household. No human presence detected. Confidence: low threat.",
        "Door sensor triggered at {time}. Cross-referenced with family schedule - expected arrival. No unusual activity detected. Normal operation.",
        "Temperature anomaly in {location} at {time}. HVAC system cycling detected. No security implications. System operating normally."
    ],
    'potential_threat': [
        "Motion detected in {location} at {time}. Unusual pattern - no scheduled occupancy. Cross-check with camera shows unknown individual. Recommend verification.",
        "Multiple door sensors triggered within {timeframe} seconds at {time}. Coordinated entry attempt detected. Alert authorities recommended.",
        "Glass break detected at {location} at {time}. No authorized entry scheduled. High probability of forced entry attempt. Immediate response required."
    ],
    'environmental': [
        "Smoke detected in {location} at {time}. Temperature rising rapidly. Potential fire hazard. Evacuation and emergency services recommended.",
        "Carbon monoxide levels rising at {time}. Immediate evacuation required. Ventilation systems activated. Emergency services notified.",
        "Temperature anomaly detected system-wide at {time}. HVAC malfunction suspected. Maintenance required but no immediate security threat."
    ],
    'maintenance': [
        "Sensor {sensor} offline at {time}. Battery replacement or maintenance required. System redundancy active. No security gap.",
        "Camera {camera} night vision malfunction at {time}. Daytime operation normal. Schedule maintenance for next service window.",
        "Network latency detected across sensors at {time}. System performance degraded but functional. Network optimization recommended."
    ]
}

def generate_event_sequence():
    """Generate a realistic sequence of security events"""
    num_events = random.randint(1, 5)
    events = []
    base_time = fake.date_time_this_year()
    
    for i in range(num_events):
        event_type = random.choice(list(SECURITY_EVENTS.keys()))
        event_config = SECURITY_EVENTS[event_type]
        
        # Generate realistic timing
        if i > 0:
            time_offset = random.randint(1, 300)  # 1-5 minutes between events
            event_time = base_time + timedelta(seconds=time_offset)
        else:
            event_time = base_time
        
        sensor = random.choice(event_config['sensors'])
        severity = random.choice(event_config['severity'])
        
        event = {
            'timestamp': event_time.isoformat(),
            'event_type': event_type,
            'sensor': sensor,
            'severity': severity,
            'confidence': round(random.uniform(0.7, 0.99), 2),
            'metadata': {
                'location': extract_location_from_sensor(sensor),
                'device_id': f"DEV_{random.randint(1000, 9999)}",
                'battery_level': random.randint(20, 100)
            }
        }
        events.append(event)
    
    return events

def extract_location_from_sensor(sensor):
    """Extract human-readable location from sensor name"""
    location_map = {
        'living_room': 'Living Room',
        'hallway': 'Hallway', 
        'bedroom': 'Bedroom',
        'kitchen': 'Kitchen',
        'front_door': 'Front Entrance',
        'back_door': 'Back Entrance',
        'garage_door': 'Garage',
        'side_door': 'Side Entrance',
        'front_cam': 'Front Camera',
        'back_cam': 'Back Camera',
        'garage_cam': 'Garage Camera',
        'driveway_cam': 'Driveway Camera'
    }
    
    for key, value in location_map.items():
        if key in sensor:
            return value
    return sensor.replace('_', ' ').title()

def generate_reasoning(events):
    """Generate reasoning trace for the event sequence"""
    if not events:
        return "No events detected. System operating normally."
    
    # Determine scenario type based on events
    max_severity = max(event['severity'] for event in events)
    
    if max_severity in ['low', 'medium'] and random.random() < 0.7:
        scenario = 'false_positive'
    elif max_severity in ['high', 'critical']:
        scenario = 'potential_threat' if 'motion' in events[0]['event_type'] or 'door' in events[0]['event_type'] else 'environmental'
    else:
        scenario = random.choice(['false_positive', 'potential_threat', 'environmental', 'maintenance'])
    
    # Select template and fill with event details
    template = random.choice(REASONING_TEMPLATES[scenario])
    
    # Extract key details for template filling
    primary_event = events[0]
    location = extract_location_from_sensor(primary_event['sensor'])
    time = primary_event['timestamp'].split('T')[1][:5]  # HH:MM format
    
    reasoning = template.format(
        location=location,
        time=time,
        timeframe=str(random.randint(5, 60)),
        sensor=primary_event['sensor']
    )
    
    # Add chain-of-thought analysis
    cot_analysis = []
    cot_analysis.append(f"Analyzing {len(events)} security events...")
    cot_analysis.append(f"Primary event: {primary_event['event_type']} at {location}")
    cot_analysis.append(f"Severity assessment: {max_severity}")
    cot_analysis.append(f"Time pattern analysis: {primary_event['timestamp']}")
    
    if len(events) > 1:
        cot_analysis.append(f"Event correlation detected across {len(set(e['sensor'] for e in events))} sensors")
    
    cot_analysis.append(f"Conclusion: {reasoning}")
    
    return "\n".join(cot_analysis)

def generate_training_sample():
    """Generate a single training sample with events and reasoning"""
    events = generate_event_sequence()
    reasoning = generate_reasoning(events)
    
    # Determine if response is needed
    max_severity = max([e['severity'] for e in events]) if events else 'low'
    needs_response = max_severity in ['high', 'critical']
    
    # Generate appropriate response
    if needs_response:
        response = random.choice([
            "Emergency services dispatched. Homeowner notified. All systems on high alert.",
            "Security team mobilized. Perimeter secured. Authorities contacted.",
            "Evacuation protocol initiated. Fire department notified. Safety systems activated."
        ])
    else:
        response = random.choice([
            "Event logged. No immediate action required. Continued monitoring.",
            "System operating normally. Event documented for pattern analysis.",
            "Routine event detected. No security implications. Standard monitoring continues."
        ])
    
    return {
        "events": events,
        "reasoning": reasoning,
        "response": response,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "sample_id": f"SAMPLE_{random.randint(100000, 999999)}",
            "scenario_type": "security_event_analysis"
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic security dataset")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/unified_security_dataset.jsonl", help="Output file path")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"Generating {args.samples} synthetic security samples...")
    
    with open(args.output, 'w') as f:
        for i in range(args.samples):
            if i % 1000 == 0:
                print(f"Generated {i} samples...")
            
            sample = generate_training_sample()
            f.write(json.dumps(sample) + '\n')
    
    print(f"Dataset generation complete. Saved to {args.output}")
    print(f"Generated {args.samples} training samples")

if __name__ == "__main__":
    main()

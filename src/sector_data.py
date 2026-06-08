"""
Static sector data for the Quantum Computing universe.

Defines technology clusters, revenue exposures, competitive relationships,
and company profiles used by both the model (for signal generation) and
the frontend (for the sector map visualization).
"""

# All tickers in our quantum computing universe
QUANTUM_UNIVERSE = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON", "NVDA"]

# Technology clusters
TECHNOLOGY_CLUSTERS = {
    "trapped_ion": {
        "name": "Trapped Ion",
        "description": "Uses individual ions held in electromagnetic traps as qubits. Known for high gate fidelity and all-to-all connectivity.",
        "companies": ["IONQ", "HON"],
    },
    "superconducting": {
        "name": "Superconducting",
        "description": "Uses superconducting circuits cooled to near absolute zero. Fastest gate speeds but limited connectivity.",
        "companies": ["RGTI", "IBM", "GOOGL"],
    },
    "quantum_annealing": {
        "name": "Quantum Annealing",
        "description": "Specialized approach for optimization problems. Not gate-based. Limited to specific problem types.",
        "companies": ["QBTS"],
    },
    "topological": {
        "name": "Topological",
        "description": "Uses exotic quasiparticles (Majorana fermions) for inherently error-resistant qubits. Still early stage.",
        "companies": ["MSFT"],
    },
    "neutral_atom": {
        "name": "Neutral Atom",
        "description": "Uses arrays of neutral atoms held by optical tweezers. Highly scalable architecture.",
        "companies": ["QUBT"],
    },
    "adjacent": {
        "name": "Adjacent / Enablers",
        "description": "Companies that benefit from quantum computing growth without building quantum hardware directly.",
        "companies": ["NVDA"],
    },
}

# Revenue exposure to quantum computing (estimated percentage of total revenue)
# This determines how much a quantum-specific event should affect the stock price
REVENUE_EXPOSURE = {
    "IONQ": 1.00,   # 100% quantum computing
    "RGTI": 1.00,   # 100% quantum computing
    "QBTS": 1.00,   # 100% quantum computing (D-Wave)
    "QUBT": 1.00,   # 100% quantum computing
    "HON":  0.05,   # ~5% (Quantinuum subsidiary)
    "IBM":  0.02,   # ~2% (Quantum division is small relative to total)
    "GOOGL": 0.001, # <0.1% (Quantum AI is tiny vs advertising revenue)
    "MSFT": 0.001,  # <0.1% (Azure Quantum is tiny vs cloud/Office)
    "NVDA": 0.01,   # ~1% (HPC/simulation hardware sold to quantum companies)
}

# Competitive relationships: which companies compete directly
# Format: {ticker: [list of direct competitors]}
COMPETITIVE_RELATIONSHIPS = {
    "IONQ": ["HON", "RGTI", "IBM", "GOOGL"],  # Competes with all gate-based approaches
    "RGTI": ["IBM", "GOOGL", "IONQ", "HON"],  # Superconducting competitors + trapped-ion
    "QBTS": ["IONQ", "RGTI", "IBM"],          # Annealing competes with gate-based for optimization
    "QUBT": ["IONQ", "HON"],                   # Neutral atom competes most with trapped-ion (scalability)
    "IBM":  ["RGTI", "GOOGL", "IONQ"],        # Superconducting peers
    "GOOGL": ["IBM", "RGTI"],                  # Superconducting peers
    "MSFT": [],                                # Topological is orthogonal, minimal direct competition
    "HON":  ["IONQ", "RGTI", "IBM"],          # Quantinuum competes broadly
    "NVDA": [],                                # Enabler, not direct competitor
}

# Company profiles
COMPANY_PROFILES = {
    "IONQ": {
        "name": "IonQ",
        "full_name": "IonQ, Inc.",
        "technology": "trapped_ion",
        "description": "Leading pure-play trapped-ion quantum computing company. Known for high qubit fidelity and algorithmic qubit metric.",
        "key_metric": "Algorithmic qubits (#AQ)",
        "market_cap_approx": "$8B",
    },
    "RGTI": {
        "name": "Rigetti",
        "full_name": "Rigetti Computing, Inc.",
        "technology": "superconducting",
        "description": "Superconducting quantum computing company focused on hybrid classical-quantum systems. Vertically integrated (designs and fabricates own chips).",
        "key_metric": "Qubit count and gate fidelity",
        "market_cap_approx": "$3B",
    },
    "QBTS": {
        "name": "D-Wave",
        "full_name": "D-Wave Quantum Inc.",
        "technology": "quantum_annealing",
        "description": "Pioneer in quantum annealing. Only company with commercial quantum annealing systems. Also developing gate-based systems.",
        "key_metric": "Qubit count (5000+ annealing qubits)",
        "market_cap_approx": "$2B",
    },
    "QUBT": {
        "name": "Quantum Computing Inc.",
        "full_name": "Quantum Computing Inc.",
        "technology": "neutral_atom",
        "description": "Developing photonic and neutral-atom quantum computing solutions for optimization and machine learning applications.",
        "key_metric": "Optimization problem size",
        "market_cap_approx": "$1B",
    },
    "IBM": {
        "name": "IBM",
        "full_name": "International Business Machines",
        "technology": "superconducting",
        "description": "Major technology company with a significant quantum computing division. Operates IBM Quantum Network with 100+ enterprise clients.",
        "key_metric": "Quantum Volume, Eagle/Condor processors",
        "market_cap_approx": "$200B",
    },
    "GOOGL": {
        "name": "Google",
        "full_name": "Alphabet Inc.",
        "technology": "superconducting",
        "description": "Google Quantum AI division. Achieved quantum supremacy (2019) and below-threshold error correction (2024, Willow processor).",
        "key_metric": "Error correction threshold, Sycamore/Willow processors",
        "market_cap_approx": "$2.1T",
    },
    "MSFT": {
        "name": "Microsoft",
        "full_name": "Microsoft Corporation",
        "technology": "topological",
        "description": "Pursuing topological qubits (Majorana-based). Also operates Azure Quantum cloud platform offering access to third-party quantum hardware.",
        "key_metric": "Topological qubit demonstration",
        "market_cap_approx": "$3.3T",
    },
    "HON": {
        "name": "Honeywell (Quantinuum)",
        "full_name": "Honeywell International Inc.",
        "technology": "trapped_ion",
        "description": "Owns Quantinuum, a leading trapped-ion quantum computing company. Quantinuum has the highest quantum volume of any commercial system.",
        "key_metric": "Quantum Volume (record holder)",
        "market_cap_approx": "$150B",
    },
    "NVDA": {
        "name": "NVIDIA",
        "full_name": "NVIDIA Corporation",
        "technology": "adjacent",
        "description": "Provides GPU hardware used for quantum circuit simulation, hybrid quantum-classical algorithms, and HPC workloads adjacent to quantum.",
        "key_metric": "cuQuantum SDK adoption",
        "market_cap_approx": "$3.0T",
    },
}

# Signal propagation rules: how different event types affect each technology cluster
# These encode the domain knowledge about competitive dynamics
SIGNAL_PROPAGATION = {
    "trapped_ion_breakthrough": {
        "trapped_ion": "strongly_positive",
        "superconducting": "slightly_negative",
        "quantum_annealing": "slightly_negative",
        "topological": "neutral",
        "neutral_atom": "slightly_negative",
        "adjacent": "slightly_positive",
    },
    "superconducting_breakthrough": {
        "trapped_ion": "slightly_negative",
        "superconducting": "strongly_positive",
        "quantum_annealing": "slightly_negative",
        "topological": "neutral",
        "neutral_atom": "slightly_negative",
        "adjacent": "slightly_positive",
    },
    "error_correction_general": {
        "trapped_ion": "positive",
        "superconducting": "positive",
        "quantum_annealing": "neutral",
        "topological": "positive",
        "neutral_atom": "positive",
        "adjacent": "positive",
    },
    "government_funding": {
        "trapped_ion": "positive",
        "superconducting": "positive",
        "quantum_annealing": "positive",
        "topological": "positive",
        "neutral_atom": "positive",
        "adjacent": "slightly_positive",
    },
    "sector_negative_macro": {
        "trapped_ion": "negative",
        "superconducting": "negative",
        "quantum_annealing": "negative",
        "topological": "slightly_negative",
        "neutral_atom": "negative",
        "adjacent": "slightly_negative",
    },
}

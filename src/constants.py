"""Static dictionaries and mappings used across the pipeline."""

THREATS = [
    "Account Hijacking", "Adversarial Attack", "APT", "Backdoor", "Botnet",
    "Brute Force Attack", "Cryptojacking", "DDoS", "Data Poisoning", "Deepfake",
    "Disinformation", "DNS Spoofing", "Dropper", "Insider Threat",
    "IoT Device Attack", "Malware", "MITM", "Password Attack", "Phishing",
    "Ransomware", "Session Hijacking", "Supply Chain Attack", "Targeted Attack",
    "Trojan", "Vulnerability", "Zero-day",
]

THREAT_TYPE = {
    "Account Hijacking": "RI", "Adversarial Attack": "E", "APT": "RI", "Backdoor": "RI",
    "Botnet": "RI", "Brute Force Attack": "RI", "Cryptojacking": "E", "DDoS": "RI",
    "Data Poisoning": "E", "Deepfake": "E", "Disinformation": "RI", "DNS Spoofing": "RI",
    "Dropper": "RI", "Insider Threat": "RI", "IoT Device Attack": "E", "Malware": "RI",
    "MITM": "RI", "Password Attack": "RI", "Phishing": "RI", "Ransomware": "E",
    "Session Hijacking": "RI", "Supply Chain Attack": "RI", "Targeted Attack": "RI",
    "Trojan": "RI", "Vulnerability": "RI", "Zero-day": "RI",
}

THREAT_PAT_MAP = {
    "Account Hijacking": ["AC", "AD", "CAPTCHA", "CR", "IDS/IPS", "IdM", "LP", "MFA", "ML/DL", "NLP/LLM", "PT", "SM"],
    "Adversarial Attack": ["AD", "AdT", "BN", "DA", "DD", "DP", "DR", "DS", "ML/DL", "NI", "NLP/LLM", "OD", "RRAM", "SS", "TAI"],
    "APT": ["AC", "DLP", "DRM", "DT", "GT", "IDS/IPS", "LP", "MFA", "ML/DL", "NLP/LLM", "NS", "PT", "RA", "UBA"],
    "Backdoor": ["AD", "DAS", "IDS/IPS", "ML/DL", "PT", "SA"],
    "Botnet": ["AD", "BC", "BH", "BT", "CAPTCHA", "GM", "GT", "HP", "IDS/IPS", "ML/DL", "NLP/LLM", "PF", "PT", "RC", "RL", "SDN", "TS"],
    "Brute Force Attack": ["CAPTCHA", "CR", "DBI", "IDS/IPS", "MFA", "ML/DL", "OTP", "PH", "PT"],
    "Cryptojacking": ["BT", "ML/DL", "PT", "TA"],
    "DDoS": ["BC", "BH", "BT", "IDS/IPS", "ML/DL", "NLP/LLM", "PF", "PT", "RC", "RL", "TS"],
    "Data Poisoning": ["AD", "AdT", "BN", "DP", "DS", "ML/DL", "NLP/LLM", "OD", "TAI"],
    "Deepfake": ["3DFR", "AD", "BO", "DW", "LD", "ML/DL", "NLP/LLM"],
    "Disinformation": ["BC", "CA", "DLT", "DP", "DT", "GT", "HG", "IR", "ML/DL", "NLP/LLM", "SI"],
    "DNS Spoofing": ["BC", "CR", "DNSSEC", "ML/DL", "PT", "RA"],
    "Dropper": ["AW", "CS", "FIM", "IDS/IPS", "ML/DL", "NLP/LLM", "PT", "SBX"],
    "Insider Threat": ["AC", "AD", "AM", "AT", "CR", "DLD", "IDS/IPS", "KD", "LP", "ML/DL", "MTD", "NLP/LLM", "PT", "UBA"],
    "IoT Device Attack": ["AD", "BC", "CR", "IDS/IPS", "IdM", "MFA", "ML/DL", "MS", "PT", "SB"],
    "Malware": ["AC", "AD", "AW", "BBD", "BC", "CR", "CS", "DAS", "DB", "DM", "DT", "FIM", "FV", "GT", "HP", "IDS/IPS", "ML/DL", "NLP/LLM", "PMT", "PT", "SA", "SB", "SBX", "SHMM", "SMF", "VK"],
    "MITM": ["BC", "CAPTCHA", "CP", "CR", "ML/DL", "PKI", "PT", "SSL/TLS", "SSP", "VPN"],
    "Password Attack": ["CAPTCHA", "CR", "GA", "IDS/IPS", "MA", "MFA", "ML/DL", "NLP/LLM", "OTP", "PH", "PM", "PP", "PSM", "PT"],
    "Phishing": ["AC", "BT", "CR", "DT", "MA", "MFA", "ML/DL", "NLP/LLM", "PKI"],
    "Ransomware": ["AC", "AD", "AW", "BC", "CR", "DAS", "DB", "DT", "IDS/IPS", "ML/DL", "NLP/LLM", "PMT", "PT", "SA", "SHMM"],
    "Session Hijacking": ["AD", "CA", "CR", "Https", "IBE", "ML/DL", "PT", "SAT", "SM", "SSL/TLS"],
    "Supply Chain Attack": ["AC", "AD", "BC", "CR", "IdM", "ML/DL", "NLP/LLM", "PT", "SCRM"],
    "Targeted Attack": ["AC", "DRM", "DT", "GT", "IDS/IPS", "LP", "MFA", "ML/DL", "NLP/LLM", "NS", "PT", "RA", "UBA"],
    "Trojan": ["AD", "BBD", "CR", "FV", "GT", "IDS/IPS", "ML/DL", "NLP/LLM", "PT", "SMF"],
    "Vulnerability": ["CFI", "IDS/IPS", "ML/DL", "NLP/LLM", "PMT", "PT", "SC", "SIEM", "VA", "VM", "VS"],
    "Zero-day": ["AD", "DT", "FIM", "GT", "IDS/IPS", "ML/DL", "NLP/LLM", "PrP", "VM", "VPN"],
}

PAT_CODES = sorted({pat for pats in THREAT_PAT_MAP.values() for pat in pats})

COUNTRIES_36 = [
    "US", "UK", "DE", "FR", "ES", "IT", "NL", "SE", "NO", "FI",
    "PL", "IE", "CH", "AT", "BE", "PT", "RU", "UA", "TR", "IN",
    "CN", "JP", "KR", "SG", "AU", "NZ", "CA", "MX", "BR", "AR",
    "ZA", "EG", "NG", "AE", "SA", "IL",
]

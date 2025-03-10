import pandas as pd
import re
import tldextract
import whois
import requests
from urllib.parse import urlparse
from datetime import datetime
from collections import Counter
from math import log2

def calculate_entropy(url):
    counts = Counter(url)
    length = len(url)
    entropy = -sum((count/length) * log2(count/length) for count in counts.values())
    return entropy


def extract_features(url):
    features = {}
    
    # Parse URL
    extracted = tldextract.extract(url)
    parsed_url = urlparse(url)

    # 📌 Lexical Features
    features["url_length"] = len(url)
    features["num_dots"] = url.count(".")
    features["num_special_chars"] = sum(url.count(c) for c in ['@', '-', '_', '?', '&', '=', '%', '#', '/'])
    features["has_at_symbol"] = 1 if "@" in url else 0
    features["has_hyphen"] = 1 if "-" in extracted.domain else 0
    features["is_ip"] = 1 if re.match(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$", parsed_url.netloc) else 0
    features["has_redirect"] = 1 if url[url.find("//", 8)+2:].count("//") > 0 else 0
    features["num_subdomains"] = extracted.subdomain.count(".") if extracted.subdomain else 0
    features["is_shortened"] = 1 if extracted.registered_domain in ["bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd"] else 0
    features["url_entropy"] = calculate_entropy(url)

    suspicious_words = ["login", "verify", "bank", "secure", "update", "account", "payment"]
    features["has_suspicious_keyword"] = 1 if any(word in url.lower() for word in suspicious_words) else 0

    popular_brands = ["google", "facebook", "paypal", "amazon", "bank"]
    features["is_brand_spoofed"] = 1 if any(brand in url.lower() and brand not in extracted.domain for brand in popular_brands) else 0

    # 📌 Domain-Based Features
    domain = extracted.registered_domain
    try:
        domain_info = whois.whois(domain)
        if isinstance(domain_info.creation_date, list):
            creation_date = domain_info.creation_date[0]
        else:
            creation_date = domain_info.creation_date

        if isinstance(domain_info.expiration_date, list):
            expiry_date = domain_info.expiration_date[0]
        else:
            expiry_date = domain_info.expiration_date

        features["domain_age"] = (datetime.now() - creation_date).days if creation_date else -1
        features["domain_expiry"] = (expiry_date - datetime.now()).days if expiry_date else -1
        features["whois_private"] = 1 if domain_info.registrant_name is None else 0
    except:
        features["domain_age"] = -1
        features["domain_expiry"] = -1
        features["whois_private"] = -1

    # 📌 Host-Based Features
    try:
        response = requests.get(url, timeout=5)
        features["https_used"] = 1 if parsed_url.scheme == "https" else 0
    except:
        features["https_used"] = -1

    return features

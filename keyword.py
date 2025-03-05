import json
import re
from tqdm import tqdm
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
keywords_dict = {
    "Race Condition": ['race', 'racy'],
    "Crash": ['crash', 'exception'],
    "Resource Leak": ['leak'],
    "Integer Overflow": ['integer', 'overflow', 'signedness', 'widthness', 'underflow'],
    "Improper Access": ['improper', 'unauthenticated', 'gain access', 'permission', 'hijack', 'authenticate', 'privilege', 'forensic', 'hacker', 'root', 'URL', 'form', 'field', 'sensitive'],
    "Buffer Overflow": ['buffer', 'overflow', 'stack', 'strcpy', 'strcat', 'strtok', 'gets', 'makepath', 'splitpath', 'heap', 'strlen', 'out of memory'],
    "Denial of Service (DoS)": ['denial service', 'dos', 'ddos'],
    "Deadlock": ['deadlock'],
    "Encryption": ['encrypt', 'decrypt', 'password', 'cipher', 'trust', 'checksum', 'nonce', 'salt', 'crypto', 'mismatch'],
    "Cross Site Scripting (XSS)": ['cross site', 'CSS', 'XSS', 'malform', 'htmlspecialchar'],
    "Use After Free": ['use-after-free', 'dynamic'],
    "Command Injection": ['command', 'exec'],
    "Cross Site Request Forgery": ['cross site', 'request forgery', 'CSRF', 'XSRF', 'forged', 'cookie', 'xhttp'],
    "Format String": ['format', 'string', 'printf', 'scanf', 'sanitize'],
    "SQL Injection": ['SQL', 'SQLI', 'injection', 'ondelete'],
    "Common Keywords": ['security', 'vulnerability', 'vulnerable', 'hole', 'exploit', 'attack', 'bypass', 'backdoor', 'threat', 'expose', 'breach', 'violate', 'fatal', 'blacklist', 'overrun', 'insecure', 'scare', 'scary', 'conflict', 'trojan', 'firewall', 'spyware', 'adware', 'virus', 'ransom', 'malware', 'malicious', 'risk', 'dangling', 'unsafe', 'steal', 'worm', 'phishing', 'cve', 'cwe', 'collusion', 'covert', 'mitm', 'sniffer', 'quarantine', 'scam', 'spam', 'spoof', 'tamper', 'zombie', 'cast', 'xml']
}

nltk.download('punkt', quiet=True)

input_file_path = ''
output_file_path = ''

matched_records = []
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def match_keywords(text, keywords_dict):

    processed_text = preprocess_text(text)
    matches = {}
    
    for category, keywords in keywords_dict.items():
        processed_keywords = [preprocess_text(keyword) for keyword in keywords]
        matched_keywords = []

        for i, keyword in enumerate(keywords):
            processed_keyword = processed_keywords[i]
            if re.search(r'\b' + re.escape(processed_keyword) + r'\b', processed_text):
                matched_keywords.append(keyword)
        
        if matched_keywords:
            matches[category] = matched_keywords
    return matches

with open(input_file_path, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

for record in tqdm(data, desc="Processing"):
    combined_text = f"{record.get('msg', '')}"
    matches = match_keywords(combined_text, keywords_dict)
    if matches:
        record['matched_keywords'] = matches
        if 'oldf' in record:
            del record['oldf']
        matched_records.append(record)

with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(matched_records, outfile, indent=4)


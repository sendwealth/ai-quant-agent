#!/usr/bin/env python3
"""
Minimal email test for ai-quant-agent
"""

import smtplib
from email.mime.text import MIMEText
from pathlib import Path

def read_env_file():
    """Read .env file"""
    env_path = Path('.env')
    config = {}
    
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    
    return config

def send_minimal_email():
    """Send minimal test email"""
    
    config = read_env_file()
    
    smtp_server = config.get('EMAIL_SMTP_SERVER', 'smtp.163.com')
    smtp_port = int(config.get('EMAIL_SMTP_PORT', 465))
    sender_email = config.get('EMAIL_SENDER', '')
    sender_password = config.get('EMAIL_PASSWORD', '')
    receiver_email = config.get('EMAIL_RECIPIENTS', 'sendwealth@163.com')
    
    print(f"Config: {smtp_server}:{smtp_port}")
    print(f"Sender: {sender_email}")
    print(f"Receiver: {receiver_email}")
    
    # Test with minimal ASCII content only
    subject = "Test Report"
    body = "Test email from ai-quant-agent project. System check completed."
    
    try:
        msg = MIMEText(body, 'plain', 'ascii')
        msg['From'] = sender_email
        msg['To'] = receiver_email  
        msg['Subject'] = subject
        
        print("Connecting...")
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        print("SUCCESS: Minimal email sent!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == '__main__':
    print("Minimal Email Test")
    print("-" * 30)
    
    success = send_minimal_email()
    
    if success:
        print("\nSUCCESS: Email test passed!")
        print("=" * 30)
    else:
        print("\nERROR: Email test failed!")
        print("=" * 30)